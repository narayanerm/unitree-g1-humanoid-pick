###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""PPO training loop with shape-filtered warm-start, progressive unfreeze, and trunk-frozen diagnostic.

Input:  task_id, cfg (num_envs, max_iters, ...), log_dir, optional warm_start_ckpt or resume_ckpt.
Output: checkpoints + TensorBoard events written under log_dir/; returns None.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch


def warm_start_shape_filtered(runner, ckpt_path: str | Path) -> None:
    """Load actor and critic weights from a checkpoint, copying only shape-matching tensors.
    Args: runner — mjlab runner; ckpt_path — source checkpoint path.
    Returns: none.
    """
    loaded = torch.load(str(ckpt_path), map_location=runner.device, weights_only=False)

    def _apply(source_sd: dict, dest_module, tag: str) -> None:
        """Copy tensors from source_sd into dest_module when shapes match.
        Args: source_sd — source state dict; dest_module — target nn.Module; tag — log label.
        Returns: none.
        """
        dest_sd = dest_module.state_dict()
        keep, skip = {}, []
        for k, v in source_sd.items():
            if k in dest_sd and dest_sd[k].shape == v.shape:
                keep[k] = v
            else:
                reason = "no dest" if k not in dest_sd else f"{tuple(v.shape)} -> {tuple(dest_sd[k].shape)}"
                skip.append(f"{k}  [{reason}]")
        dest_module.load_state_dict(keep, strict=False)
        print(f"[warm-start] {tag:6s}: loaded {len(keep)}/{len(source_sd)} tensors")
        for s in skip[:6]:
            print(f"             skip: {s}")
        if len(skip) > 6:
            print(f"             ... +{len(skip) - 6} more skipped")

    # rsl-rl 4.x to 5.x key renames matching mjlab's runner.load.
    actor_sd = loaded.get("actor_state_dict", {})
    if "log_std" in actor_sd:
        actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")
    if "std" in actor_sd:
        actor_sd["distribution.std_param"] = actor_sd.pop("std")

    if actor_sd:
        _apply(actor_sd, runner.alg.actor, "actor")
    critic_sd = loaded.get("critic_state_dict", {})
    if critic_sd:
        _apply(critic_sd, runner.alg.critic, "critic")


def _mean_reward(runner) -> Optional[float]:
    """Compute mean of the rsl_rl 100-episode reward buffer.
    Args: runner — mjlab runner exposing logger.rewbuffer.
    Returns: mean reward as float, or None if the buffer is empty.
    """
    buf = getattr(getattr(runner, "logger", None), "rewbuffer", None)
    if buf is None or len(buf) == 0:
        return None
    return float(sum(buf) / len(buf))


def _train_with_early_stop(
    runner,
    max_iters: int,
    patience: int,
    window: int,
    min_improve: float,
) -> None:
    """Train iteration-by-iteration and stop when the moving-average reward plateaus.
    Args: runner — mjlab runner; max_iters — cap on iterations; patience — stagnant iters tolerated; window — moving-average length; min_improve — relative gain threshold.
    Returns: none.
    """
    history: list[float] = []
    best_avg = float("-inf")
    stagnant = 0
    for it in range(max_iters):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 0))
        mr = _mean_reward(runner)
        if mr is None:
            continue
        history.append(mr)
        if len(history) < window:
            continue
        current_avg = sum(history[-window:]) / window
        if current_avg > best_avg * (1.0 + min_improve):
            best_avg = current_avg
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= patience:
            print(f"[early-stop] plateau at iter {it + 1}/{max_iters}: "
                  f"avg(last {window})={current_avg:.2f}, best={best_avg:.2f}, stagnant={stagnant} iters.")
            break


def _freeze_actor(runner, freeze: bool) -> None:
    """Toggle requires_grad on every actor parameter (legacy critic-warmup path).
    Args: runner — mjlab runner; freeze — True to freeze, False to unfreeze.
    Returns: none.
    """
    for p in runner.alg.actor.parameters():
        p.requires_grad_(not freeze)


# Progressive unfreeze schedule (Howard & Ruder 2018): release actor layers gradually.
_DEFAULT_UNFREEZE_SCHEDULE = {
    0:  {0},
    30: {0, 2},
    60: {0, 2, 4},
    90: {0, 2, 4, 6},
}


def _set_actor_layers_frozen(runner, unfrozen_layers: set[int]) -> None:
    """Freeze actor.mlp layers outside unfrozen_layers while keeping normalizer and distribution trainable.
    Args: runner — mjlab runner; unfrozen_layers — set of mlp indices to keep trainable.
    Returns: none.
    """
    for name, p in runner.alg.actor.named_parameters():
        m = re.match(r"mlp\.(\d+)\.", name)
        if m:
            p.requires_grad_(int(m.group(1)) in unfrozen_layers)
        else:
            p.requires_grad_(True)


def _run_progressive_unfreeze(
    runner,
    max_iters: int,
    schedule: dict[int, set[int]] | None = None,
) -> None:
    """Run PPO while progressively unfreezing actor layers per a step schedule.
    Args: runner — mjlab runner; max_iters — total learning iterations; schedule — {start_iter: unfrozen mlp indices} (default if None).
    Returns: none.
    """
    schedule = schedule or _DEFAULT_UNFREEZE_SCHEDULE
    if 0 not in schedule:
        raise ValueError("Progressive schedule must include iter 0.")
    checkpoints = sorted(schedule.keys())

    trained = 0
    for i, iter_start in enumerate(checkpoints):
        unfrozen = schedule[iter_start]
        _set_actor_layers_frozen(runner, unfrozen)
        iter_end = min(checkpoints[i + 1], max_iters) if i + 1 < len(checkpoints) else max_iters
        chunk = iter_end - iter_start
        if chunk <= 0:
            break
        print(f"[unfreeze] iter {iter_start}-{iter_end-1}: trainable Linear layers = {sorted(unfrozen)}")
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(trained == 0))
        trained += chunk
        if trained >= max_iters:
            break

    _set_actor_layers_frozen(runner, {0, 2, 4, 6})


def _apply_trunk_frozen_diagnostic(runner) -> None:
    """Freeze walk-trunk layers (mlp.2, mlp.4) and swap the head for Linear->ELU->Linear.
    Args: runner — mjlab runner whose actor has an mlp trunk of depth >= 6.
    Returns: none.
    """
    import torch.nn as nn

    actor = runner.alg.actor
    for name, p in actor.named_parameters():
        m = re.match(r"mlp\.(\d+)\.", name)
        if m and int(m.group(1)) in {2, 4}:
            p.requires_grad_(False)

    old_head = actor.mlp[6]
    in_dim = old_head.in_features
    out_dim = old_head.out_features
    device = old_head.weight.device
    dtype = old_head.weight.dtype
    new_head = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ELU(),
        nn.Linear(64, out_dim),
    ).to(device=device, dtype=dtype)
    actor.mlp[6] = new_head

    # Rebuild optimizer so the new head's parameters are registered.
    lr = runner.alg.optimizer.param_groups[0]["lr"]
    runner.alg.optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(runner.alg.critic.parameters()),
        lr=lr,
    )

    trainable = [n for n, p in actor.named_parameters() if p.requires_grad]
    print(f"[diagnostic] trunk (mlp.2, mlp.4) frozen; mlp.6 replaced by ({in_dim}->64->{out_dim}).")
    print(f"[diagnostic] actor trainable tensors: {trainable}")


def run(
    *,
    task_id: str,
    cfg,
    log_dir: Path,
    warm_start_ckpt: Optional[str | Path] = None,
    resume_ckpt: Optional[str | Path] = None,
    reset_std: Optional[float] = None,
    early_stopping: bool = False,
    patience: int = 200,
    window: int = 50,
    min_improve: float = 0.005,
    critic_warmup_iters: Optional[int] = None,
    diagnostic_trunk_frozen: bool = False,
) -> None:
    """Entrypoint for PPO training with warm-start, resume, progressive unfreeze, and diagnostic modes.
    Args: task_id — mjlab task id; cfg — TrainConfig (env, agent, ...); log_dir — output dir; warm_start_ckpt — optional shape-filtered source ckpt; resume_ckpt — optional identical-shape resume ckpt (exclusive with warm_start_ckpt); reset_std — after resume, force the actor's Gaussian std to this scalar (typical: 1.0 to recover a stalled/blown-up std); early_stopping/patience/window/min_improve — plateau-stop controls; critic_warmup_iters — legacy freeze-actor duration (0 uses progressive unfreeze); diagnostic_trunk_frozen — freeze mlp.2/mlp.4 and swap head (requires warm_start_ckpt).
    Returns: none.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    if warm_start_ckpt is not None and resume_ckpt is not None:
        raise ValueError("Use warm_start_ckpt OR resume_ckpt, not both.")

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)
    log_dir = Path(log_dir)
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, agent_cfg, str(log_dir), device)

    if resume_ckpt is not None:
        print(f"[INFO] Resuming from: {resume_ckpt}")
        runner.load(str(resume_ckpt))
        if reset_std is not None:
            # Adaptive KL can inflate the Gaussian std when reward plateaus; forcing a
            # smaller std on resume lets the actor re-explore without a runaway noise level.
            dist = runner.alg.actor.distribution
            with torch.no_grad():
                if hasattr(dist, "std_param"):
                    dist.std_param.fill_(float(reset_std))
                    new_mean = float(dist.std_param.mean().item())
                elif hasattr(dist, "log_std_param"):
                    import math
                    dist.log_std_param.fill_(math.log(float(reset_std)))
                    new_mean = float(dist.log_std_param.exp().mean().item())
                else:
                    raise AttributeError(
                        "Actor distribution has no std_param/log_std_param — cannot reset std."
                    )
            print(f"[INFO] Reset policy std to {reset_std:.3f} (verified mean std={new_mean:.3f}).")
    elif warm_start_ckpt is not None:
        print(f"[INFO] Warm-start from: {warm_start_ckpt}")
        warm_start_shape_filtered(runner, warm_start_ckpt)
    elif reset_std is not None:
        print("[WARN] reset_std ignored: only takes effect together with resume_ckpt.")

    if critic_warmup_iters is None:
        critic_warmup_iters = 0

    max_iters = cfg.agent.max_iterations
    use_progressive = warm_start_ckpt is not None and critic_warmup_iters == 0 and not diagnostic_trunk_frozen

    # Diagnostic mode: PPO on swapped head, no progressive unfreeze so features are measured in isolation.
    if diagnostic_trunk_frozen:
        if warm_start_ckpt is None:
            raise ValueError("diagnostic_trunk_frozen requires a warm_start_ckpt.")
        _apply_trunk_frozen_diagnostic(runner)
        if early_stopping:
            print(f"[INFO] Early-stopping ON  (patience={patience}, window={window}, min_improve={min_improve:.3%})")
            _train_with_early_stop(runner, max_iters, patience, window, min_improve)
        else:
            runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)
        env.close()
        print("\nTraining complete.")
        return

    if use_progressive:
        print("[INFO] Progressive unfreezing: transferred deep layers frozen while the reinitialised first Linear calibrates.")
        _run_progressive_unfreeze(runner, max_iters)
    elif critic_warmup_iters > 0:
        warmup = min(critic_warmup_iters, max_iters)
        print(f"[INFO] Critic warmup: freezing actor for {warmup} iters (legacy).")
        _freeze_actor(runner, freeze=True)
        try:
            runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=True)
        finally:
            _freeze_actor(runner, freeze=False)
        remaining = max_iters - warmup
        if remaining <= 0:
            env.close()
            print("\nTraining complete.")
            return
        if early_stopping:
            print(f"[INFO] Early-stopping ON  (patience={patience}, window={window}, min_improve={min_improve:.3%})")
            _train_with_early_stop(runner, remaining, patience, window, min_improve)
        else:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    else:
        if early_stopping:
            print(f"[INFO] Early-stopping ON  (patience={patience}, window={window}, min_improve={min_improve:.3%})")
            _train_with_early_stop(runner, max_iters, patience, window, min_improve)
        else:
            runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)

    env.close()
    print("\nTraining complete.")
