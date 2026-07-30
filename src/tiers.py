###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""Tier presets for headless training on the B300 pod.

T1 = sanity (~2 min), T2 = diagnostic (~15 min), T3 = full (~90 min).
Tuned for B300 SXM6 (288 GB HBM3e). Scale down for smaller GPUs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    num_envs: int
    max_iterations: int
    save_interval: int
    label: str


TIERS = {
    "T1": Tier("T1",   512,   50,  25, "sanity — 2 min"),
    "T2": Tier("T2",  4096,  500, 100, "diagnostic — 15 min"),
    # T15 targets ~15 min for fast iteration.
    "T15": Tier("T15", 8192, 450, 100, "fast full — ~15 min"),
    # T15_XL: 2x envs (16384) for lower gradient variance. ~23 min at ~3.5 s/iter.
    "T15_XL": Tier("T15_XL", 16384, 400, 100, "wide fast — ~23 min"),
    # T10_XL: 200 iters at 16384 envs (~11-12 min). Half of T15_XL.
    "T10_XL": Tier("T10_XL", 16384, 200, 50, "quick wide — ~11 min"),
    # T45_XL: same width as T15_XL, 800 iters (~44 min).
    "T45_XL": Tier("T45_XL", 16384, 800, 100, "wide long — ~44 min"),
    # T2.5 targets ~40 min wall-clock on B300 with warm-started manipulation.
    "T2.5": Tier("T2.5", 8192, 1200, 100, "mid — ~40 min"),
    # T30 targets ~30 min from scratch on B300 for simpler tasks (drop-only).
    "T30": Tier("T30", 8192, 900, 100, "quick full — ~30 min"),
    # T1000: 1000 iters at 8192 envs, ~33 min.
    "T1000": Tier("T1000", 8192, 1000, 100, "1000 iters — ~33 min"),
    # T1500: 1500 iters at 8192 envs, ~50 min.
    "T1500": Tier("T1500", 8192, 1500, 100, "1500 iters — ~50 min"),
    "T3": Tier("T3", 24576, 1500, 200, "full — 90 min"),
    # T3000: 8192 envs x 3000 iters, targeted at drop full-run on B300 (~100 min).
    "T3000": Tier("T3000", 8192, 3000, 200, "drop full — ~100 min"),
    # T3K_XL: max-useful GPU on B300 with the same data budget as T3000
    # (24576*1000 = 8192*3000 = 24.576M "iter units"). Triples parallel envs,
    # thirds iterations. Uses ~55 GB VRAM.
    "T3K_XL": Tier("T3K_XL", 24576, 1000, 100, "wide drop — ~90 min"),
}


def apply_tier(cfg, tier_name: str) -> Tier:
    """Mutate cfg.env.scene.num_envs, cfg.agent.max_iterations, cfg.agent.save_interval in place.
    Args: cfg — a TrainConfig; tier_name — one of T1/T2/T3.
    Returns: the resolved Tier for logging.
    """
    tier = TIERS[tier_name]
    cfg.env.scene.num_envs = tier.num_envs
    cfg.agent.max_iterations = tier.max_iterations
    cfg.agent.save_interval = tier.save_interval
    return tier
