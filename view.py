###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""Standalone viewer launcher for the newest g1_pick checkpoint via mjlab viser.

Input:  outputs/logs/g1_pick/<run>/model_*.pt (auto-discovered).
Output: interactive viser session at http://localhost:8080 with the curriculum locked to the checkpoint iter.
"""

import os, re, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

root = Path(__file__).resolve().parent
os.chdir(root); sys.path.insert(0, str(root))

from src import policy_pick
from src.utils import find_latest_checkpoint
from mjlab.scripts.play import PlayConfig, run_play

policy_pick.register()

ckpt = find_latest_checkpoint("outputs/logs", experiment_hint="g1_pick")
print(f"checkpoint: {ckpt}")

# Freeze the curriculum at the checkpoint's iter so the viewer reflects the trained regime, not the warmup phase.
match = re.search(r"model_(\d+)\.pt$", str(ckpt))
if match:
    iter_num = int(match.group(1))
    policy_pick.lock_curriculum_at_iter(iter_num)
    print(f"curriculum locked at iter {iter_num} (step {iter_num * 24})")
else:
    print("warn: could not parse iter from checkpoint filename; curriculum starts fresh")

print(f"OBJECT_XY_JITTER_X   : {policy_pick.OBJECT_XY_JITTER_X}")

play_cfg = PlayConfig(
    agent="trained",
    checkpoint_file=str(ckpt),
    num_envs=1,
    viewer="viser",
)
print("open http://localhost:8080")
run_play(policy_pick.TASK_ID, play_cfg)
