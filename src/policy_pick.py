###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""G1 pick task definition shared by training and analysis notebooks.

Task: G1 humanoid stands in front of a table, reaches for a box with both
hands, grasps it, and lifts it to a target height.

Rewards combine reach (hand-to-box shaping, palm alignment, foot support) and
carry (symmetric grasp anchors, box-at-target pos/quat, upright torso), plus a
volumetric body/table collision penalty that also catches the "hand under the
table" cheat.

Independent from policy_carry.py; helpers are intentionally duplicated.
"""

from __future__ import annotations

import math
import re

import mujoco
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME, get_spec
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import base_ang_vel, base_lin_vel, projected_gravity
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


# Task identity and G1 constants.
TASK_ID = "Mjlab-G1-Pick"

# Contract for src.collect_buffer (pick produces pick_terminal_states.npz for carry).
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True

WALK_BUFFER_PATH = "outputs/walk_terminal_states.npz"

_spec_model = get_spec().compile()
G1_JOINTS = tuple(
    mujoco.mj_id2name(_spec_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(_spec_model.njnt)
    if _spec_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
)
G1_HOME_ROOT_POS = KNEES_BENT_KEYFRAME.pos
G1_HOME_ROOT_ROT = KNEES_BENT_KEYFRAME.rot
G1_HOME_JOINT_POS = {
    j: next((v for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items() if re.fullmatch(pat, j)), 0.0)
    for j in G1_JOINTS
}
G1_FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


# Ready-arm pose snapped at reset (after reset_from_walk_buffer) so every env
# starts with hands forward-down, roughly above the box on the table. Cuts
# the initial random arm scramble that used to dominate the first ~100 iters
# and drops ep_length from 9 to something usable from tick 0.
# Not fully closed like HOLDING_ARM_POSE — the policy still has to descend
# and squeeze to actually grasp the box.


# Curriculum start pose: arms on a diagonal pointing forward-down toward the
# box. G1 shoulder_pitch convention: 0 = arm hanging straight down, -pi/2 =
# arm horizontal forward. So LESS negative pitch = arm closer to hanging =
# hand LOWER. READY at -0.85 puts hand at z=0.79; PINCH at -0.60 puts hand
# lower (closer to box at z=0.60). Shoulder roll opens the hands sideways
# for the box y=+/-0.13 side-face targets. Elbow kept small so the forearm
# extends the shoulder direction rather than folding back up.
PINCH_POSE = {
    # Upper arms angled forward-up (like carry hugging pose) so the shoulders
    # sit higher and the arms wrap around the box from above; elbow flexion
    # then drops the forearm down onto the box side faces for the pinch.
    "left_shoulder_pitch_joint":  -1.00,
    "left_shoulder_roll_joint":    0.20,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            1.40,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -1.00,
    "right_shoulder_roll_joint":  -0.20,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           1.40,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
    # Torso leans forward + legs bend to keep CoM over the feet. Without the
    # hip/knee/ankle compensation, the 23-degree torso lean tips the robot
    # forward on tick 0 and every episode dies before the policy can react
    # (the ~1000-iter ep_length stall observed empirically).
    "waist_pitch_joint":           0.40,
    "left_hip_pitch_joint":        0.30,
    "left_knee_joint":             0.30,
    "left_ankle_pitch_joint":     -0.15,
    "right_hip_pitch_joint":       0.30,
    "right_knee_joint":            0.30,
    "right_ankle_pitch_joint":    -0.15,
}
PINCH_JOINTS = tuple(PINCH_POSE.keys())


# Scene and task geometry.
TABLE_X          = 0.60
TABLE_HEIGHT     = 0.44
TABLE_TOP_HALF   = 0.04
TABLE_TOP_Z      = TABLE_HEIGHT + TABLE_TOP_HALF   # top surface where the box sits
BOX_HALF         = 0.105
BOX_START_X      = TABLE_X - 0.09
BOX_START_Z      = TABLE_TOP_Z + BOX_HALF
BOX_MASS_MIN     = 0.30
BOX_MASS_MAX     = 3.00
LIFT_TARGET_Z    = TABLE_TOP_Z + 0.34   # for notebook prints and analysis

# Centre of the lift-target z range (matches carry's chest-hold height).
CHEST_Z          = 0.93


# 20-D per-reset command: grasp offsets, object spawn pose, lift target (yaw-only quats).
GRASP_NOMINAL_Y   = 0.14   # left-wrist target on box side face (box_half=0.105 + wrist sphere radius 0.035); right mirrors
GRASP_NOMINAL_Z   = 0.02
GRASP_JITTER      = 0.03

# Box always spawns on the table with zero velocity + tight domain randomization.
# X jitter kept at 5 cm; Y jitter reduced to 1 cm so the wrist pinch target
# (fixed at world y = +/- 0.14 via the reset pose) reliably lands on the box
# side face regardless of the jitter draw. Wider Y jitter used to leave the
# wrists either 5 cm inside or 5 cm outside the wall on any given episode.
OBJECT_XY_JITTER_X = 0.05
OBJECT_XY_JITTER_Y = 0.01
OBJECT_YAW_RANGE   = math.radians(20.0)

LIFT_X_CENTER   = 0.30
LIFT_X_HALF     = 0.10
LIFT_Y_HALF     = 0.20
LIFT_Z_HALF     = 0.15
LIFT_YAW_RANGE  = 0.30

_CMD_KEY = "pick_cmd_"    # env.extras namespace prefix

CMD_DIMS = {
    "grasp_offset_left":   3,
    "grasp_offset_right":  3,
    "object_target_pos":   3,
    "object_target_quat":  4,
    "lift_target_pos":     3,
    "lift_target_quat":    4,
}


# Arms and wrists only; leg symmetry excluded to allow weight-shifting toward the box.
_SYMMETRIC_PAIRS = (
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", +1),
    ("left_shoulder_roll_joint",  "right_shoulder_roll_joint",  -1),
    ("left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",   +1),
    ("left_elbow_joint",          "right_elbow_joint",          +1),
    ("left_wrist_roll_joint",     "right_wrist_roll_joint",     -1),
    ("left_wrist_pitch_joint",    "right_wrist_pitch_joint",    +1),
    ("left_wrist_yaw_joint",      "right_wrist_yaw_joint",      +1),
)
_LEFT_SYM_JOINTS  = tuple(p[0] for p in _SYMMETRIC_PAIRS)
_RIGHT_SYM_JOINTS = tuple(p[1] for p in _SYMMETRIC_PAIRS)
_SYM_SIGNS        = tuple(float(p[2]) for p in _SYMMETRIC_PAIRS)


# Full 29-joint mirror map for PPO data augmentation.
# Format: for i in [0, 29), _JOINT_MIRROR_MAP[i] = (j, sign) such that
# mirrored_joint[i] = sign * original_joint[j] under a sagittal-plane mirror
# (world y=0, robot yaw ≈ 0 — valid for pick since the robot faces the table
# along +x for the whole episode).
# Sign convention matches _SYMMETRIC_PAIRS: pitch/knee/elbow = +1; roll = -1;
# yaw = +1 for arm/hip joints (G1's per-joint frame absorbs the mirror), -1
# for the centerline waist_yaw and waist_roll (true axial rotations).
_JOINT_MIRROR_MAP = (
    ( 6, +1.0), ( 7, -1.0), ( 8, +1.0), ( 9, +1.0), (10, +1.0), (11, -1.0),  # left leg → right leg
    ( 0, +1.0), ( 1, -1.0), ( 2, +1.0), ( 3, +1.0), ( 4, +1.0), ( 5, -1.0),  # right leg → left leg
    (12, -1.0), (13, -1.0), (14, +1.0),                                      # waist yaw/roll flip; pitch keeps
    (22, +1.0), (23, -1.0), (24, +1.0), (25, +1.0), (26, -1.0), (27, +1.0), (28, +1.0),  # left arm → right arm
    (15, +1.0), (16, -1.0), (17, +1.0), (18, +1.0), (19, -1.0), (20, +1.0), (21, +1.0),  # right arm → left arm
)
assert len(_JOINT_MIRROR_MAP) == len(G1_JOINTS), "_JOINT_MIRROR_MAP out of sync with G1_JOINTS"


# Scene builders.
def _get_table_spec() -> mujoco.MjSpec:
    """Build the MjSpec for the table (top + four legs).
    Args: none.
    Returns: MjSpec containing the table body.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table", pos=(TABLE_X, 0.0, TABLE_HEIGHT))
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.34, 0.42, TABLE_TOP_HALF),
        rgba=(0.47, 0.40, 0.32, 1.0),
    )
    leg_len_z    = TABLE_HEIGHT - TABLE_TOP_HALF
    leg_half_z   = leg_len_z / 2.0
    leg_center_z = leg_half_z - TABLE_HEIGHT
    for name, pos_xy in {
        "table_leg_fl": ( 0.29,  0.37),
        "table_leg_fr": ( 0.29, -0.37),
        "table_leg_bl": (-0.29,  0.37),
        "table_leg_br": (-0.29, -0.37),
    }.items():
        body.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.025, 0.025, leg_half_z),
            pos=(*pos_xy, leg_center_z),
            rgba=(0.47, 0.40, 0.32, 1.0),
        )
    return spec


def _get_box_spec() -> mujoco.MjSpec:
    """Build the MjSpec for the graspable box with a free joint.
    Args: none.
    Returns: MjSpec containing the box body.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    freejoint = body.add_freejoint()
    freejoint.name = "box_free"
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(BOX_HALF, BOX_HALF, BOX_HALF),
        mass=BOX_MASS_MIN,
        rgba=(0.85, 0.22, 0.18, 1.0),
        condim=6,
        friction=(4.0, 0.25, 0.08),
        # solref[0] must be >= 2 * timestep for MuJoCo stability. With
        # dt=0.005 the previous 0.004 was below the rule, causing the box
        # to sink slightly into the table each step. 0.02 is safely above.
        solref=(0.02, 1.0),
        solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
    )
    return spec


# Helpers.
def _entity_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the world-frame position of the entity or one of its bodies.
    Args: env — the RL env; asset_cfg — scene entity config, optionally with body_ids.
    Returns: (num_envs, 3) tensor of positions.
    """
    entity: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_names is not None or isinstance(asset_cfg.body_ids, list):
        return entity.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)
    return entity.data.root_link_pos_w


def body_vector(env, source_cfg, target_cfg):
    """Compute the world-frame vector from source body to target body.
    Args: env — the RL env; source_cfg — origin entity; target_cfg — endpoint entity.
    Returns: (num_envs, 3) tensor of displacement.
    """
    return _entity_pos(env, target_cfg) - _entity_pos(env, source_cfg)


# Box signals sanitized so MuJoCo divergences never trip rsl_rl.check_nan.
def box_position_obs(env, box_cfg):
    """Box position relative to the robot base, sanitized and clamped.
    Args: env — the RL env; box_cfg — scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    box_pos = env.scene[box_cfg.name].data.root_link_pos_w
    robot_pos = env.scene["robot"].data.root_link_pos_w
    rel = box_pos - robot_pos
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)


def box_lin_vel_obs(env, box_cfg):
    """Box linear velocity in world frame, sanitized and clamped.
    Args: env — the RL env; box_cfg — scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    v = env.scene[box_cfg.name].data.root_link_lin_vel_w
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)


def box_ang_vel_obs(env, box_cfg):
    """Box angular velocity in world frame, sanitized and clamped.
    Args: env — the RL env; box_cfg — scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    w = env.scene[box_cfg.name].data.root_link_ang_vel_w
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)


def box_orientation_obs(env, box_cfg):
    """Box orientation quaternion in world frame, sanitized and clamped.
    Args: env — the RL env; box_cfg — scene entity config for the box.
    Returns: (num_envs, 4) tensor.
    """
    q = env.scene[box_cfg.name].data.root_link_quat_w
    return torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def box_size_obs(env):
    """Broadcast the fixed box size to every env for observation input.
    Args: env — the RL env.
    Returns: (num_envs, 3) tensor of box dimensions.
    """
    root_pos = env.scene["robot"].data.root_link_pos_w
    size = torch.tensor([2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF], device=root_pos.device)
    return size.expand(root_pos.shape[0], -1)


def wrist_actuator_force_obs(env, asset_cfg: SceneEntityCfg):
    """Actuator force at wrist joints; spikes when the hand is loaded during grasp.
    Args: env — the RL env; asset_cfg — entity config selecting wrist actuators.
    Returns: (num_envs, n_actuators) tensor of forces.
    """
    robot: Entity = env.scene[asset_cfg.name]
    forces = robot.data.actuator_force
    if asset_cfg.actuator_ids is not None and not isinstance(asset_cfg.actuator_ids, slice):
        idx = asset_cfg.actuator_ids
        if isinstance(idx, (list, tuple)):
            forces = forces[:, list(idx)]
        else:
            forces = forces[:, idx]
    return forces


# Pick task command buffers live on env.extras and are allocated on first reset/obs.
def _ensure_cmd_buffers(env: ManagerBasedRlEnv) -> None:
    """Lazily allocate per-env command tensors on env.extras.
    Args: env — the RL env.
    Returns: none.
    """
    for name, dim in CMD_DIMS.items():
        key = _CMD_KEY + name
        buf = env.extras.get(key)
        if buf is None or buf.shape[0] != env.num_envs:
            env.extras[key] = torch.zeros(env.num_envs, dim, device=env.device)
    # Identity quat default for yaw-only slots.
    for qname in ("object_target_quat", "lift_target_quat"):
        env.extras[_CMD_KEY + qname][:, 0] = 1.0


def reset_arms_to_ready(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pinch_bias: float = 0.0,
) -> None:
    """Blend the joints in PINCH_POSE from the current walk-buffer state toward PINCH.
    pinch_bias=1.0 → snap to PINCH_POSE (arms + torso + hip/knee compensation).
    pinch_bias=0.0 → no-op: the walk-buffer reset that just ran wins.
    Blend: (1-bias) * current + bias * PINCH_target.
    Args: env — RL env; env_ids — envs to reset (None = all); asset_cfg — robot entity with joint_names=PINCH_JOINTS; pinch_bias — blend in [0, 1], ramped by pick_curriculum.
    Returns: none.
    """
    pb = float(max(0.0, min(1.0, pinch_bias)))
    if pb <= 0.0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids_t = torch.tensor(joint_ids, device=env.device)
    else:
        joint_ids_t = joint_ids
    current = asset.data.joint_pos[env_ids][:, joint_ids_t]
    pinch_target = torch.tensor(
        [PINCH_POSE[j] for j in asset_cfg.joint_names],
        device=env.device,
    ).unsqueeze(0).expand(n, -1)
    pose = ((1.0 - pb) * current + pb * pinch_target).clone()
    vel = torch.zeros_like(pose)
    asset.write_joint_state_to_sim(pose, vel, joint_ids=joint_ids_t, env_ids=env_ids)


def reset_pick_command_and_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale: float = 1.0,
) -> None:
    """Sample the 20-D command and write the box spawn on the table.
    Args: env — the RL env; env_ids — envs to reset (all if None); scale — curriculum range multiplier in [0, 1] applied to xy/yaw jitter.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import sample_uniform, quat_from_euler_xyz

    _ensure_cmd_buffers(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    origins = env.scene.env_origins[env_ids]
    zero = torch.zeros(n, device=device)

    # Box always spawns at rest on the table (no "in hands" branch): removed the
    # hand_bias interpolation because the policy was learning to catch the box
    # mid-fall instead of picking it up from the surface.
    ox = sample_uniform(-OBJECT_XY_JITTER_X * scale, OBJECT_XY_JITTER_X * scale, (n,), device=device)
    oy = sample_uniform(-OBJECT_XY_JITTER_Y * scale, OBJECT_XY_JITTER_Y * scale, (n,), device=device)
    o_yaw = sample_uniform(-OBJECT_YAW_RANGE * scale, OBJECT_YAW_RANGE * scale, (n,), device=device)
    obj_pos_w = torch.stack([
        origins[:, 0] + BOX_START_X + ox,
        origins[:, 1] + oy,
        origins[:, 2] + BOX_START_Z,
    ], dim=-1)
    obj_quat_w = quat_from_euler_xyz(zero, zero, o_yaw)

    box: Entity = env.scene["box"]
    box.write_root_link_pose_to_sim(
        torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )

    # object_target_* echoes the spawn pose exactly.
    env.extras[_CMD_KEY + "object_target_pos"][env_ids]  = obj_pos_w
    env.extras[_CMD_KEY + "object_target_quat"][env_ids] = obj_quat_w

    # Grasp offsets in box frame; right mirrors left across sagittal plane.
    jitter = GRASP_JITTER * scale
    gdx = sample_uniform(-jitter, jitter, (n,), device=device)
    gdy = sample_uniform(-jitter, jitter, (n,), device=device)
    gdz = sample_uniform(-jitter, jitter, (n,), device=device)
    grasp_left = torch.stack([
        gdx,
        torch.full((n,), GRASP_NOMINAL_Y, device=device) + gdy,
        torch.full((n,), GRASP_NOMINAL_Z, device=device) + gdz,
    ], dim=-1)
    mirror = torch.tensor([1.0, -1.0, 1.0], device=device)
    grasp_right = grasp_left * mirror
    env.extras[_CMD_KEY + "grasp_offset_left"][env_ids]  = grasp_left
    env.extras[_CMD_KEY + "grasp_offset_right"][env_ids] = grasp_right

    # Lift target pose.
    lx = sample_uniform(-LIFT_X_HALF * scale, LIFT_X_HALF * scale, (n,), device=device)
    ly = sample_uniform(-LIFT_Y_HALF * scale, LIFT_Y_HALF * scale, (n,), device=device)
    lz = sample_uniform(-LIFT_Z_HALF * scale, LIFT_Z_HALF * scale, (n,), device=device)
    lift_pos_w = torch.stack([
        origins[:, 0] + LIFT_X_CENTER + lx,
        origins[:, 1] + ly,
        origins[:, 2] + CHEST_Z + lz,
    ], dim=-1)
    l_yaw = sample_uniform(-LIFT_YAW_RANGE * scale, LIFT_YAW_RANGE * scale, (n,), device=device)
    lift_quat_w = quat_from_euler_xyz(zero, zero, l_yaw)
    env.extras[_CMD_KEY + "lift_target_pos"][env_ids]  = lift_pos_w
    env.extras[_CMD_KEY + "lift_target_quat"][env_ids] = lift_quat_w

    # Reseed the per-episode max-box-z tracker used by box_fell_back termination
    # to the current box spawn z; without this, max would persist across
    # episodes and cause spurious early terminations.
    max_key = "pick_max_box_z"
    if max_key not in env.extras or env.extras[max_key].shape[0] != env.num_envs:
        env.extras[max_key] = torch.zeros(env.num_envs, device=env.device)
    env.extras[max_key][env_ids] = obj_pos_w[:, 2]


def _pos_in_base_frame(env: ManagerBasedRlEnv, world_pos: torch.Tensor) -> torch.Tensor:
    """Transform a world-frame position tensor to the robot base frame.
    Args: env — the RL env; world_pos — (num_envs, 3) tensor of world positions.
    Returns: (num_envs, 3) tensor in base frame.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    robot: Entity = env.scene["robot"]
    delta = world_pos - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, delta)


def cmd_grasp_offset_left_obs(env):
    """Observation for left-hand grasp offset in box frame.
    Args: env — the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "grasp_offset_left"]


def cmd_grasp_offset_right_obs(env):
    """Observation for right-hand grasp offset in box frame.
    Args: env — the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "grasp_offset_right"]


def cmd_object_target_pos_obs(env):
    """Object-target position expressed in robot base frame.
    Args: env — the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "object_target_pos"])


def cmd_object_target_quat_obs(env):
    """Object-target orientation quaternion in world frame.
    Args: env — the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "object_target_quat"]


def cmd_lift_target_pos_obs(env):
    """Lift-target position expressed in robot base frame.
    Args: env — the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "lift_target_pos"])


def cmd_lift_target_quat_obs(env):
    """Lift-target orientation quaternion in world frame.
    Args: env — the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "lift_target_quat"]


# Carry-style fixed side-face target: forces wrist-pinching geometry instead of
# single-sphere gripping. Left wrist targets +y face of box; right wrist targets -y face.
# Deterministic (no per-episode jitter) so the policy converges to the pinch pose.
def _desired_grasp_pos_w(env, box_cfg, side: str) -> torch.Tensor:
    """Return world-frame position of the wrist-pinch target on the box side face.
    Args: env — the RL env; box_cfg — box entity config; side — "left" or "right".
    Returns: (num_envs, 3) tensor of pinch points on the box side.
    """
    box_w    = _entity_pos(env, box_cfg)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    qw, qx, qy, qz = box_quat.unbind(dim=-1)
    box_yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = torch.cos(box_yaw), torch.sin(box_yaw)
    # +y outward normal of box in world xy is (-sin_yaw, cos_yaw); flip for right side.
    sign = 1.0 if side == "left" else -1.0
    dx = -sign * GRASP_NOMINAL_Y * sy
    dy =  sign * GRASP_NOMINAL_Y * cy
    dz = torch.full_like(box_w[:, 2], GRASP_NOMINAL_Z)
    return box_w + torch.stack([dx, dy, dz], dim=-1)


def reach_reward(env, hand_cfg, box_cfg, side: str, std: float = 0.20):
    """Gaussian shaping on hand distance to the commanded grasp point.
    Args: env — the RL env; hand_cfg — hand entity config; box_cfg — box entity config; side — "left" or "right"; std — kernel width.
    Returns: (num_envs,) tensor of reward values in (0, 1].
    """
    hand_w = _entity_pos(env, hand_cfg)
    d = torch.linalg.norm(_desired_grasp_pos_w(env, box_cfg, side) - hand_w, dim=-1)
    return torch.exp(-d * d / (2.0 * std * std))


def both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, threshold: float = 0.08):
    """+1 when both hands are within threshold of their commanded side grasps.
    Args: env — the RL env; left_hand_cfg — left hand entity; right_hand_cfg — right hand entity; box_cfg — box entity; threshold — hit distance.
    Returns: (num_envs,) tensor of 0/1 values.
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    return ((dl < threshold) & (dr < threshold)).float()


# Dense-monotonic reward triplet copying walk/carry's shape: ungated Gaussian
# kernels over a single error dim, always positive, always paying gradient.
# Provide the "there's something to optimize on tick 0" signal that the gated
# bundle rewards (hand_placement, stability, lift) can't produce until the
# robot already stands and reaches. Small weights so they shape the local
# optimum without overshadowing the true task rewards once training rolls.
def hand_to_grasp_target_dense(env, left_hand_cfg, right_hand_cfg, box_cfg, std: float = 0.25):
    """Ungated Gaussian on each hand's distance to its commanded grasp point.
    Args: env — RL env; left/right_hand_cfg — hand entity selectors; box_cfg — box entity; std — kernel width in meters (0.25 pays partial reward from ~40 cm out).
    Returns: (num_envs,) tensor in [0, 2] (sum of two per-hand Gaussians).
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    return torch.exp(-0.5 * (dl / std) ** 2) + torch.exp(-0.5 * (dr / std) ** 2)


def upright_dense(env, trunk_cfg, std: float = 0.45):
    """Ungated Gaussian on torso tilt: 1 when perfectly upright, decays smoothly.
    Args: env — RL env; trunk_cfg — trunk body selector; std — kernel width on (1 - up_z).
    Returns: (num_envs,) tensor in [0, 1]. Same shape as walk's mdp.upright.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    if trunk_cfg.body_ids is not None:
        quat = entity.data.body_link_quat_w[:, trunk_cfg.body_ids].squeeze(1)
    else:
        quat = entity.data.root_link_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return torch.exp(-0.5 * ((1.0 - up_z) / std) ** 2)


def hand_pinch_progress(env, left_hand_cfg, right_hand_cfg, box_cfg, std: float = 0.05):
    """Smooth "both hands near pinch target" — Gaussian replacement for the binary hit_bonus.
    Product of two 5 cm Gaussians so it saturates at 1 only when BOTH hands are on target.
    Args: env — RL env; left/right_hand_cfg — hand entity selectors; box_cfg — box entity; std — per-hand kernel width in meters.
    Returns: (num_envs,) tensor in [0, 1].
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    return torch.exp(-0.5 * (dl / std) ** 2) * torch.exp(-0.5 * (dr / std) ** 2)


def palm_inward_reward(env, hand_cfg, palm_axis_local, target_axis_body):
    """Reward palm-normal alignment with a target axis in the base frame.
    Args: env — the RL env; hand_cfg — hand entity; palm_axis_local — palm normal in hand frame; target_axis_body — target axis in base frame.
    Returns: (num_envs,) tensor in [0, 1], max when aligned.
    """
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    robot = env.scene["robot"]
    asset = env.scene[hand_cfg.name]
    hand_quat = asset.data.body_link_quat_w[:, hand_cfg.body_ids[0], :]
    root_quat = robot.data.root_link_quat_w
    palm_local = torch.tensor(palm_axis_local, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    palm_world = quat_apply(hand_quat, palm_local)
    palm_body  = quat_apply_inverse(root_quat, palm_world)
    tgt = torch.tensor(target_axis_body, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    cos_a = (palm_body * tgt).sum(dim=-1)
    return 0.5 * (1.0 + cos_a)


def feet_on_ground_reward(env, sensor_name: str):
    """Graded ground-contact reward: 0.5 for one foot, 1.0 for both.
    Args: env — the RL env; sensor_name — key of the feet-contact sensor in the scene.
    Returns: (num_envs,) tensor in [0, 1].
    """
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    return 0.5 * in_contact.sum(dim=1).clamp(0.0, 2.0)


def stability_gate(env, trunk_cfg, sensor_name):
    """Binary gate: 1 if torso upright (≤25° tilt) AND both feet in contact.
    Used to multiply into task rewards so a falling episode earns zero credit
    for reaching/grasping/lifting — forces the policy to solve balance first.
    """
    entity = env.scene[trunk_cfg.name]
    up_align = -entity.data.projected_gravity_b[:, 2]   # 1.0 upright, 0 sideways
    upright = up_align > 0.9
    sensor = env.scene[sensor_name]
    both_feet = (sensor.data.found > 0).sum(dim=1) >= 2
    return (upright & both_feet).float()


def stay_still_penalty(
    env,
    left_hand_cfg: SceneEntityCfg | None = None,
    right_hand_cfg: SceneEntityCfg | None = None,
    box_cfg: SceneEntityCfg | None = None,
    gate_dist: float = 0.20,
):
    """Penalise base linear velocity, gated on both hands being near the grasp.
    Args: env — the RL env; left_hand_cfg — left hand entity or None; right_hand_cfg — right hand entity or None; box_cfg — box entity or None; gate_dist — activation threshold in metres.
    Returns: (num_envs,) tensor of |v|^2 masked by the gate.
    """
    root_vel = env.scene["robot"].data.root_link_lin_vel_w
    v_sq = torch.sum(root_vel * root_vel, dim=-1)
    if left_hand_cfg is None or right_hand_cfg is None or box_cfg is None:
        return v_sq  # ungated fallback for backward compatibility

    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    gate = ((dl < gate_dist) & (dr < gate_dist)).float()
    return v_sq * gate


def base_ang_vel_penalty(env):
    """Squared L2 penalty on the base angular velocity.
    Args: env — the RL env.
    Returns: (num_envs,) tensor.
    """
    root_ang_vel = env.scene["robot"].data.root_link_ang_vel_w
    return torch.sum(root_ang_vel * root_ang_vel, dim=-1)


def action_rate_l2(env):
    """Squared L2 penalty on action rate between consecutive steps."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def action_2nd_order_l2(env):
    """Squared L2 penalty on action second-order difference (jerk).
    ETH standard: penalises |a_t − 2·a_{t−1} + a_{t−2}|², complements action_rate.
    """
    a = env.action_manager.action
    a1 = env.action_manager.prev_action
    if not hasattr(env, "_prev_prev_action") or env._prev_prev_action.shape != a.shape:
        env._prev_prev_action = a1.clone()
    a2 = env._prev_prev_action
    diff = a - 2.0 * a1 + a2
    env._prev_prev_action = a1.clone()
    return torch.sum(diff * diff, dim=-1)


def dof_vel_l2(env):
    """ETH standard: squared L2 on joint velocities. Discourages fast joints."""
    v = env.scene["robot"].data.joint_vel
    return torch.sum(v * v, dim=-1)


def dof_acc_l2(env):
    """ETH standard: squared L2 on joint accelerations. Most effective smoothing regulariser."""
    a = env.scene["robot"].data.joint_acc
    return torch.sum(a * a, dim=-1)


def dof_torque_l2(env):
    """ETH standard: squared L2 on joint torques (energy proxy)."""
    tau = env.scene["robot"].data.qfrc_actuator
    return torch.sum(tau * tau, dim=-1)


def hand_approach_velocity_penalty(env, left_hand_cfg, right_hand_cfg, box_cfg,
                                    gate_dist: float = 0.20):
    """Penalise hand LINEAR velocity when close to box (forces gentle approach).
    Uses body_link_lin_vel_w (proper sim velocity, no reset spike bug).
    """
    lh_pos = _entity_pos(env, left_hand_cfg)
    rh_pos = _entity_pos(env, right_hand_cfg)
    box_pos = _entity_pos(env, box_cfg)
    dl = torch.linalg.norm(lh_pos - box_pos, dim=-1)
    dr = torch.linalg.norm(rh_pos - box_pos, dim=-1)
    close_gate = (torch.minimum(dl, dr) < gate_dist).float()

    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    if not hasattr(env, "_hand_body_idx"):
        lh_name = left_hand_cfg.body_names[0] if isinstance(left_hand_cfg.body_names, (list, tuple)) else left_hand_cfg.body_names
        rh_name = right_hand_cfg.body_names[0] if isinstance(right_hand_cfg.body_names, (list, tuple)) else right_hand_cfg.body_names
        env._hand_body_idx = (body_names.index(lh_name), body_names.index(rh_name))
    lv = robot.data.body_link_lin_vel_w[:, env._hand_body_idx[0]]
    rv = robot.data.body_link_lin_vel_w[:, env._hand_body_idx[1]]
    speed = torch.linalg.norm(lv, dim=-1) + torch.linalg.norm(rv, dim=-1)
    return speed * close_gate


def palm_inward_bilateral_reward(env, left_hand_cfg, right_hand_cfg,
                                  left_palm_axis_local=(0.0, -1.0, 0.0),
                                  right_palm_axis_local=(0.0,  1.0, 0.0),
                                  left_target_axis_body=(0.0, -1.0, 0.0),
                                  right_target_axis_body=(0.0,  1.0, 0.0),
                                  w_left: float = 0.5, w_right: float = 0.5):
    """Fused palm-inward alignment for both hands. Weighted sum of per-hand rewards.
    Args: env — RL env; left/right_hand_cfg — hand entities; left/right_palm_axis_local — palm normal in hand frame; left/right_target_axis_body — target axis in base frame; w_left/w_right — per-side weights folded in.
    Returns: (num_envs,) tensor.
    """
    lr = palm_inward_reward(env, left_hand_cfg,  left_palm_axis_local,  left_target_axis_body)
    rr = palm_inward_reward(env, right_hand_cfg, right_palm_axis_local, right_target_axis_body)
    return w_left * lr + w_right * rr


def body_collision_bundle_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    left_hand_cfg, right_hand_cfg, box_cfg,
    non_hand_box_threshold: float = 0.22,
    table_margin_xy: float = 0.02,
    table_z_slack: float = 0.02,
    w_non_hand_box: float = -2.0,
    w_body_table: float = -2.0,
):
    """Fused body-vs-box proximity + body-vs-table volumetric penalty.
    Positions of the 7 non-hand/hand bodies are read once; box distances use the
    5 non-hand ones, table distances use all 7. Weighted sum of the two penalties.
    Args: env — RL env; body cfgs; box_cfg — box entity; non_hand_box_threshold — proximity threshold; table_margin_xy — footprint shrink; table_z_slack — vertical tolerance; w_non_hand_box/w_body_table — per-signal weights (negative).
    Returns: (num_envs,) tensor.
    """
    device = env.scene["robot"].data.root_link_pos_w.device
    n_envs = env.scene["robot"].data.root_link_pos_w.shape[0]
    box_pos = _entity_pos(env, box_cfg)

    non_hand_positions = [_entity_pos(env, cfg) for cfg in
                          (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg)]
    hand_positions = [_entity_pos(env, cfg) for cfg in (left_hand_cfg, right_hand_cfg)]

    non_hand_box_pen = torch.zeros(n_envs, device=device)
    for pos in non_hand_positions:
        dist = torch.linalg.norm(pos - box_pos, dim=-1)
        non_hand_box_pen += torch.clamp(non_hand_box_threshold - dist, min=0.0) / non_hand_box_threshold

    t_xy = torch.tensor([TABLE_X, 0.0], device=device)
    t_half_xy = torch.tensor([0.34 - table_margin_xy, 0.42 - table_margin_xy], device=device)
    top_z = TABLE_TOP_Z + TABLE_TOP_HALF - table_z_slack
    body_table_pen = torch.zeros(n_envs, device=device)
    for pos in non_hand_positions + hand_positions:
        xy_into = t_half_xy - torch.abs(pos[:, :2] - t_xy)
        z_below = top_z - pos[:, 2]
        depth = torch.min(
            torch.stack([xy_into[:, 0], xy_into[:, 1], z_below], dim=-1),
            dim=-1,
        ).values
        body_table_pen += torch.clamp(depth, min=0.0)

    return w_non_hand_box * non_hand_box_pen + w_body_table * body_table_pen


def stability_bundle_reward(env, trunk_cfg, sensor_name, pelvis_cfg=None,
                             w_upright: float = 10.0,
                             w_upright_pelvis: float = 5.0,
                             w_feet: float = 5.0):
    """Fused torso-upright + optional pelvis-upright + feet-on-ground stability.
    Args: env — RL env; trunk_cfg — trunk entity; sensor_name — feet contact sensor; pelvis_cfg — optional pelvis entity (if given, adds a second upright term); w_upright/w_upright_pelvis/w_feet — per-signal weights folded in.
    Returns: (num_envs,) tensor.
    """
    total = (w_upright * upright_torso_reward(env, trunk_cfg)
           + w_feet    * feet_on_ground_reward(env, sensor_name))
    if pelvis_cfg is not None:
        total = total + w_upright_pelvis * upright_torso_reward(env, pelvis_cfg)
    return total


# Carry-style rewards: symmetric grasp, box pose, posture.
def box_at_chest_reward(env, box_cfg, target_z: float = CHEST_Z):
    """Gaussian reward on box z-height around chest height. Copied from policy_carry.
    Args: env — RL env; box_cfg — box entity; target_z — desired box height in metres.
    Returns: (num_envs,) reward in (0, 1].
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    diff = box_z - target_z
    return torch.exp(-(diff ** 2) / 0.05)


def box_upright_reward(env, box_cfg):
    """Reward alignment of box local +z with world +z. Copied from policy_carry.
    Args: env — RL env; box_cfg — box entity.
    Returns: (num_envs,) reward clamped to [0, 1].
    """
    entity = env.scene[box_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(up_z, 0.0, 1.0)


def box_front_aligned_reward(env, box_cfg):
    """Reward alignment of box local +x with robot heading. Copied from policy_carry.
    Args: env — RL env; box_cfg — box entity.
    Returns: (num_envs,) reward clamped to [0, 1].
    """
    entity = env.scene[box_cfg.name]
    quat = entity.data.root_link_quat_w
    qw, qx, qy, qz = quat.unbind(dim=-1)
    front_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    front_y = 2.0 * (qx * qy + qw * qz)
    robot_yaw = env.scene["robot"].data.heading_w
    rx = torch.cos(robot_yaw)
    ry = torch.sin(robot_yaw)
    cos_diff = front_x * rx + front_y * ry
    return torch.clamp(cos_diff, 0.0, 1.0)


def box_orientation_reward(env, box_cfg) -> torch.Tensor:
    """Fused (not lift-gated) orientation reward: box +z aligned with world +z AND box +x aligned with robot heading.
    Args: env — RL env; box_cfg — box entity.
    Returns: (num_envs,) reward in [0, 2] (sum of two clamped cosines). Copied from policy_carry.
    """
    return box_upright_reward(env, box_cfg) + box_front_aligned_reward(env, box_cfg)


def box_carry_pose_bundle_reward(env, box_cfg,
                                  w_upright: float = 1.5,
                                  w_front: float = 1.5,
                                  w_chest: float = 1.5,
                                  lift_threshold: float = 0.05):
    """Fused carry-style box pose: upright + front aligned + at chest height,
    gated on the box being lifted off the table.

    Without the lift gate this term paid ~3.14/step from tick 0 (box spawns
    upright and forward-facing on the table), a free lunch that enabled the
    "throw-yourself-backwards, ep ends at 9" local optimum observed
    2026-07-27. Now: zero unless box centre is at least ``lift_threshold``
    above the tabletop.
    Args: env — RL env; box_cfg — box entity; w_upright/w_front/w_chest — per-signal weights folded in; lift_threshold — box_z clearance above tabletop required for the reward to fire.
    Returns: (num_envs,) tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    lifted = (box_z > TABLE_TOP_Z + lift_threshold).float()
    return lifted * (w_upright * box_upright_reward(env, box_cfg)
                   + w_front   * box_front_aligned_reward(env, box_cfg)
                   + w_chest   * box_at_chest_reward(env, box_cfg))


def hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg):
    """Sharp Gaussian on the sum of hand errors to commanded grasp anchors.
    Args: env — the RL env; left_hand_cfg — left hand entity; right_hand_cfg — right hand entity; box_cfg — box entity.
    Returns: (num_envs,) tensor in (0, 1].
    """
    left  = _entity_pos(env, left_hand_cfg)
    right = _entity_pos(env, right_hand_cfg)
    error = (
        torch.linalg.norm(left  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
        + torch.linalg.norm(right - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    )
    return torch.exp(-8.0 * error)


def hand_placement_bundle_reward(
    env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name,
    w_reach: float = 2.5, w_hit: float = 3.0, w_hold: float = 3.0,
    std_reach: float = 0.20, hit_threshold: float = 0.08,
):
    """Fused hand-placement for pick: reach_left + reach_right + hit_bonus + hold_box.
    All 4 signals share the same (dl, dr) hand-to-anchor distances; all were gated
    by the same stability_gate. Weighted sum multiplied by the gate is equivalent.
    Args: env — RL env; left/right_hand_cfg — hand entities; box_cfg — box entity; trunk_cfg — trunk for gate; sensor_name — feet sensor for gate; w_reach — per-side reach weight; w_hit — hit bonus weight; w_hold — hold weight; std_reach — reach Gaussian bandwidth; hit_threshold — binary hit distance.
    Returns: (num_envs,) tensor.
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    gate = stability_gate(env, trunk_cfg, sensor_name)
    two_std2 = 2.0 * std_reach * std_reach
    reach = torch.exp(-dl * dl / two_std2) + torch.exp(-dr * dr / two_std2)
    hit = ((dl < hit_threshold) & (dr < hit_threshold)).float()
    hold = torch.exp(-8.0 * (dl + dr))
    return (w_reach * reach + w_hit * hit + w_hold * hold) * gate


def box_at_lift_target_pos_reward(
    env, box_cfg,
    sigma_xy: float = 0.30,
    sigma_z_below: float = 0.30,
    sigma_z_above: float = 0.08,
):
    """Additive tracking with broad-below / narrow-above asymmetry in z.

    XY uses a symmetric broad kernel. Z uses asymmetric widths:
      * broad when box is below target so the policy gets a strong gradient
        from the squat position all the way up to the target,
      * narrow when above so overshoot becomes a sharp cliff instead of a
        soft mirror of the pulling gradient.
    Returned as 0.5 * (xy_r + z_r) — additive keeps signal alive when only
    one component is far, so the policy can still see gradient toward the
    correction on the other axis.
    """
    _ensure_cmd_buffers(env)
    box = _entity_pos(env, box_cfg)
    tgt = env.extras[_CMD_KEY + "lift_target_pos"]
    xy_dist = torch.linalg.norm(box[:, :2] - tgt[:, :2], dim=-1)
    z_diff  = box[:, 2] - tgt[:, 2]
    sigma_z = torch.where(z_diff < 0,
                          torch.full_like(z_diff, sigma_z_below),
                          torch.full_like(z_diff, sigma_z_above))
    xy_r = torch.exp(-(xy_dist ** 2) / (2.0 * sigma_xy ** 2))
    z_r  = torch.exp(-(z_diff  ** 2) / (2.0 * sigma_z  ** 2))
    return 0.5 * (xy_r + z_r)


def lift_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    hold_dist: float = 0.10,
):
    """Linear reward proportional to box height above the tabletop, gated on
    both hands being close to the commanded grasp anchors (not the box centre).

    Bug fix: previously the gate used ``dl = norm(hand - box_centre)`` with
    ``hold_dist=0.10``, but the anchors sit on the box side face at ~13 cm
    from the centre by design, so ``dl < 0.10`` never fires and lift_progress
    stayed at 0 across entire runs (mirrors the same fix in policy_drop's
    descent_progress_reward). Now: gate on hand-to-ANCHOR distance.
    """
    box = _entity_pos(env, box_cfg)
    height_norm = torch.clamp((box[:, 2] - TABLE_TOP_Z) / (CHEST_Z - TABLE_TOP_Z), 0.0, 1.0)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    holding = ((dl < hold_dist) & (dr < hold_dist)).float()
    return height_norm * holding


def box_at_lift_target_quat_reward(env, box_cfg):
    """SO(3) similarity |q_box . q_target|^2, invariant to quaternion double cover.
    Args: env — the RL env; box_cfg — box entity.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    tgt_quat = env.extras[_CMD_KEY + "lift_target_quat"]
    q_dot = (box_quat * tgt_quat).sum(dim=-1)
    return q_dot * q_dot


def upright_torso_reward(env, trunk_cfg):
    """Quat-based upright score, saturating at 1 when the torso z-axis is up.
    Args: env — the RL env; trunk_cfg — trunk entity config.
    Returns: (num_envs,) tensor in [0, 1].
    """
    entity: Entity = env.scene[trunk_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(up_z, 0.0, 1.0)


def alive_bonus(env) -> torch.Tensor:
    """Constant +1 per surviving step — dense survival signal that dominates the
    early sparse-grasp-reward phase, matching walk's implicit "stay upright to
    keep tracking" gradient. Weighted small (~0.5) so it doesn't drown out
    grasp/lift once those start firing.
    Args: env — RL env.
    Returns: (num_envs,) tensor of ones.
    """
    return torch.ones(env.num_envs, device=env.device)


def fell_over_penalty(env, trunk_cfg,
                      tilt_threshold: float = 0.5,
                      early_steps: int = 30):
    """Per-step penalty when the trunk has tilted past the threshold within the
    first ``early_steps`` policy steps.

    Mirrors drop's hand_lost_grip_penalty pattern: ``fell_over`` already
    terminates the episode (agent misses future rewards), but missing-future-
    rewards is insufficient when episodes are already short (~9 steps) and
    the "throw-yourself-backwards" local optimum still earned a few free
    rewards before the terminal. This term makes each early falling-step
    cost substantially more than a step of "cheat" reward.

    Fires when ``up_z < tilt_threshold`` (up_z = 1 for perfectly upright,
    0 for horizontal) AND ``episode_length_buf < early_steps``.
    Args: env — RL env; trunk_cfg — trunk entity; tilt_threshold — up_z value below which the trunk is considered "fallen"; early_steps — window in policy steps.
    Returns: (num_envs,) 0/1 tensor.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    fallen = (up_z < tilt_threshold).float()
    early = (env.episode_length_buf < early_steps).float()
    return fallen * early


def symmetric_pose_penalty(env, left_cfg, right_cfg):
    """Sum of squared mirrored-joint deviations across arm/wrist pairs.
    Args: env — the RL env; left_cfg — left joint set entity; right_cfg — right joint set entity.
    Returns: (num_envs,) tensor of the squared-error sum.
    """
    robot = env.scene[left_cfg.name]
    if not hasattr(env, "_sym_signs_tensor"):
        env._sym_signs_tensor = torch.tensor(
            _SYM_SIGNS, device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype,
        )
    qL = robot.data.joint_pos[:, left_cfg.joint_ids]
    qR = robot.data.joint_pos[:, right_cfg.joint_ids]
    diff = qL - env._sym_signs_tensor * qR
    return (diff * diff).sum(dim=-1)


# Task rewards gated by `stability_gate` so a falling robot earns zero task credit.
def gated_reach_reward(env, hand_cfg, box_cfg, trunk_cfg, sensor_name, side: str, std: float = 0.20):
    return reach_reward(env, hand_cfg, box_cfg, side, std) * stability_gate(env, trunk_cfg, sensor_name)


def gated_both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name, threshold: float = 0.08):
    return both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, threshold) * stability_gate(env, trunk_cfg, sensor_name)


def gated_hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name):
    return hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def gated_box_at_lift_target_pos_reward(
    env, box_cfg, trunk_cfg, sensor_name,
    sigma_xy: float = 0.30,
    sigma_z_below: float = 0.30,
    sigma_z_above: float = 0.08,
):
    return box_at_lift_target_pos_reward(
        env, box_cfg, sigma_xy=sigma_xy,
        sigma_z_below=sigma_z_below, sigma_z_above=sigma_z_above,
    ) * stability_gate(env, trunk_cfg, sensor_name)


def gated_box_at_lift_target_quat_reward(env, box_cfg, trunk_cfg, sensor_name):
    return box_at_lift_target_quat_reward(env, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def gated_lift_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    trunk_cfg, sensor_name, hold_dist: float = 0.10,
):
    return lift_progress_reward(
        env, box_cfg, left_hand_cfg, right_hand_cfg, hold_dist,
    ) * stability_gate(env, trunk_cfg, sensor_name)


# Contact penalties.
def non_hand_box_contact_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg, box_cfg,
    threshold: float = 0.22,
):
    """Proximity ramp discouraging non-hand body parts from touching the box.
    Args: env — the RL env; trunk_cfg — torso entity; left_shank_cfg — left knee entity; right_shank_cfg — right knee entity; left_hip_cfg — left hip entity; right_hip_cfg — right hip entity; box_cfg — box entity; threshold — activation distance.
    Returns: (num_envs,) tensor of accumulated penalty.
    """
    box_pos = _entity_pos(env, box_cfg)
    penalty = torch.zeros(box_pos.shape[0], device=box_pos.device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg):
        body_pos = _entity_pos(env, cfg)
        dist = torch.linalg.norm(body_pos - box_pos, dim=-1)
        penalty += torch.clamp(threshold - dist, min=0.0) / threshold
    return penalty


def body_table_collision_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    left_hand_cfg, right_hand_cfg,
    margin_xy: float = 0.02,
    z_slack: float = 0.02,
):
    """Volumetric penalty for entering the tabletop shadow extended to the floor.

    Forbidden region: xy inside the (shrunk) table footprint AND z below the
    top surface minus z_slack. Includes hands to block the "hand under the
    table" cheat that pushes the box off from below.
    Args: env — the RL env; trunk_cfg — torso entity; left_shank_cfg — left knee; right_shank_cfg — right knee; left_hip_cfg — left hip; right_hip_cfg — right hip; left_hand_cfg — left hand; right_hand_cfg — right hand; margin_xy — footprint shrink; z_slack — vertical tolerance above the top.
    Returns: (num_envs,) tensor of accumulated penetration depth.
    """
    device = env.scene["robot"].data.root_link_pos_w.device
    n_envs = env.scene["robot"].data.root_link_pos_w.shape[0]
    t_xy = torch.tensor([TABLE_X, 0.0], device=device)
    t_half_xy = torch.tensor([0.34 - margin_xy, 0.42 - margin_xy], device=device)
    top_z = TABLE_TOP_Z + TABLE_TOP_HALF - z_slack
    penalty = torch.zeros(n_envs, device=device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
                left_hand_cfg, right_hand_cfg):
        pos = _entity_pos(env, cfg)
        xy_into = t_half_xy - torch.abs(pos[:, :2] - t_xy)   # positive inside footprint
        z_below = top_z - pos[:, 2]                          # positive below top surface
        depth = torch.min(
            torch.stack([xy_into[:, 0], xy_into[:, 1], z_below], dim=-1),
            dim=-1,
        ).values
        penalty += torch.clamp(depth, min=0.0)
    return penalty


# Termination.
_BOX_FLOOR_LIMIT_Z = TABLE_TOP_Z - 0.12   # box fell off the table


def fell_over(env, box_cfg, trunk_cfg, limit_angle: float = math.radians(65.0)):
    """Terminate if the torso tilts past limit_angle or the box falls to the floor.
    Args: env — the RL env; box_cfg — box entity; trunk_cfg — trunk entity; limit_angle — max tilt in radians.
    Returns: (num_envs,) bool tensor of termination flags.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    proj_grav = entity.data.projected_gravity_b
    up_alignment = -proj_grav[:, 2]
    tilted = up_alignment < math.cos(limit_angle)
    box_pos = _entity_pos(env, box_cfg)
    box_fell = box_pos[:, 2] < _BOX_FLOOR_LIMIT_Z
    return tilted | box_fell


def box_touched_body(
    env, box_cfg, trunk_cfg,
    left_hip_cfg, right_hip_cfg,
    left_shank_cfg, right_shank_cfg,
    threshold: float = 0.15,
):
    """Termination flag when the box collapses into trunk, hips or shanks.
    Kills the "let box fall into the lap" local optimum: below `threshold` the box centre is essentially overlapping a body link, which cannot happen while the hands are actually holding it in front of the torso.
    Args: env — RL env; box_cfg — box entity; trunk/left_hip/right_hip/left_shank/right_shank_cfg — body entities; threshold — box-centre-to-body-origin distance below which the box counts as touching.
    Returns: (num_envs,) bool tensor.
    """
    box_pos = _entity_pos(env, box_cfg)
    hit = torch.zeros(box_pos.shape[0], dtype=torch.bool, device=box_pos.device)
    for cfg in (trunk_cfg, left_hip_cfg, right_hip_cfg, left_shank_cfg, right_shank_cfg):
        dist = torch.linalg.norm(_entity_pos(env, cfg) - box_pos, dim=-1)
        hit |= dist < threshold
    return hit


def body_touched_table(env, sensor_name: str = "body_table_contact",
                        force_threshold: float = 5.0):
    """Termination flag when any monitored body geom physically contacts the table.
    Ground-truth physics contact between torso / hip / thigh / shin subtree geoms and the table body — catches "slump onto tabletop", "thigh resting on edge" and "collapse forward" without any geometric threshold. Uses the sensor's netforce when available and falls back to the boolean `found` field.
    Args: env — RL env; sensor_name — ContactSensor name; force_threshold — N above which contact counts (used only when the sensor exposes force).
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


def box_fell_back(env, box_cfg,
                  high_thresh: float = 0.10,
                  settle_thresh: float = 0.03,
                  min_step: int = 30):
    """Terminate when the box, after ever rising ``high_thresh`` above the
    resting-on-table position, drops back within ``settle_thresh`` of it.

    Also gated on ``episode_length_buf >= min_step`` — a grace period during
    which the free-fall of the ``hand_bias=1.0`` spawn (box in the air) does
    NOT fire the termination. Without this, the box drops from z=0.79 to the
    tabletop in ~7-8 steps at 40 Hz, killing the episode before the policy
    can react to grip.

    Unified across both curriculum phases:
      * ``hand_bias=1.0`` (box spawns in hands): was-ever-high True from
        tick 0. After min_step, if the box has since settled back on the
        table (robot failed to grip within grace), terminate.
      * ``hand_bias=0.0`` (box on table): was-ever-high requires the robot
        to have lifted the box ``high_thresh``. If lifted then dropped
        back, terminate (min_step usually passed by then already).

    Tracks per-env running max of box z in env.extras["pick_max_box_z"]; the
    tracker is reseeded in reset_pick_command_and_object.
    Args: env — RL env; box_cfg — box entity; high_thresh — how far above the resting-on-table height counts as "lifted"; settle_thresh — how close to that resting height counts as "back on the table"; min_step — grace period in policy steps before this termination can fire.
    Returns: (num_envs,) bool tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    key = "pick_max_box_z"
    if key not in env.extras or env.extras[key].shape[0] != env.num_envs:
        env.extras[key] = torch.zeros(env.num_envs, device=env.device)
    env.extras[key] = torch.maximum(env.extras[key], box_z)
    max_z = env.extras[key]
    rested_z = TABLE_TOP_Z + BOX_HALF
    was_ever_high = max_z > rested_z + high_thresh
    now_settled   = box_z < rested_z + settle_thresh
    past_grace    = env.episode_length_buf >= min_step
    return was_ever_high & now_settled & past_grace


def body_touched_box(env, sensor_name: str = "body_box_contact",
                      force_threshold: float = 5.0):
    """Termination flag when any non-hand body geom physically contacts the box.
    Catches "leg resting on the box", "torso on box" — the geometric box_touched_body test misses these because shank origin sitting on top of the box is at delta ≈ 0.155 m, just outside the 0.15 m distance threshold.
    Args: env — RL env; sensor_name — ContactSensor name; force_threshold — N above which contact counts.
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


# Warm-start reset from a walk-terminal buffer.
_WALK_BUFFER_CACHE: dict[str, torch.Tensor] | None = None


def _load_walk_buffer(path: str, device: torch.device,
                      min_upright: float   = 0.98,
                      root_z_range: tuple[float, float] = (0.72, 0.78),
                      max_lin_vel: float   = 0.40,
                      max_ang_vel: float   = 0.60,
                      max_joint_vel: float = 4.00) -> dict[str, torch.Tensor]:
    """Load and filter the .npz walk-terminal buffer to keep only near-static entries.
    Filter (all conditions must hold) — drops mid-cycle snapshots where a foot is up
    or the robot is mid-stride, keeps only frames where the walk policy happened
    to be transitioning through a quasi-static pose:
      * trunk upright: up_z from quat_zy >= min_upright (~<18° tilt at 0.95)
      * root height inside range (default 0.70-0.80 m, brackets the G1 standing height)
      * base linear velocity magnitude <= max_lin_vel
      * base angular velocity magnitude <= max_ang_vel
      * max |joint velocity| across joints <= max_joint_vel
    If the filter leaves fewer than 1024 or n_before/20 entries, falls back to
    unfiltered (avoids resample lockup).
    Args: path — buffer file; device — torch device; min_upright/root_z_range/max_lin_vel/max_ang_vel/max_joint_vel — filter thresholds (see above).
    Returns: dict of buffer name to tensor (plus a "_path" sentinel).
    """
    global _WALK_BUFFER_CACHE
    if _WALK_BUFFER_CACHE is None or _WALK_BUFFER_CACHE.get("_path") != path:
        import numpy as _np
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"walk buffer not found at {p.resolve()}. "
                f"Run: python -m src.collect_buffer --policy-module src.policy_walk --checkpoint <walk_ckpt> --out {p}"
            )
        arr = _np.load(p)
        cache = {k: torch.from_numpy(arr[k]).float().to(device) for k in arr.files}

        # quat_zy = (w, x, y, z); up_z = 1 - 2*(x^2 + y^2) (yaw already stripped).
        q = cache["quat_zy"]
        x, y = q[:, 1], q[:, 2]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        root_z = cache["root_z"].squeeze(-1)
        lin_vel_norm = cache["lin_vel_by"].norm(dim=-1)
        ang_vel_norm = cache["ang_vel_by"].norm(dim=-1)
        joint_vel_max = cache["joint_vel"].abs().max(dim=-1).values

        keep = (
            (up_z >= min_upright)
            & (root_z >= root_z_range[0]) & (root_z <= root_z_range[1])
            & (lin_vel_norm <= max_lin_vel)
            & (ang_vel_norm <= max_ang_vel)
            & (joint_vel_max <= max_joint_vel)
        )
        n_before = q.shape[0]
        n_after  = int(keep.sum().item())
        print(f"[walk_buffer filter] kept {n_after}/{n_before} entries "
              f"({100.0 * n_after / n_before:.1f}%)")

        if n_after < max(1024, n_before // 20):
            print(f"[walk_buffer filter] fell below fallback threshold "
                  f"(max(1024, n/20) = {max(1024, n_before // 20)}) — using unfiltered buffer instead.")
            keep = torch.ones_like(keep)
            n_after = n_before

        for k in list(cache.keys()):
            if isinstance(cache[k], torch.Tensor) and cache[k].shape[0] == n_before:
                cache[k] = cache[k][keep].contiguous()
        cache["_path"] = path  # type: ignore[assignment]
        _WALK_BUFFER_CACHE = cache
    return _WALK_BUFFER_CACHE


def reset_from_home_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset the robot to KNEES_BENT_KEYFRAME with small xy/yaw jitter.
    Replaces reset_from_walk_buffer as the default reset path — the walk
    buffer contains mid-walk snapshots (one foot up, tronco tilted) that
    catapult early training into 9-step falls before the policy can react.
    This gives every env the same canonical stable start, matching drop's
    "reset to a state that already satisfies stability".
    Args: env — RL env; env_ids — envs to reset (None = all); xy_range/yaw_range — small jitter bounds; asset_cfg — robot entity config.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    asset: Entity = env.scene[asset_cfg.name]

    home_root_pos = torch.tensor(G1_HOME_ROOT_POS, device=device, dtype=torch.float32)
    home_root_rot = torch.tensor(G1_HOME_ROOT_ROT, device=device, dtype=torch.float32)
    home_joint_pos = torch.tensor(
        [G1_HOME_JOINT_POS[j] for j in asset_cfg.joint_names],
        device=device, dtype=torch.float32,
    )

    xy_jitter = sample_uniform(xy_range[0], xy_range[1], (n, 2), device=device)
    yaw_jitter = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)
    root_quat = quat_mul(q_yaw, home_root_rot.unsqueeze(0).expand(n, -1))

    origins = env.scene.env_origins[env_ids]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    root_xy = default_root_state[:, 0:2] + xy_jitter + origins[:, 0:2]
    root_z  = torch.full((n, 1), float(home_root_pos[2]), device=device) + origins[:, 2:3]
    positions = torch.cat([root_xy, root_z], dim=-1)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, root_quat], dim=-1), env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )

    joint_pos = home_joint_pos.unsqueeze(0).expand(n, -1).clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)


def reset_from_walk_buffer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    buffer_path: str,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Sample a walk-terminal state per env and write it to the robot, with jitter.
    Args: env — the RL env; env_ids — envs to reset (all if None); buffer_path — path to the .npz buffer; xy_range — additive xy jitter bounds; yaw_range — additive yaw jitter bounds in rad; asset_cfg — robot entity config.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import (
        quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform,
    )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    buf = _load_walk_buffer(buffer_path, env.device)
    total = buf["root_z"].shape[0]
    idx = torch.randint(0, total, (n,), device=env.device)

    root_z     = buf["root_z"][idx]
    quat_zy    = buf["quat_zy"][idx]
    joint_pos  = buf["joint_pos"][idx]
    # Velocities zeroed at reset: robot inherits a walking POSTURE from the
    # buffer but starts with no inertia. Prior tests showed the residual
    # base/joint velocities were catapulting the robot into an unrecoverable
    # tilt before the policy could react (~9-step episodes).
    joint_vel  = torch.zeros_like(buf["joint_vel"][idx])

    xy_jitter = sample_uniform(xy_range[0], xy_range[1], (n, 2), device=env.device)
    yaw_jitter = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=env.device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)
    root_quat = quat_mul(q_yaw, quat_zy)
    lin_vel = torch.zeros(n, 3, device=env.device)
    ang_vel = torch.zeros(n, 3, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    root_xy = default_root_state[:, 0:2] + xy_jitter + origins[:, 0:2]
    root_z_w = root_z + origins[:, 2:3]
    positions = torch.cat([root_xy, root_z_w], dim=-1)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, root_quat], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
        torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
    )


# Curriculum.
def pick_curriculum(env, env_ids):
    """Widen the reset-base jitter, object DR jitter, and unwind the pinch bias.
    pinch_bias starts at 1.0 (robot spawns already pinching the box) and ramps
    to 0.0 by ~iteration 500, after which the reset comes straight from the
    walk-buffer snapshot with no PINCH override. Thresholds picked assuming
    ~200k env-steps per iteration (NUM_ENVS=8192, rollout=24). Timing:
      * iters   0-100 (step < 20M):   scale=0.30, pinch_bias=1.0
      * iters 100-500 (step 20M-100M): scale ramps 0.30 -> 0.65, pinch 1.0 -> 0.0
      * iters   > 500 (step > 100M):  scale ramps to 1.0 by 150M, pinch=0.0
    Args: env — the RL env; env_ids — envs being updated (unused, curriculum is global).
    Returns: dict with pick_scale and pick_pinch_bias for logging.
    """
    step = int(getattr(env, "common_step_counter", 0))
    if step < 20_000_000:
        scale = 0.30
        pinch_bias = 1.0
    elif step < 100_000_000:
        t = (step - 20_000_000) / (100_000_000 - 20_000_000)
        scale = 0.30 + 0.35 * t
        pinch_bias = 1.0 - t
    elif step < 150_000_000:
        scale = 0.65 + 0.35 * (step - 100_000_000) / (150_000_000 - 100_000_000)
        pinch_bias = 0.0
    else:
        scale = 1.00
        pinch_bias = 0.0
    xy_range  = (-0.12 * scale, 0.12 * scale)
    yaw_range = (-0.30 * scale, 0.30 * scale)
    try:
        term_cfg = env.event_manager.get_term_cfg("reset_base")
        term_cfg.params["xy_range"]  = xy_range
        term_cfg.params["yaw_range"] = yaw_range
    except Exception:
        pass
    try:
        term_cfg = env.event_manager.get_term_cfg("reset_pick_cmd")
        term_cfg.params["scale"] = scale
    except Exception:
        pass
    try:
        term_cfg = env.event_manager.get_term_cfg("reset_arms_ready")
        term_cfg.params["pinch_bias"] = pinch_bias
    except Exception:
        pass
    return {"pick_scale": scale, "pick_pinch_bias": pinch_bias}


# MDP assembly.
def build_g1_pick_mdp():
    """Assemble the MDP dict (entities, sensors, obs, actions, rewards, events, terminations, curriculum).
    Args: none.
    Returns: dict of MDP components consumed by g1_pick_env_cfg.
    """
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=G1_HOME_ROOT_POS,
        rot=G1_HOME_ROOT_ROT,
        joint_pos=G1_HOME_JOINT_POS,
    )
    table_cfg = EntityCfg(spec_fn=_get_table_spec)
    box_cfg_e = EntityCfg(
        spec_fn=_get_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(BOX_START_X, 0.0, BOX_START_Z),
            joint_pos={},
        ),
    )

    g1_asset    = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    arm_asset   = SceneEntityCfg("robot", joint_names=PINCH_JOINTS, preserve_order=True)
    left_sym    = SceneEntityCfg("robot", joint_names=_LEFT_SYM_JOINTS,  preserve_order=True)
    right_sym   = SceneEntityCfg("robot", joint_names=_RIGHT_SYM_JOINTS, preserve_order=True)
    wrist_acts  = SceneEntityCfg("robot", actuator_names=(r".*wrist.*",), preserve_order=True)
    trunk       = SceneEntityCfg("robot", body_names=("torso_link",))
    pelvis      = SceneEntityCfg("robot", body_names=("pelvis",))
    left_hand   = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand  = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_shank  = SceneEntityCfg("robot", body_names=("left_knee_link",))
    right_shank = SceneEntityCfg("robot", body_names=("right_knee_link",))
    left_hip    = SceneEntityCfg("robot", body_names=("left_hip_pitch_link",))
    right_hip   = SceneEntityCfg("robot", body_names=("right_hip_pitch_link",))
    box         = SceneEntityCfg("box")

    # Sensors mirrored from walk.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    body_table_contact_cfg = ContactSensorCfg(
        name="body_table_contact",
        # Explicit list of torso + leg bodies (excludes arms so hands touching
        # the tabletop during a reach don't fire, and excludes feet which walk
        # on the ground).
        primary=ContactMatch(
            mode="body",
            pattern=r"^(torso_link"
                     r"|left_hip_pitch_link|left_hip_roll_link|left_hip_yaw_link|left_knee_link|left_ankle_pitch_link"
                     r"|right_hip_pitch_link|right_hip_roll_link|right_hip_yaw_link|right_knee_link|right_ankle_pitch_link"
                     r")$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="table", entity="table"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    body_box_contact_cfg = ContactSensorCfg(
        name="body_box_contact",
        # Same explicit torso + leg body list, with the box as secondary.
        # Catches "leg resting on box", "torso on box" — arms omitted because
        # they legitimately touch the box during grasp.
        primary=ContactMatch(
            mode="body",
            pattern=r"^(torso_link"
                     r"|left_hip_pitch_link|left_hip_roll_link|left_hip_yaw_link|left_knee_link|left_ankle_pitch_link"
                     r"|right_hip_pitch_link|right_hip_roll_link|right_hip_yaw_link|right_knee_link|right_ankle_pitch_link"
                     r")$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="box", entity="box"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot")
                    for s in ("left_foot", "right_foot")),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )

    # Observations: base proprioception + task-relative vectors + 20-D command.
    actor_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_lin_vel,      noise=Unoise(n_min=-0.2,   n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=base_ang_vel,      noise=Unoise(n_min=-0.1,   n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.5,  n_max=0.5)),
        "actions":   ObservationTermCfg(func=last_action),
        # Task-relative vectors in world frame; the policy learns to un-rotate.
        "left_hand_to_box":  ObservationTermCfg(func=body_vector, params={"source_cfg": left_hand,  "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "right_hand_to_box": ObservationTermCfg(func=body_vector, params={"source_cfg": right_hand, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "trunk_to_box":      ObservationTermCfg(func=body_vector, params={"source_cfg": trunk,      "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        # Grasp detection signal.
        "wrist_torque":      ObservationTermCfg(func=wrist_actuator_force_obs, params={"asset_cfg": wrist_acts}, noise=Unoise(n_min=-0.2, n_max=0.2)),
        # Full sanitized box state in world frame.
        "box_pos":           ObservationTermCfg(func=box_position_obs,    params={"box_cfg": box}, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "box_lin_vel":       ObservationTermCfg(func=box_lin_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_ang_vel":       ObservationTermCfg(func=box_ang_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_orientation":   ObservationTermCfg(func=box_orientation_obs, params={"box_cfg": box}),
        "box_size":          ObservationTermCfg(func=box_size_obs),
        # 20-D command: grasp offsets in box frame plus base-frame target poses.
        "cmd_grasp_offset_left":  ObservationTermCfg(func=cmd_grasp_offset_left_obs),
        "cmd_grasp_offset_right": ObservationTermCfg(func=cmd_grasp_offset_right_obs),
        "cmd_object_target_pos":  ObservationTermCfg(func=cmd_object_target_pos_obs),
        "cmd_object_target_quat": ObservationTermCfg(func=cmd_object_target_quat_obs),
        "cmd_lift_target_pos":    ObservationTermCfg(func=cmd_lift_target_pos_obs),
        "cmd_lift_target_quat":   ObservationTermCfg(func=cmd_lift_target_quat_obs),
    }
    observations = {
        "actor":  ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    # Actions.
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=G1_JOINTS,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        )
    }

    # Task rewards gated on upright torso (≤25°) AND both feet in contact; palm-inward stays ungated.
    rewards = {
        # DENSE-MONOTONIC TRIPLET (ungated, always paying since tick 0). These
        # copy walk/carry's reward shape (small Gaussian per single-dim error)
        # so PPO has a non-zero gradient before the robot ever stands or reaches.
        # Weights kept modest so the actual bundle rewards below dominate once
        # the policy graduates from "just don't fall".
        "hand_to_grasp_dense": RewardTermCfg(
            func=hand_to_grasp_target_dense, weight=1.5,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "std": 0.25},
        ),
        "upright_dense": RewardTermCfg(
            func=upright_dense, weight=1.0,
            params={"trunk_cfg": trunk, "std": 0.45},
        ),
        "hand_pinch_progress": RewardTermCfg(
            func=hand_pinch_progress, weight=2.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "std": 0.05},
        ),
        # Fused: reach_left + reach_right + hit_bonus + hold_box → hand_placement_bundle.
        # All 4 shared the same stability_gate and the same (dl, dr) hand-to-anchor
        # distances; distances now computed once. Weights folded in.
        "hand_placement": RewardTermCfg(
            func=hand_placement_bundle_reward, weight=1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "trunk_cfg": trunk,
                    "sensor_name": "feet_ground_contact",
                    "w_reach": 2.5, "w_hit": 3.0, "w_hold": 3.0,
                    "std_reach": 0.20, "hit_threshold": 0.08},
        ),
        # Fused: palm_inward_left + palm_inward_right → palm_inward_bilateral (w=0.5 each folded in).
        "palm_inward": RewardTermCfg(
            func=palm_inward_bilateral_reward, weight=1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "w_left": 0.5, "w_right": 0.5},
        ),
        # Lift target: asymmetric z-kernel (broad below to pull up from squat,
        # narrow above to punish overshoot). Weight bumped 2.0 -> 4.0.
        "box_at_lift_target_pos":  RewardTermCfg(func=gated_box_at_lift_target_pos_reward,  weight=4.0,
                                                 params={"box_cfg": box, "trunk_cfg": trunk,
                                                         "sensor_name": "feet_ground_contact",
                                                         "sigma_xy": 0.30,
                                                         "sigma_z_below": 0.30,
                                                         "sigma_z_above": 0.08}),
        # Always-on box orientation (carry-style): rewards box upright + facing
        # the robot's heading. Provides a gradient against the ±20° spawn jitter
        # so the policy learns to keep the box aligned from tick 0, not only
        # after lift.
        "box_orientation": RewardTermCfg(func=box_orientation_reward, weight=1.5, params={"box_cfg": box}),
        # Lift-gated fused: box_upright + box_front_aligned + box_at_chest → box_carry_pose_bundle.
        "box_carry_pose": RewardTermCfg(
            func=box_carry_pose_bundle_reward, weight=1.0,
            params={"box_cfg": box, "w_upright": 1.5, "w_front": 1.5, "w_chest": 1.5},
        ),
        # Linear progress on box height above the table, gated on holding.
        # Complements the Gaussian above with a dense monotonic signal so
        # every cm of lift pays, breaking the squat local optimum.
        "lift_progress": RewardTermCfg(func=gated_lift_progress_reward, weight=3.0,
                                       params={"box_cfg": box,
                                               "left_hand_cfg": left_hand,
                                               "right_hand_cfg": right_hand,
                                               "trunk_cfg": trunk,
                                               "sensor_name": "feet_ground_contact",
                                               "hold_dist": 0.10}),
        # Posture and stability.
        # Bumped 1.0 -> 5.0 each: previous training showed critic collapse
        # (Loss/value -> 0) because posture reward was too small compared to
        # task rewards to shape a non-trivial value function.
        # Fused: upright + feet_on_ground → stability_bundle (w=5.0 each folded in).
        "stability": RewardTermCfg(
            func=stability_bundle_reward, weight=1.0,
            params={"trunk_cfg": trunk, "pelvis_cfg": pelvis,
                    "sensor_name": "feet_ground_contact",
                    "w_upright": 10.0, "w_upright_pelvis": 5.0, "w_feet": 5.0},
        ),
        # Dense survival signal — pays every surviving step, dominates the
        # early sparse-grasp phase so the policy first learns "don't die",
        # then "reach and grip". Weight small so it doesn't drown out the
        # grasp / lift rewards once those start firing.
        "alive_bonus":    RewardTermCfg(func=alive_bonus, weight=1.5, params={}),
        # Weight -0.5 -> -2.0: at -0.5 the penalty was negligible during the
        # hold phase (~0.1-0.5 units of motion × 0.5 ≈ trivial vs +30 stability
        # + +5 grip_force), so the policy jittered its arms freely while
        # holding the box aloft. Bump encourages "hold still" once hands are
        # on target.
        "stay_still":     RewardTermCfg(func=stay_still_penalty,    weight=-2.0,
                                        params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                                                "box_cfg": box, "gate_dist": 0.20}),
        "symmetric_pose": RewardTermCfg(func=symmetric_pose_penalty, weight=-0.3,
                                        params={"left_cfg": left_sym, "right_cfg": right_sym}),
        # Regularization.
        # rev4 (ETH-style smoothness bundle 2026-07-25):
        # Escala Rudin/Miki/Hwangbo (ANYmal): action_rate_l2 ≈ -1e-2. Meu -3.0
        # antes era 300× — mascarava a falta de dof_acc.
        # Rev5 (2026-07-27, anti-jitter): action_rate_l2 -1e-2 -> -3e-2 and
        # dof_vel_l2 -1e-3 -> -5e-3 to punish the visible arm oscillation
        # observed while the robot held the box aloft.
        "action_rate_l2": RewardTermCfg(func=action_rate_l2,           weight=-3.0e-2),
        # NOVO: 2ª ordem (jerk) — |a_t − 2a_{t−1} + a_{t−2}|².
        "action_2nd_order": RewardTermCfg(func=action_2nd_order_l2,    weight=-1.0e-2),
        # NOVO: joint velocity — pune juntas rápidas em geral.
        "dof_vel_l2":       RewardTermCfg(func=dof_vel_l2,             weight=-5.0e-3),
        # Weight -3.0 -> -1.0: at -3, the last run showed -1.59 accumulated
        # penalty growing throughout training — hands ARE moving fast near
        # the box (which is exactly what a squeeze motion requires), so
        # the strong penalty was fighting the pick task itself.
        "hand_approach_vel": RewardTermCfg(
            func=hand_approach_velocity_penalty, weight=-1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "gate_dist": 0.25},
        ),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": g1_asset}),
        "base_ang_vel":   RewardTermCfg(func=base_ang_vel_penalty,     weight=-0.05),
        # Contact penalties.
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        # Fused: non_hand_box + body_table → body_collision_bundle (w=-2.0 each folded in).
        "body_collision": RewardTermCfg(
            func=body_collision_bundle_penalty, weight=1.0,
            params={"trunk_cfg": trunk, "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
                    "left_hip_cfg": left_hip, "right_hip_cfg": right_hip,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand, "box_cfg": box,
                    "w_non_hand_box": -2.0, "w_body_table": -2.0},
        ),
        # Explicit fall penalty tied to the fell_over termination — mirrors
        # drop's hand_lost_grip_penalty pattern. fell_over already ends the
        # episode, but that alone was insufficient against the "throw-
        # yourself-backwards to collect box_carry_pose free lunch" local
        # optimum observed 2026-07-27 (ep=9, reward<10). -50/step over the
        # first 30 steps makes each falling step cost far more than any
        # cheat-reward earned while tipping.
        "fell_over_penalty": RewardTermCfg(
            func=fell_over_penalty, weight=-50.0,
            params={"trunk_cfg": trunk, "tilt_threshold": 0.5, "early_steps": 30},
        ),
    }

    # Events: warm-buffer reset, command/box reset, domain randomization.
    events = {
        # Walk buffer with strict stability filter (see _load_walk_buffer):
        # only entries where the walk policy passed through a quasi-static
        # pose (upright, standing height, low velocities) are sampled. This
        # gives the pick reset the walking-realistic distribution (pose
        # variety, subtle joint offsets) without the mid-cycle "foot in the
        # air" snapshots that used to trigger the 9-step-fall failure mode.
        "reset_base": EventTermCfg(
            func=reset_from_walk_buffer,
            mode="reset",
            params={
                "buffer_path": WALK_BUFFER_PATH,
                "xy_range":  (-0.12, 0.12),
                "yaw_range": (-0.30, 0.30),
                "asset_cfg": g1_asset,
            },
        ),
        # Snap arms after the walk-buffer reset. Blend between PINCH_ARM_POSE
        # (hands already at the box grasp target — curriculum start) and
        # READY_ARM_POSE (hands forward-down above box — curriculum end).
        # Default pinch_bias=1.0 (start of curriculum); pick_curriculum ramps
        # it to 0.0 over training. Cuts early-training ep_length collapse.
        "reset_arms_ready": EventTermCfg(
            func=reset_arms_to_ready,
            mode="reset",
            params={"asset_cfg": arm_asset, "pinch_bias": 1.0},
        ),
        # Samples the 20-D command and writes the matching box spawn.
        "reset_pick_cmd": EventTermCfg(
            func=reset_pick_command_and_object,
            mode="reset",
            params={"scale": 0.30},   # overwritten by pick_curriculum
        ),
        "push_robot": EventTermCfg(
            func=vel_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                "roll":  (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
            }},
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
        "box_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
                "operation": "abs",
                "ranges": (BOX_MASS_MIN, BOX_MASS_MAX),
            },
        ),
    }

    # Terminations.
    terminations = {
        "time_out":  TerminationTermCfg(func=time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=fell_over, params={"box_cfg": box, "trunk_cfg": trunk},
        ),
        # Kill the "let box collapse into the lap" attractor.
        "box_touched_body": TerminationTermCfg(
            func=box_touched_body,
            params={
                "box_cfg": box, "trunk_cfg": trunk,
                "left_hip_cfg": left_hip, "right_hip_cfg": right_hip,
                "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
            },
        ),
        # Kill the "slump / lean onto the tabletop" attractor via ground-truth
        # physics contact between torso / hip / thigh / shin subtree geoms and
        # the table body.
        "body_touched_table": TerminationTermCfg(
            func=body_touched_table,
            params={"sensor_name": body_table_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
        # "Grabbed but didn't squeeze": box ever above the resting-on-table
        # position by 10 cm and now back within 3 cm of it. Fires for both
        # curriculum phases — box spawned in the hands and free-fell, or box
        # was lifted from the table and dropped back. min_step=30 grace so
        # the free-fall of the hand_bias=1.0 spawn (~7-8 steps) doesn't kill
        # the episode before the policy can grip.
        "box_fell_back": TerminationTermCfg(
            func=box_fell_back,
            params={"box_cfg": box,
                    "high_thresh": 0.10,
                    "settle_thresh": 0.03,
                    "min_step": 30},
        ),
        # Kill the "rest legs on the box" attractor via ground-truth physics
        # contact between torso / hip / thigh / shin geoms and the box body.
        "body_touched_box": TerminationTermCfg(
            func=body_touched_box,
            params={"sensor_name": body_box_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
    }

    # Curriculum widens base-reset and command ranges together.
    curriculum = {
        "reset_stages": CurriculumTermCfg(func=pick_curriculum, params={}),
    }

    return {
        "robot_cfg": robot_cfg,
        "table_cfg": table_cfg,
        "box_cfg":   box_cfg_e,
        "sensors":   (feet_ground_cfg, self_collision_cfg, body_table_contact_cfg, body_box_contact_cfg, foot_height_scan_cfg),
        "observations": observations,
        "actions":      actions,
        "rewards":      rewards,
        "events":       events,
        "terminations": terminations,
        "curriculum":   curriculum,
    }


# PPO mirror data augmentation.
# The mirror hypothesis: for a symmetric task like pick, any (obs, action)
# pair reflected across the robot's sagittal plane is equally valid training
# data. Doubling the effective sample size via mirror augmentation biases
# the policy toward left-right symmetric strategies (like wrist-pinching).
#
# Assumption: robot's sagittal plane ≈ world y=0 plane. Valid for pick
# because the robot spawns facing +x and stays there (no yaw command).
# Would break for locomotion policies where yaw varies freely.
_PICK_OBS_LAYOUT = (
    # (term_name, dim, mirror_op) — MUST match the actor_terms dict order in
    # build_pick_mdp exactly. Terms with mirror_op="pair_left" MUST be
    # immediately followed by a "pair_right" of the same dim; they swap.
    ("base_lin_vel",           3, "vec_body"),      # [x,-y,z]
    ("base_ang_vel",           3, "pseudo_body"),   # [-x,y,-z] (angular vel is a pseudovector)
    ("projected_gravity",      3, "vec_body"),
    ("joint_pos",             29, "joints"),
    ("joint_vel",             29, "joints"),
    ("actions",               29, "joints"),
    ("left_hand_to_box",       3, "pair_left"),     # swap with right, then vec_body flip
    ("right_hand_to_box",      3, "pair_right"),
    ("trunk_to_box",           3, "vec_body"),      # trunk is centerline
    ("wrist_torque",           6, "wrist_torque"),  # 3 left + 3 right actuators, swap+sign
    ("box_pos",                3, "vec_body"),      # body-relative delta
    ("box_lin_vel",            3, "vec_body"),
    ("box_ang_vel",            3, "pseudo_body"),
    ("box_orientation",        4, "quat"),          # (w, -x, y, -z)
    ("box_size",               3, "identity"),
    ("cmd_grasp_offset_left",  3, "pair_left"),
    ("cmd_grasp_offset_right", 3, "pair_right"),
    ("cmd_object_target_pos",  3, "vec_body"),
    ("cmd_object_target_quat", 4, "quat"),
    ("cmd_lift_target_pos",    3, "vec_body"),
    ("cmd_lift_target_quat",   4, "quat"),
)


def _mirror_joint_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply _JOINT_MIRROR_MAP to the last-dim = 29 joint-ordered tensor.
    Args: t — tensor (..., 29).
    Returns: tensor (..., 29) with joints swapped left↔right and signs applied.
    """
    idx  = torch.tensor([m[0] for m in _JOINT_MIRROR_MAP], device=t.device, dtype=torch.long)
    sign = torch.tensor([m[1] for m in _JOINT_MIRROR_MAP], device=t.device, dtype=t.dtype)
    return t.index_select(-1, idx) * sign


# Wrist actuators, in the order emitted by wrist_actuator_force_obs (from wrist_acts
# SceneEntityCfg with pattern r".*wrist.*"). G1_JOINTS gives us canonical order.
_WRIST_ACTUATOR_INDICES = tuple(i for i, j in enumerate(G1_JOINTS) if "wrist" in j)  # 6 indices
# Mirror map restricted to the 6 wrist actuators: swap left↔right + signs.
_WRIST_MIRROR = tuple(
    (
        _WRIST_ACTUATOR_INDICES.index(_JOINT_MIRROR_MAP[wi][0]),
        _JOINT_MIRROR_MAP[wi][1],
    )
    for wi in _WRIST_ACTUATOR_INDICES
)


def _mirror_wrist_torque(t: torch.Tensor) -> torch.Tensor:
    """Swap and sign-flip the 6 wrist actuator torques.
    Args: t — tensor (..., 6).
    Returns: tensor (..., 6) mirrored.
    """
    idx  = torch.tensor([m[0] for m in _WRIST_MIRROR], device=t.device, dtype=torch.long)
    sign = torch.tensor([m[1] for m in _WRIST_MIRROR], device=t.device, dtype=t.dtype)
    return t.index_select(-1, idx) * sign


def _mirror_obs_slice(chunk: torch.Tensor, op: str) -> torch.Tensor:
    """Apply the given mirror op to one obs slice.
    Args: chunk — tensor (..., dim); op — mirror-op string from _PICK_OBS_LAYOUT.
    Returns: mirrored slice (same shape as chunk).
    """
    if op == "identity":
        return chunk
    if op == "vec_body":
        out = chunk.clone(); out[..., 1] = -out[..., 1]; return out
    if op == "pseudo_body":
        out = chunk.clone(); out[..., 0] = -out[..., 0]; out[..., 2] = -out[..., 2]; return out
    if op == "quat":
        out = chunk.clone(); out[..., 1] = -out[..., 1]; out[..., 3] = -out[..., 3]; return out
    if op == "joints":
        return _mirror_joint_tensor(chunk)
    if op == "wrist_torque":
        return _mirror_wrist_torque(chunk)
    raise ValueError(f"unknown mirror op: {op}")


def _mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the flat actor obs tensor by walking _PICK_OBS_LAYOUT slice by slice.
    Handles 'pair_left'/'pair_right' by swapping the two contiguous chunks (both are
    body-frame vecs, so also apply vec_body after the swap).
    Args: obs — tensor (batch, total_dim).
    Returns: tensor (batch, total_dim) mirrored.
    """
    parts: list[torch.Tensor] = []
    i = 0
    idx = 0
    layout = _PICK_OBS_LAYOUT
    while idx < len(layout):
        name, dim, op = layout[idx]
        if op == "pair_left":
            # peek next; must be pair_right of the same dim.
            _, dim_r, op_r = layout[idx + 1]
            assert op_r == "pair_right" and dim_r == dim, f"pair mismatch at {name}"
            left  = obs[..., i:i + dim]
            right = obs[..., i + dim:i + 2 * dim]
            # After sagittal mirror: what was on the right is now on the left, and
            # each swapped chunk also has its y component flipped (body-frame vec).
            parts.append(_mirror_obs_slice(right, "vec_body"))
            parts.append(_mirror_obs_slice(left,  "vec_body"))
            i += 2 * dim
            idx += 2
        else:
            parts.append(_mirror_obs_slice(obs[..., i:i + dim], op))
            i += dim
            idx += 1
    assert i == obs.shape[-1], (
        f"obs mirror layout mismatch: consumed {i} dims, obs has {obs.shape[-1]}. "
        f"Verify _PICK_OBS_LAYOUT matches the actor_terms dict in build_pick_mdp."
    )
    return torch.cat(parts, dim=-1)


def mirror_pick_augmentation(env, obs=None, actions=None, obs_type: str = "policy"):
    """rsl_rl data_augmentation_func: return the mirrored (obs, actions) pair.
    Accepts plain torch tensors OR TensorDicts (rsl_rl 5.x wraps obs as TensorDict
    with actor/critic leaves that share pick's layout).
    Args: env — VecEnv wrapper (unused; kept for API); obs — (batch, obs_dim) tensor or TensorDict or None; actions — (batch, action_dim) tensor or TensorDict or None; obs_type — 'policy' or 'critic', both share the same 147-dim layout in pick.
    Returns: (mirrored_obs, mirrored_actions), each None if the input was None.
    """
    def _apply_mirror(x, fn):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return fn(x)
        # TensorDict: apply fn to every leaf tensor.
        return x.apply(fn)

    return (
        _apply_mirror(obs, _mirror_obs),
        _apply_mirror(actions, _mirror_joint_tensor),
    )


# Wired into the runner by train_loop.py when this constant is present on the module.
SYMMETRY_CFG = {
    "use_data_augmentation": True,
    "use_mirror_loss": False,
    "mirror_loss_coeff": 0.0,
    "data_augmentation_func": mirror_pick_augmentation,
}


def g1_pick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Assemble the ManagerBasedRlEnvCfg from the MDP pieces.
    Args: play — if True, disable noise/pushes and use a long episode for playback.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_pick_mdp()
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": parts["robot_cfg"], "table": parts["table_cfg"], "box": parts["box_cfg"]},
            sensors=parts["sensors"],
            num_envs=1,
            extent=2.0,
        ),
        observations=parts["observations"],
        actions=parts["actions"],
        events=parts["events"],
        rewards=parts["rewards"],
        terminations=parts["terminations"],
        curriculum=parts["curriculum"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=1.7,
            elevation=-12.0,
            azimuth=125.0,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=800,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=30),
        ),
        decimation=5,
        episode_length_s=3.25,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_pick_ppo_runner_cfg():
    """Build the RSL-RL PPO runner config used to train the pick policy.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.45, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        clip_actions=3.0,
        experiment_name="g1_pick", save_interval=50, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the pick task with mjlab; safe to call multiple times.
    Args: none.
    Returns: the registered task id.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_pick_env_cfg(),
        play_env_cfg=g1_pick_env_cfg(play=True),
        rl_cfg=g1_pick_ppo_runner_cfg(),
    )
    return TASK_ID
