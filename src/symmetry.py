##########################################################################################
# Project    : unitree-g1-humanoid-pick                                                  #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-pick                    #
##########################################################################################
"""Left-right symmetry (sagittal-plane mirror) for the G1 pick task.

Same joint permutation / sign convention as walk and carry (verified against
the G1 MJCF: roll axes are +x, yaw axes are +z, pitch axes are +y; roll and
yaw joints flip sign under sagittal mirror, pitch joints do not). The pick
task adds 6 command observations on top of the carry layout (grasp offsets +
object/lift target pose), plus separate box_lin_vel and box_ang_vel; each
term gets its own mirror rule.

Assumes robot yaw ~= 0 during pick (robot faces the table along +x for the
whole episode) so world-y mirror agrees with body-y mirror.

Numerical operations on the concatenated obs vector:
- body-frame vector (base_lin_vel, projected_gravity, cmd_*_pos): flip y
- body-frame pseudo-vector (base_ang_vel): flip x, z
- world-frame vector (body_vector's, box_pos, box_lin_vel): flip y
- world-frame pseudo-vector (box_ang_vel): flip x, z
- box_orientation / cmd_*_quat: (w, -x, y, -z)
- wrist_torque (6-D actuator order): swap L<->R with roll/yaw sign flip
- cmd_grasp_offset_left <-> cmd_grasp_offset_right: swap + y flip
- box_size: unchanged (positive scalars)
"""

from __future__ import annotations

import numpy as np
import torch

# Joint mapping.
G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",           "left_wrist_roll_joint",     "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",          "right_wrist_roll_joint",    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_perm = list(range(29))
_perm[0:6] = list(range(6, 12))
_perm[6:12] = list(range(0, 6))
_perm[15:22] = list(range(22, 29))
_perm[22:29] = list(range(15, 22))
G1_MIRROR_PERM = tuple(_perm)

G1_MIRROR_SIGN_FLIP_IDX = (1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28)


def _assert_joint_order():
    """Verify runtime G1_JOINTS matches the hard-coded G1_29DOF_JOINT_ORDER."""
    from src.policy_pick import G1_JOINTS
    if tuple(G1_JOINTS) != G1_29DOF_JOINT_ORDER:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(G1_JOINTS, G1_29DOF_JOINT_ORDER)) if a != b]
        raise AssertionError(
            "G1_JOINTS order in policy_pick differs from src.symmetry G1_29DOF_JOINT_ORDER. "
            f"Mismatches (idx, actual, expected): {mismatches[:5]} ..."
        )


_assert_joint_order()


def _switch_g1_joints_left_right(x: torch.Tensor) -> torch.Tensor:
    """Permute L<->R on the last dim of a (..., 29) tensor and flip roll/yaw signs.
    Args: x: tensor whose last dim is exactly 29 (joint_pos, joint_vel or actions).
    Returns: mirrored tensor, same shape.
    """
    perm = torch.as_tensor(G1_MIRROR_PERM, device=x.device, dtype=torch.long)
    out = torch.index_select(x, dim=-1, index=perm)
    signs = torch.ones(29, device=x.device, dtype=x.dtype)
    signs[list(G1_MIRROR_SIGN_FLIP_IDX)] = -1.0
    return out * signs


# wrist_torque mirror (6-D). Entity order for actuator_names=(r".*wrist.*",) preserve_order=True is [L_roll, R_roll, L_pitch, L_yaw, R_pitch, R_yaw] on G1 in mjlab (verified 2026-08-17). Swap L<->R and flip roll/yaw sign.
_WRIST_PERM  = (1, 0, 4, 5, 2, 3)
_WRIST_SIGNS = (-1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def _mirror_wrist_torque(x: torch.Tensor) -> torch.Tensor:
    """Mirror 6-D wrist torque obs. Args: x: (..., 6). Returns: same shape."""
    perm = torch.as_tensor(_WRIST_PERM, device=x.device, dtype=torch.long)
    signs = torch.as_tensor(_WRIST_SIGNS, device=x.device, dtype=x.dtype)
    return torch.index_select(x, dim=-1, index=perm) * signs


# Pick obs layout (147 dims).
PICK_OBS_DIM = 147

PICK_OBS_LAYOUT = {
    "base_lin_vel":          (0,   3),
    "base_ang_vel":          (3,   6),
    "projected_gravity":     (6,   9),
    "joint_pos":             (9,  38),
    "joint_vel":             (38, 67),
    "actions":               (67, 96),
    "left_hand_to_box":      (96,  99),
    "right_hand_to_box":     (99, 102),
    "trunk_to_box":          (102, 105),
    "wrist_torque":          (105, 111),
    "box_pos":               (111, 114),
    "box_lin_vel":           (114, 117),
    "box_ang_vel":           (117, 120),
    "box_orientation":       (120, 124),
    "box_size":              (124, 127),
    "cmd_grasp_offset_left":  (127, 130),
    "cmd_grasp_offset_right": (130, 133),
    "cmd_object_target_pos":  (133, 136),
    "cmd_object_target_quat": (136, 140),
    "cmd_lift_target_pos":    (140, 143),
    "cmd_lift_target_quat":   (143, 147),
}
assert PICK_OBS_LAYOUT["cmd_lift_target_quat"][1] == PICK_OBS_DIM

# Per-term element-wise sign vector for simple flip terms.
_VEC_Y_FLIP        = (1.0, -1.0, 1.0)
_PSEUDOVEC_Y_FLIP  = (-1.0, 1.0, -1.0)
_QUAT_BODY_MIRROR  = (1.0, -1.0, 1.0, -1.0)
_TWIST_MIRROR      = (1.0, -1.0, -1.0)  # (vx, vy, wz)

PICK_OBS_SIGNS = {
    "base_lin_vel":          _VEC_Y_FLIP,
    "base_ang_vel":          _PSEUDOVEC_Y_FLIP,
    "projected_gravity":     _VEC_Y_FLIP,
    "trunk_to_box":          _VEC_Y_FLIP,
    "box_pos":               _VEC_Y_FLIP,
    "box_lin_vel":           _VEC_Y_FLIP,
    "box_ang_vel":           _PSEUDOVEC_Y_FLIP,
    "box_orientation":       _QUAT_BODY_MIRROR,
    "box_size":              (1.0, 1.0, 1.0),  # unchanged
    "cmd_object_target_pos":  _VEC_Y_FLIP,
    "cmd_object_target_quat": _QUAT_BODY_MIRROR,
    "cmd_lift_target_pos":    _VEC_Y_FLIP,
    "cmd_lift_target_quat":   _QUAT_BODY_MIRROR,
}


def mirror_pick_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a concatenated pick obs tensor.
    Args: obs: (..., k*PICK_OBS_DIM) tensor for k >= 1 stacked timesteps.
    Returns: mirrored tensor, same shape.
    """
    assert obs.shape[-1] % PICK_OBS_DIM == 0, (
        f"Last dim {obs.shape[-1]} not multiple of {PICK_OBS_DIM}"
    )
    n_stack = obs.shape[-1] // PICK_OBS_DIM
    lead = obs.shape[:-1]
    x = obs.clone().view(*lead, n_stack, PICK_OBS_DIM)
    for term, (a, b) in PICK_OBS_LAYOUT.items():
        if term in {"joint_pos", "joint_vel", "actions"}:
            x[..., a:b] = _switch_g1_joints_left_right(x[..., a:b])
        elif term == "wrist_torque":
            x[..., a:b] = _mirror_wrist_torque(x[..., a:b])
        elif term == "left_hand_to_box":
            lh_a, lh_b = PICK_OBS_LAYOUT["left_hand_to_box"]
            rh_a, rh_b = PICK_OBS_LAYOUT["right_hand_to_box"]
            lh = x[..., lh_a:lh_b].clone()
            rh = x[..., rh_a:rh_b].clone()
            signs = torch.tensor(_VEC_Y_FLIP, device=obs.device, dtype=obs.dtype)
            x[..., lh_a:lh_b] = rh * signs
            x[..., rh_a:rh_b] = lh * signs
        elif term == "right_hand_to_box":
            pass  # handled with left_hand_to_box
        elif term == "cmd_grasp_offset_left":
            gl_a, gl_b = PICK_OBS_LAYOUT["cmd_grasp_offset_left"]
            gr_a, gr_b = PICK_OBS_LAYOUT["cmd_grasp_offset_right"]
            gl = x[..., gl_a:gl_b].clone()
            gr = x[..., gr_a:gr_b].clone()
            signs = torch.tensor(_VEC_Y_FLIP, device=obs.device, dtype=obs.dtype)
            x[..., gl_a:gl_b] = gr * signs
            x[..., gr_a:gr_b] = gl * signs
        elif term == "cmd_grasp_offset_right":
            pass  # handled with cmd_grasp_offset_left
        else:
            signs = torch.tensor(PICK_OBS_SIGNS[term], device=obs.device, dtype=obs.dtype)
            x[..., a:b] = x[..., a:b] * signs
    return x.view(*lead, obs.shape[-1])


def mirror_pick_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror pick actions (29-D joint targets).
    Args: actions: (..., 29) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right(actions)


# Snapshot buffer mirror (numpy). Used by src.collect_buffer to double a terminal-state buffer by appending the sagittal mirror of every sample. Operates on the yaw-aligned schema dumped by collect_buffer._snapshot: root_z, quat_zy, lin_vel_by, ang_vel_by, joint_pos, joint_vel (+ optionally box_pos_by, box_quat_by, box_lin_vel_by, box_ang_vel_by).


def _np_switch_joints(x: "np.ndarray") -> "np.ndarray":
    """Numpy version of _switch_g1_joints_left_right for (..., 29) arrays."""
    out = x[..., list(G1_MIRROR_PERM)].copy()
    for i in G1_MIRROR_SIGN_FLIP_IDX:
        out[..., i] *= -1.0
    return out


def _np_mirror_body_quat(q: "np.ndarray") -> "np.ndarray":
    """Sagittal mirror of a body-frame quaternion (w, x, y, z) -> (w, -x, y, -z)."""
    m = q.copy()
    m[..., 1] *= -1.0
    m[..., 3] *= -1.0
    return m


def _np_mirror_vec(v: "np.ndarray") -> "np.ndarray":
    """Regular 3-vector sagittal mirror (x, y, z) -> (x, -y, z)."""
    m = v.copy()
    m[..., 1] *= -1.0
    return m


def _np_mirror_pseudovec(w: "np.ndarray") -> "np.ndarray":
    """Pseudo-vector sagittal mirror (wx, wy, wz) -> (-wx, wy, -wz)."""
    m = w.copy()
    m[..., 0] *= -1.0
    m[..., 2] *= -1.0
    return m


_SNAPSHOT_MIRROR_OPS = {
    "root_z":         lambda a: a.copy(),
    "quat_zy":        _np_mirror_body_quat,
    "lin_vel_by":     _np_mirror_vec,
    "ang_vel_by":     _np_mirror_pseudovec,
    "joint_pos":      _np_switch_joints,
    "joint_vel":      _np_switch_joints,
    "box_pos_by":     _np_mirror_vec,
    "box_quat_by":    _np_mirror_body_quat,
    "box_lin_vel_by": _np_mirror_vec,
    "box_ang_vel_by": _np_mirror_pseudovec,
}


def mirror_snapshot_buffer(buf: dict) -> dict:
    """Return a dict with the same keys where each array is doubled: [original; mirror].
    Args: buf: dict from a collect_buffer .npz.
    Returns: same keys, arrays of shape (2*N, ...).
    """
    out = {}
    for k, v in buf.items():
        op = _SNAPSHOT_MIRROR_OPS.get(k)
        if op is None:
            out[k] = np.concatenate([v, v], axis=0)
        else:
            out[k] = np.concatenate([v, op(v)], axis=0)
    return out


# rsl_rl entrypoint.
@torch.no_grad()
def mirror_g1_pick(env=None, obs=None, actions=None):
    """rsl_rl symmetry_cfg entrypoint. Returns doubled batch [original; mirrored].
    Args: env: mjlab env (unused, kept for the rsl_rl API contract); obs: TensorDict with per-group keys, or plain tensor; actions: (N, 29) tensor, or None.
    Returns: (obs_out, actions_out); each is None if the corresponding input is None, else doubled along the batch dim.
    """
    obs_out = None
    actions_out = None

    if obs is not None:
        if hasattr(obs, "keys") and hasattr(obs, "batch_size"):
            batch_size = obs.batch_size[0]
            obs_out = obs.repeat(2)
            for key in obs.keys():
                obs_out[key][:batch_size] = obs[key]
                obs_out[key][batch_size:] = mirror_pick_obs(obs[key])
        else:
            obs_mirror = mirror_pick_obs(obs)
            obs_out = torch.cat([obs, obs_mirror], dim=0)

    if actions is not None:
        actions_mirror = mirror_pick_actions(actions)
        actions_out = torch.cat([actions, actions_mirror], dim=0)

    return obs_out, actions_out
