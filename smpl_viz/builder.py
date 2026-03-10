"""
builder.py — load an AMASS .npz and produce everything the player needs:
  - a MuJoCo model/data pair (with scene)
  - a pre-computed qpos trajectory
  - the effective playback framerate
"""
import sys
from copy import deepcopy

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import mujoco

from smpl_viz import smpl_model_dict
from smpl_viz.joints import BODY_CHAIN, N_JOINTS
from smpl_viz.core.mesh import assemble_mjcf, _append_ground

np.set_printoptions(4, threshold=sys.maxsize, suppress=True, linewidth=np.inf)

# quaternion correction matrix (SMPL root orientation → MuJoCo convention)
_QUAT_FIX = 0.5 * np.array([
    [ 1, -1, -1, -1],
    [ 1,  1,  1, -1],
    [ 1, -1,  1,  1],
    [ 1,  1, -1,  1],
])


def _rotvec_to_wxyz(rv_batch):
    """Convert (N, 3) axis-angle to (N, 4) quaternion [w, x, y, z]."""
    q_xyzw = Rotation.from_rotvec(rv_batch).as_quat()
    return np.roll(q_xyzw, shift=1, axis=1)


def _build_smpl_mesh(gender, betas, target_height=None):
    """Compute shaped SMPL-H vertices and joint positions."""
    sd = smpl_model_dict[gender]
    verts = sd["v_template"] + (sd["shapedirs"] @ betas).reshape(-1, 3)
    verts[:] = verts[:, [2, 0, 1]]
    joints = sd["J_regressor"] @ verts

    natural_height = joints[15, 2] - (joints[7, 2] + joints[8, 2]) / 2.0
    scale = (target_height / natural_height) if target_height is not None else 1.0
    if scale != 1.0:
        verts *= scale
        joints *= scale

    return verts, joints, sd["kintree_table"], sd["weights"], sd["f"], scale


def _compute_qpos_trajectory(npz, mj_model, mj_data, scale):
    """Return (qpos_array [T, nq], framerate)."""
    raw_fps = float(
        npz["mocap_frame_rate"] if "mocap_frame_rate" in npz else npz["mocap_framerate"]
    )
    fps = raw_fps / np.sqrt(scale)

    poses = npz["poses"]
    root_rv = poses[:, :3]
    body_rv = poses[:, 3:66]
    hand_rv = poses[:, 66: N_JOINTS * 3]

    rv_all = np.hstack([root_rv, body_rv, hand_rv]).reshape(-1, N_JOINTS, 3)
    rv_all[:, 1:] = rv_all[:, 1:, [2, 0, 1]]

    pelvis_offset = mj_model.body("pelvis").pos[[1, 2, 0]]
    translation = npz["trans"] * scale + pelvis_offset

    T = len(translation)
    nq = mj_data.qpos.shape[0]
    qpos = np.zeros((T, nq))
    qpos[:, :3] = translation
    qpos[:, 3:7] = _rotvec_to_wxyz(rv_all[:, 0]) @ _QUAT_FIX

    for jid in range(1, N_JOINTS):
        jadr = mj_model.joint(mj_model.body(BODY_CHAIN[jid]).jntadr[0]).qposadr[0]
        qpos[:, jadr: jadr + 4] = _rotvec_to_wxyz(rv_all[:, jid])

    return qpos, fps


def _ground_trajectory(qpos, mj_model, mj_data, fps):
    """Shift qpos[:,2] so the feet hover just above the floor."""
    ankle_bodies = ["left_ankle", "right_ankle"]
    foot_z_samples = []
    prev_z = 0.0

    for t in range(qpos.shape[0]):
        mj_data.qpos[:] = qpos[t]
        mujoco.mj_forward(mj_model, mj_data)
        lowest = min(mj_data.body(b).xpos[2] for b in ankle_bodies)
        if lowest != np.inf and abs(prev_z - lowest) * fps < 0.01:
            foot_z_samples.append(lowest)
        prev_z = lowest

    shift = 0.0 if not foot_z_samples else (np.mean(foot_z_samples) - 0.05)
    qpos[:, 2] -= shift


def load_motion(npz_path, target_height=None):
    """Parse an AMASS .npz and return (mj_model, mj_data, qpos, fps).

    This is the single entry-point used by the player.
    """
    npz = np.load(npz_path)
    if "poses" not in npz:
        raise ValueError(f"{npz_path}: missing 'poses' array")
    if "mocap_frame_rate" not in npz and "mocap_framerate" not in npz:
        raise ValueError(f"{npz_path}: missing framerate field")

    gender = str(npz["gender"].astype(str))
    verts, joints, kintree, weights, faces, scale = _build_smpl_mesh(
        gender, npz["betas"], target_height
    )

    body_xml, body_root = assemble_mjcf(kintree, joints, weights, faces, verts)
    scene_xml = ET.tostring(
        _append_ground(deepcopy(body_root)), encoding='unicode', method='xml'
    )

    mj_model = mujoco.MjModel.from_xml_string(scene_xml)
    mj_data  = mujoco.MjData(mj_model)

    qpos, fps = _compute_qpos_trajectory(npz, mj_model, mj_data, scale)
    _ground_trajectory(qpos, mj_model, mj_data, fps)

    return mj_model, mj_data, qpos, fps
