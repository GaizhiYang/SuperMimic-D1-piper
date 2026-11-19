"""Prepare a complete, collision-checked TCP command path for policy previews.

Only an independent planning copy performs forward kinematics. The physical
simulator remains at its settled fixture; its policy must execute every motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .agent_references import ArmPathPlanner
from .math import following_frame, quat_apply_inverse, quat_conjugate, quat_mul, quat_normalize


@dataclass
class PreviewTrajectory:
    poses: torch.Tensor
    levels: torch.Tensor
    count: int
    length: int
    fingerprint: str
    steps: int
    lead_in_steps: int
    metadata: dict


def rank_trajectories(library, level) -> list[int]:
    """Rank paths at exactly ``level`` by XYZ bounding-box diagonal, then ID."""
    poses = library.poses.detach().cpu().numpy()
    levels = library.levels.detach().cpu().numpy()
    ids = np.flatnonzero(levels == level)
    if not len(ids):
        raise ValueError(f"trajectory library has no paths at level {level}")
    extent = np.linalg.norm(np.ptp(poses[ids, :, :3], axis=1), axis=-1)
    return ids[np.lexsort((ids, -extent))].tolist()


def smooth_joint_approach(start, end, steps: int) -> np.ndarray:
    """Quintic joint interpolation with zero endpoint velocity/acceleration."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (6,) or end.shape != (6,) or not np.isfinite([start, end]).all():
        raise ValueError("approach endpoints must be finite six-joint arm poses")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("approach steps must be a positive integer")
    t = np.linspace(0., 1., steps + 1)[:, None]
    weight = t**3 * (10. + t * (-15. + 6. * t))
    result = start + weight * (end - start)
    result[0], result[-1] = start, end
    return result


def resample_joints(joints, speed: float) -> np.ndarray:
    """Slow sampled joints without overshoot, retaining both exact endpoints."""
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 6 or len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError("source joints must be a finite (samples >= 2, 6) array")
    if not math.isfinite(speed) or not 0. < speed <= 1.:
        raise ValueError("preview speed must be finite and in (0, 1]")
    intervals = math.ceil((len(joints) - 1) / speed)
    position = np.linspace(0., len(joints) - 1, intervals + 1)
    original = np.arange(len(joints))
    return np.column_stack([np.interp(position, original, joints[:, i]) for i in range(6)])


def prepare_preview(library, directory, trajectory_id, sim, *, speed=.75, lead_in_s=4., hold_s=2.):
    """Build one full path, its smooth approach, final hold and schedule padding.

    Use ``segment_steps=result.steps + 1`` and ``update_steps=1`` with a
    :class:`TrajectorySchedule`. Since ``length=steps+2``, segment selection
    always starts at sample zero and never switches within the requested run.
    """
    if not isinstance(trajectory_id, (int, np.integer)) or not 0 <= trajectory_id < library.count:
        raise ValueError("trajectory ID is outside the library")
    if not math.isfinite(speed) or not 0. < speed <= 1.:
        raise ValueError("preview speed must be finite and in (0, 1]")
    if not math.isfinite(lead_in_s) or lead_in_s <= 0.:
        raise ValueError("preview lead-in duration must be finite and positive")
    if not math.isfinite(hold_s) or hold_s < 0.:
        raise ValueError("preview hold duration must be finite and nonnegative")
    manifest = library.manifest
    dt = float(manifest["dt"])
    if not math.isfinite(dt) or dt <= 0. or not np.isclose(dt, sim.dt):
        raise ValueError("preview trajectory and fixture control periods differ")
    path = Path(directory) / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("trajectory library bytes changed after validation")
    with np.load(path, allow_pickle=False) as data:
        if "joints" not in data:
            raise ValueError("trajectory library has no joint path for preview preparation")
        joints = data["joints"]
    if joints.shape != (library.count, library.length, 6) or not np.isfinite(joints).all():
        raise ValueError("trajectory library joints must be finite with shape (paths, samples, 6)")
    initial = np.asarray(sim.read_state(), dtype=float)
    if initial.shape != (1, 57) or not np.isfinite(initial).all():
        raise ValueError("preview preparation requires one finite settled simulator fixture")
    initial = initial[0].copy()
    lead_steps, hold_steps = math.ceil(lead_in_s / dt), math.ceil(hold_s / dt)
    source = resample_joints(joints[trajectory_id], speed)
    approach = smooth_joint_approach(initial[23:29], source[0], lead_steps)
    commanded = np.concatenate((approach, source[1:], np.repeat(source[-1:], hold_steps, axis=0)))
    steps = len(commanded) - 1
    commanded = np.concatenate((commanded, commanded[-1:]))

    spec = sim.spec
    lower = np.maximum(spec.low, spec.q0 - spec.action_scale)[16:]
    upper = np.minimum(spec.high, spec.q0 + spec.action_scale)[16:]
    clearance = float(manifest.get("options", {}).get("clearance", .01))
    planner = ArmPathPlanner(sim, initial, lower, upper, clearance=clearance)
    frame_pos, frame_quat = following_frame(torch.as_tensor(initial, dtype=torch.float64), manifest["follow_frame"])
    poses = []
    for index, q in enumerate(commanded):
        if not planner.valid(q):
            raise ValueError(f"preview trajectory {trajectory_id} is not collision-free or violates joint bounds "
                             f"at sample {index} (t={index * dt:.2f}s)")
        position = planner.data.site_xpos[spec.tcp_site].copy()
        if position[2] < .1:
            raise ValueError(f"preview trajectory {trajectory_id} TCP is below the 0.10 m goal-height limit "
                             f"at sample {index} (height={position[2]:.4f} m)")
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, planner.data.site_xmat[spec.tcp_site])
        local = quat_apply_inverse(frame_quat, torch.as_tensor(position) - frame_pos)
        relative = quat_normalize(quat_mul(quat_conjugate(frame_quat), torch.as_tensor(quaternion)))
        if relative[0] < 0:
            relative = -relative
        poses.append(torch.cat((local, relative)))
    poses = torch.stack(poses).to(dtype=torch.float32).unsqueeze(0)
    level = int(library.levels[trajectory_id])
    span = torch.amax(poses[0, :, :3], dim=0) - torch.amin(poses[0, :, :3], dim=0)
    fingerprint = hashlib.sha256(library.fingerprint.encode() + poses.numpy().tobytes()).hexdigest()
    metadata = dict(trajectory_id=int(trajectory_id), level=level, xyz_span_m=span.tolist(),
        duration_s=steps * dt, dt=dt, steps=steps, speed=float(speed),
        lead_in_s=lead_steps * dt, lead_in_steps=lead_steps, hold_s=hold_steps * dt,
        source_duration_s=(library.length - 1) * dt, source_library=str(Path(directory).resolve()),
        source_fingerprint=library.fingerprint, fingerprint=fingerprint,
        following_frame=manifest["follow_frame"], minimum_goal_height_m=.1,
        clearance_m=clearance, validation="every joint sample checked in a planning copy of the settled fixture")
    return PreviewTrajectory(poses=poses, levels=torch.tensor([level], dtype=torch.long),
        count=1, length=poses.shape[1], fingerprint=fingerprint, steps=steps,
        lead_in_steps=lead_steps, metadata=metadata)
