"""Generate a broad, closed TCP command route for inspecting a saved policy.

This is a separate preview profile, not a change to training curriculum ranges.
Example::

    python -m supermimic.wf.preview_trajectory --checkpoint outputs/wf_cmd_gpu \
        --output outputs/previews/wide_sweep/trajectories --duration-s 60

The route is stored in the checkpoint's following frame. Named waypoints are
solved with arm IK within physical and actor action bounds; joint corridors and
every control-rate sample are collision checked at the settled nominal stance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .agent_references import ArmPathPlanner
from .evaluate import config_from_dict
from .goal_trajectory import FORMAT
from .math import following_frame, quat_angle, quat_error
from .simulator import BatchedSimulator


PROFILE = "preview_wide_sweep"
# Metres relative to the reset TCP, expressed along the following-frame axes.
# Side reaches turn the tool with the arm to avoid large compensating wrist rolls.
WAYPOINTS = (
    ("left", (0., .30, -.06), .8),
    ("right", (0., -.30, -.06), -.8),
    ("high", (-.05, 0., .15), 0.),
    ("low", (.05, 0., -.35), 0.),
    ("extend", (.25, 0., -.12), 0.),
    ("retract", (-.25, 0., .02), 0.),
)


def _smooth_segment(start, end, steps):
    """Quintic interpolation follows the checked straight joint corridor."""
    t = np.linspace(0., 1., steps + 1)
    phase = t**3 * (10. - 15. * t + 6. * t**2)
    return start + phase[:, None] * (end - start)


def _speeds(poses, joints, dt):
    q = torch.as_tensor(poses[:, 3:], dtype=torch.float64)
    return dict(
        max_joint_speed_rad_s=float(np.abs(np.diff(joints, axis=0)).max() / dt),
        max_tcp_speed_m_s=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=-1).max() / dt),
        max_tcp_angular_rad_s=float(quat_angle(quat_error(q[1:], q[:-1])).max() / dt),
    )


def _inverse_kinematics(planner, target, rotation, start, previous, rng, tcp_site):
    from scipy.optimize import least_squares

    def residual(q):
        # valid() updates only the isolated planning copy; collision is a hard
        # acceptance check after IK, rather than a differentiable penalty.
        planner.valid(q)
        quat, error = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(quat, planner.data.site_xmat[tcp_site])
        mujoco.mju_subQuat(error, rotation, quat)
        return np.r_[planner.data.site_xpos[tcp_site] - target, .2 * error]

    candidates = []
    for initial in (start, previous, *(rng.uniform(planner.lower, planner.upper) for _ in range(2))):
        result = least_squares(residual, initial, bounds=(planner.lower, planner.upper),
                               max_nfev=130, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q, error = result.x, residual(result.x)
        position_error, angular_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:]) / .2
        if position_error <= .015 and angular_error <= .08 and planner.valid(q):
            score = position_error + .15 * angular_error + .001 * np.linalg.norm(q - start)
            candidates.append((score, q.copy()))
    if not candidates:
        raise ValueError("preview waypoint is unreachable within the checkpoint's arm/action bounds; "
                         "reduce --span-scale")
    return min(candidates, key=lambda item: item[0])[1]


def generate(cfg, output, *, checkpoint=None, expected_physics_hash=None, duration_s=60.,
             seed=271828, level=3, span_scale=1., tcp_speed=.12, angular_speed=.5,
             joint_speed=.65, margin=.05, clearance=.01):
    """Write a dedicated, checked preview route and return its manifest.

    duration_s is a minimum: the route is slowed further when necessary to obey
    all three speed limits. The level is only a library selection tag; the
    profile and manifest explicitly identify these broader preview commands.
    """
    from .train import model_manifest

    values = (duration_s, span_scale, tcp_speed, angular_speed, joint_speed, margin, clearance)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("preview parameters must be finite")
    if min(duration_s, span_scale, tcp_speed, angular_speed, joint_speed) <= 0:
        raise ValueError("preview duration, span and speeds must be positive")
    if not 0 <= level <= 3 or not 0 <= margin <= .2 or not 0 <= clearance <= .05:
        raise ValueError("level, joint margin or clearance is outside its supported range")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"preview output is not empty: {output}")
    c = cfg.sim
    sim = BatchedSimulator(num_envs=1, seed=0, friction=1., dt=c.dt, timestep=c.timestep,
        asset_dir=cfg.asset_dir, leg_action_scale=c.leg_action_scale, arm_action_scale=c.arm_action_scale,
        wheel_action_scale=c.wheel_action_scale, arm_initial_position=c.arm_initial_position)
    try:
        provenance = model_manifest(sim.spec)
        if expected_physics_hash is not None and provenance["physics_hash"] != expected_physics_hash:
            raise ValueError("checkpoint physics differs; use its original assets/configuration")
        sim.reset(); sim.settle(steps=c.reset_settle_steps)
        nominal, spec = sim.read_state()[0].copy(), sim.spec
        action_lower = np.asarray(cfg.policy.action_lower or (-1.,) * len(spec.q0))
        action_upper = np.asarray(cfg.policy.action_upper or (1.,) * len(spec.q0))
        lower = np.maximum(spec.low, spec.q0 + spec.action_scale * action_lower)[16:] + margin
        upper = np.minimum(spec.high, spec.q0 + spec.action_scale * action_upper)[16:] - margin
        if np.any(lower >= upper):
            raise ValueError("arm action bounds are narrower than the requested joint margin")
        planner = ArmPathPlanner(sim, nominal, lower, upper, clearance=clearance)
        start = np.clip(nominal[23:29], lower, upper)
        if not planner.valid(start):
            raise ValueError("nominal arm pose is not collision free with the requested clearance")
        origin = planner.data.site_xpos[spec.tcp_site].copy()
        orientation = np.empty(4)
        mujoco.mju_mat2Quat(orientation, planner.data.site_xmat[spec.tcp_site])
        frame_pos, frame_quat = following_frame(torch.as_tensor(nominal[None], dtype=torch.float64),
                                                cfg.reward.goal_follow_frame)
        fp, fq = frame_pos[0].numpy(), frame_quat[0].numpy()
        conjugate = fq.copy(); conjugate[1:] *= -1
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, fq)
        rotation = rotation.reshape(3, 3)
        rng, route = np.random.default_rng(seed), [start.copy()]
        named = [dict(name="start", joint_waypoint=0)]
        previous = start.copy()
        for name, delta, yaw in WAYPOINTS:
            target = origin + rotation @ (np.asarray(delta) * span_scale)
            local_turn = np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
            turn, temp, desired = np.empty(4), np.empty(4), np.empty(4)
            mujoco.mju_mulQuat(temp, fq, local_turn)
            mujoco.mju_mulQuat(turn, temp, conjugate)
            mujoco.mju_mulQuat(desired, turn, orientation)
            try:
                q = _inverse_kinematics(planner, target, desired, start, previous, rng, spec.tcp_site)
            except ValueError as error:
                raise ValueError(f"preview waypoint {name}: {error}") from error
            planner.initial[23:29] = previous
            path = planner.plan(q, rng)
            if path is None:
                raise ValueError(f"no collision-free joint corridor to preview waypoint {name}")
            route.extend(np.asarray(path)[1:])
            named.append(dict(name=name, joint_waypoint=len(route) - 1,
                              requested_offset_m=(np.asarray(delta) * span_scale).tolist()))
            previous = q
        planner.initial[23:29] = previous
        path = planner.plan(start, rng)
        if path is None:
            raise ValueError("no collision-free return corridor for the preview route")
        route.extend(np.asarray(path)[1:])
        named.append(dict(name="return", joint_waypoint=len(route) - 1))
        route = np.asarray(route)

        def body_pose(q):
            if not planner.valid(q):
                raise ValueError("interpolated preview route violates planning collision clearance")
            d = planner.data
            if d.site_xpos[spec.tcp_site, 2] < cfg.reward.minimum_goal_height + .05:
                raise ValueError("preview route passes below the configured safe target height")
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, d.site_xmat[spec.tcp_site])
            relative = np.empty(4); mujoco.mju_mulQuat(relative, conjugate, quat)
            relative *= 1. if relative[0] >= 0 else -1.
            return np.r_[rotation.T @ (d.site_xpos[spec.tcp_site] - fp), relative]

        waypoint_poses = np.asarray([body_pose(q) for q in route])
        segment_duration = np.maximum(1.5, 1.95 * np.abs(np.diff(route, axis=0)).max(axis=1) / joint_speed)
        segment_duration = np.maximum(segment_duration,
            1.95 * np.linalg.norm(np.diff(waypoint_poses[:, :3], axis=0), axis=1) / tcp_speed)
        segment_duration *= max(1., duration_s / segment_duration.sum())
        for _ in range(6):
            counts = np.maximum(2, np.ceil(segment_duration / c.dt).astype(int))
            parts = [_smooth_segment(a, b, int(n)) for a, b, n in zip(route[:-1], route[1:], counts)]
            joints = np.concatenate([parts[0], *(part[1:] for part in parts[1:])])
            poses = np.asarray([body_pose(q) for q in joints])
            speeds = _speeds(poses, joints, c.dt)
            ratio = max(speeds["max_joint_speed_rad_s"] / joint_speed,
                        speeds["max_tcp_speed_m_s"] / tcp_speed,
                        speeds["max_tcp_angular_rad_s"] / angular_speed)
            if ratio <= 1. + 1e-8:
                break
            segment_duration *= ratio * 1.02
        else:
            raise RuntimeError("preview route failed to meet speed limits after time stretching")
        indices = np.r_[0, np.cumsum(counts)]
        for item in named:
            index = int(indices[item["joint_waypoint"]])
            item.update(sample_index=index, time_s=float(index * c.dt),
                        pose_follow_frame=poses[index].tolist())
        stats = dict(speeds, duration_s=(len(poses) - 1) * c.dt,
                     xyz_min_m=poses[:, :3].min(axis=0).tolist(), xyz_max_m=poses[:, :3].max(axis=0).tolist(),
                     xyz_span_m=np.ptp(poses[:, :3], axis=0).tolist(),
                     path_length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()),
                     reach_m=float(np.linalg.norm(poses[:, :3] - poses[:1, :3], axis=1).max()))
        output.mkdir(parents=True, exist_ok=True)
        path = output / "trajectories.npz"
        np.savez_compressed(path, poses=poses[None].astype(np.float32), joints=joints[None].astype(np.float32),
                            levels=np.array([level], dtype=np.int64), seeds=np.array([seed], dtype=np.int64))
        with (output / "commands.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"])
            for index, pose in enumerate(poses):
                writer.writerow([f"{index * c.dt:.6f}", *(f"{value:.9f}" for value in pose)])
        (output / "waypoints.json").write_text(json.dumps(named, indent=2, allow_nan=False) + "\n")
        manifest = dict(format=FORMAT, complete=True, profile=PROFILE,
            profile_description="Broad closed preview route; independent of training curriculum ranges",
            checkpoint=str(checkpoint) if checkpoint else None, provenance=provenance, dt=c.dt,
            steps=len(poses), follow_frame=cfg.reward.goal_follow_frame,
            pose_layout="xyz, quaternion wxyz in follow frame", levels=[level], per_level=1, seed=seed,
            arm_initial_position=c.arm_initial_position, preview_stats=stats, waypoints=named,
            options=dict(minimum_duration_s=duration_s, span_scale=span_scale, margin=margin, clearance=clearance),
            limits=dict(tcp_speed_m_s=tcp_speed, tcp_angular_rad_s=angular_speed, joint_speed_rad_s=joint_speed,
                        arm_lower_rad=lower.tolist(), arm_upper_rad=upper.tolist()),
            playback=dict(full_trajectory=True, base_command=[0., 0., 0.]),
            commands_csv_frame=cfg.reward.goal_follow_frame,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="Arm-only kinematic feasibility and collision checks at the settled nominal stance; "
                  "policy following quality is measured by playback, not guaranteed by this generator")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint file or training directory")
    parser.add_argument("--output", required=True, help="new preview trajectory directory")
    parser.add_argument("--duration-s", type=float, default=60., help="minimum route duration; speed limits can extend it")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--level", type=int, choices=range(4), default=3, help="selection tag; preview spans override training level ranges")
    parser.add_argument("--span-scale", type=float, default=1., help="scale the broad Cartesian waypoint offsets")
    parser.add_argument("--asset-dir")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint /= "ckpt_latest.pt"
    torch.set_num_threads(1)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "wf_goal_bptt_v1":
        raise ValueError("expected a WF policy checkpoint")
    cfg = config_from_dict(ck["config"])
    if args.asset_dir:
        cfg.asset_dir = args.asset_dir
    manifest = generate(cfg, args.output, checkpoint=checkpoint,
        expected_physics_hash=ck["manifest"]["physics_hash"], duration_s=args.duration_s,
        seed=args.seed, level=args.level, span_scale=args.span_scale)
    print(json.dumps(dict(output=str(Path(args.output).expanduser().resolve()), profile=PROFILE,
                          stats=manifest["preview_stats"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
