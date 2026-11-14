"""Body-frame TCP target trajectories: offline library + per-environment schedule.

Library (offline, CPU)
    For every curriculum level, smooth arm joint-space trajectories are sampled
    around the reset arm pose (amplitude, joint speed and TCP speed grow with
    the level), every control-rate sample is collision-checked on the settled
    nominal robot, and the TCP pose is stored in the goal-following frame
    (``reward.goal_follow_frame``). Each trajectory is therefore reachable by the
    arm alone from the nominal stance; legs and wheels remain free to help.

Schedule (training and evaluation)
    Every ``segment_steps`` (Ta) control steps an environment draws a new window
    of ``segment_steps`` samples from the library (choosing, among K random
    candidates, the one starting closest to the current TCP pose). Every
    ``update_steps`` (Tb) control steps the window's current body-frame pose is
    written into the TCP part ``goal[..., :7]`` of the command:

    * ``mode = 1`` (driving): relative to the CURRENT following frame, so the
      target moves with the base (between updates ``step_goal`` transports it);
    * ``mode = 0`` (stationary): relative to the frame frozen at segment start,
      i.e. a world-fixed moving target.

    The Actor input format is unchanged (11-D command): only the command
    values change over time. The deployed policy needs no library.

Usage:
    python -m supermimic.wf.goal_trajectory --config configs/examples/wf_cmd_gpu4096.json \
        --output outputs/wf_tcp_trajectories --per-level 256 --workers 32
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .math import following_frame, quat_angle, quat_apply, quat_apply_inverse, quat_conjugate, quat_error, quat_mul, quat_normalize

FORMAT = "wf_tcp_trajectories_v1"
ARM_SPREAD = np.array([.60, .45, .50, .60, .50, .70])
LEVEL_AMPLITUDE = (.15, .35, .60, 1.0)
LEVEL_JOINT_SPEED = (.25, .40, .60, .80)      # rad/s, per joint
LEVEL_TCP_SPEED = (.04, .08, .12, .16)        # m/s
LEVEL_TCP_ANGULAR = (.20, .35, .50, .70)      # rad/s


# ----------------------------------------------------------------- generation
_W: dict = {}


def _init_worker(raw_config, options):
    from .agent_references import ArmPathPlanner
    from .evaluate import config_from_dict
    from .simulator import BatchedSimulator
    torch.set_num_threads(1)
    cfg = config_from_dict(raw_config)
    c = cfg.sim
    sim = BatchedSimulator(num_envs=1, seed=0, friction=1., dt=c.dt, timestep=c.timestep, asset_dir=cfg.asset_dir,
        leg_action_scale=c.leg_action_scale, arm_action_scale=c.arm_action_scale,
        wheel_action_scale=c.wheel_action_scale, arm_initial_position=c.arm_initial_position)
    sim.reset(); sim.settle(steps=c.reset_settle_steps)
    nominal = sim.read_state()[0].copy()
    s = sim.spec
    lower = np.maximum(s.low, s.q0 - s.action_scale)[16:] + options["margin"]
    upper = np.minimum(s.high, s.q0 + s.action_scale)[16:] - options["margin"]
    planner = ArmPathPlanner(sim, nominal, lower, upper, clearance=options["clearance"])
    frame_pos, frame_quat = following_frame(torch.as_tensor(nominal[None], dtype=torch.float64),
                                            cfg.reward.goal_follow_frame)
    _W.update(cfg=cfg, sim=sim, planner=planner, nominal=nominal, lower=lower, upper=upper,
              frame=(frame_pos[0].numpy(), frame_quat[0].numpy()), options=options)


def _tcp_body_pose(q):
    """TCP pose in the nominal following frame for arm joints q (planning copy)."""
    planner, s = _W["planner"], _W["sim"].spec
    if not planner.valid(q):
        return None
    d = planner.data
    position = d.site_xpos[s.tcp_site].copy()
    if position[2] < _W["cfg"].reward.minimum_goal_height + .05:
        return None
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, d.site_xmat[s.tcp_site])
    fp, fq = _W["frame"]
    p = torch.as_tensor(position, dtype=torch.float64); r = torch.as_tensor(quat, dtype=torch.float64)
    fpt, fqt = torch.as_tensor(fp), torch.as_tensor(fq)
    local = quat_apply_inverse(fqt, p - fpt)
    rel = quat_normalize(quat_mul(quat_conjugate(fqt), r))
    rel = torch.where(rel[..., :1] < 0, -rel, rel)
    return np.r_[local.numpy(), rel.numpy()]


def _make(job):
    from scipy.interpolate import CubicSpline
    seed, level = job
    o, cfg = _W["options"], _W["cfg"]
    rng = np.random.default_rng(seed)
    dt, steps = cfg.sim.dt, o["steps"]
    lower, upper = _W["lower"], _W["upper"]
    center = _W["nominal"][23:29]
    amplitude = ARM_SPREAD * LEVEL_AMPLITUDE[level]
    vmax, tcp_v, tcp_w = LEVEL_JOINT_SPEED[level], LEVEL_TCP_SPEED[level], LEVEL_TCP_ANGULAR[level]
    record = dict(seed=int(seed), level=int(level))

    def waypoint():
        for _ in range(200):
            q = np.clip(center + rng.uniform(-1, 1, 6) * amplitude, lower, upper)
            if _tcp_body_pose(q) is not None:
                return q
        return None

    stretch = 1.
    for _ in range(4):
        points, times = [waypoint()], [0.]
        if points[0] is None:
            return dict(record, accepted=False, reason="no collision-free waypoint")
        tries = 0
        while times[-1] < steps * dt:
            tries += 1
            if tries > 400:
                return dict(record, accepted=False, reason="no collision-free waypoint edge")
            q = waypoint()
            if q is None:
                return dict(record, accepted=False, reason="no collision-free waypoint")
            if not _W["planner"].edge(points[-1], q):
                continue
            duration = max(o["min_segment_s"], 1.5 * np.abs(q - points[-1]).max() / vmax) * stretch
            points.append(q); times.append(times[-1] + duration)
        spline = CubicSpline(np.asarray(times), np.asarray(points), axis=0, bc_type="clamped")
        t = np.arange(steps) * dt
        joints = spline(t)
        joint_speed = np.abs(spline(t, 1)).max()
        poses = []
        for q in joints:
            pose = _tcp_body_pose(np.clip(q, lower, upper)) if np.all(q >= lower) and np.all(q <= upper) else None
            if pose is None:
                break
            poses.append(pose)
        if len(poses) < steps:
            return dict(record, accepted=False, reason="spline left the collision-free joint region")
        poses = np.asarray(poses)
        linear = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=-1).max() / dt
        qt = torch.as_tensor(poses[:, 3:], dtype=torch.float64)
        angular = float(quat_angle(quat_error(qt[1:], qt[:-1])).max()) / dt
        ratio = max(joint_speed / vmax, linear / tcp_v, angular / tcp_w)
        if ratio <= 1.:
            reach = float(np.linalg.norm(poses[:, :3] - poses[:1, :3], axis=-1).max())
            return dict(record, accepted=True, poses=poses.astype(np.float32), joints=joints.astype(np.float32),
                        max_joint_speed=float(joint_speed), max_tcp_speed=float(linear),
                        max_tcp_angular_speed=angular, reach_m=reach, waypoints=len(points))
        stretch *= ratio * 1.1
    return dict(record, accepted=False, reason="speed limits not met after time stretching")


def generate(cfg, raw_config, output, *, per_level, levels, seed, workers, **options):
    from .simulator import BatchedSimulator
    from .train import model_manifest
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("trajectory library already exists; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    options["steps"] = int(round(options.pop("duration_s") / cfg.sim.dt))
    probe = BatchedSimulator(num_envs=1, friction=1., dt=cfg.sim.dt, timestep=cfg.sim.timestep, asset_dir=cfg.asset_dir,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale, arm_initial_position=cfg.sim.arm_initial_position)
    provenance = model_manifest(probe.spec)
    probe.close()
    jobs = [(seed + 100_000 * level + i, level) for level in levels for i in range(per_level * 4)]
    accepted = {level: [] for level in levels}
    rejected = []
    started = time.monotonic()
    ctx = mp.get_context("fork" if os.name == "posix" else "spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(raw_config, options)) as pool:
        for result in pool.imap(_make, jobs, chunksize=4):
            level = result["level"]
            if len(accepted[level]) >= per_level:
                continue
            if result["accepted"]:
                accepted[level].append(result)
            else:
                rejected.append(result)
            if all(len(v) >= per_level for v in accepted.values()):
                pool.terminate()
                break
    rows = [r for level in levels for r in accepted[level]]
    if any(len(accepted[level]) < per_level for level in levels):
        raise RuntimeError(f"insufficient trajectories: { {k: len(v) for k, v in accepted.items()} }")
    poses = np.stack([r["poses"] for r in rows])
    path = output / "trajectories.npz"
    np.savez_compressed(path, poses=poses, joints=np.stack([r["joints"] for r in rows]),
                        levels=np.asarray([r["level"] for r in rows], dtype=np.int64),
                        seeds=np.asarray([r["seed"] for r in rows], dtype=np.int64))
    stats = {str(level): dict(count=len(accepted[level]),
                              reach_m_p50=float(np.median([r["reach_m"] for r in accepted[level]])),
                              max_tcp_speed_m_s=float(max(r["max_tcp_speed"] for r in accepted[level])),
                              max_tcp_angular_rad_s=float(max(r["max_tcp_angular_speed"] for r in accepted[level])))
             for level in levels}
    manifest = dict(format=FORMAT, complete=True, provenance=provenance, dt=cfg.sim.dt,
        steps=options["steps"], follow_frame=cfg.reward.goal_follow_frame, pose_layout="xyz, quaternion wxyz in follow frame",
        levels=list(levels), per_level=per_level, seed=seed, options=options, stats=stats,
        level_limits=dict(amplitude=LEVEL_AMPLITUDE, joint_speed=LEVEL_JOINT_SPEED, tcp_speed=LEVEL_TCP_SPEED,
                          tcp_angular=LEVEL_TCP_ANGULAR),
        arm_initial_position=cfg.sim.arm_initial_position, rejected=len(rejected),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="arm-only kinematic feasibility at the settled nominal stance; not a whole-body dynamic guarantee")
    with (output / "rejected.jsonl").open("w") as log:
        for r in rejected:
            log.write(json.dumps(r) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(trajectories=len(rows), rejected=len(rejected), stats=stats,
                          elapsed_s=round(time.monotonic() - started, 1))), flush=True)
    return manifest


# ------------------------------------------------------------------- runtime
class TrajectoryLibrary:
    def __init__(self, directory, task, physics_hash, dt, device="cpu"):
        directory = Path(directory)
        raw = (directory / "manifest.json").read_bytes()
        m = json.loads(raw)
        if m.get("format") != FORMAT or not m.get("complete"):
            raise ValueError("expected a complete WF TCP trajectory library")
        if m["provenance"]["physics_hash"] != physics_hash:
            raise ValueError("trajectory library was generated for different robot physics")
        if m["follow_frame"] != task.cfg.goal_follow_frame:
            raise ValueError("trajectory follow frame differs from reward.goal_follow_frame")
        if not np.isclose(m["dt"], dt):
            raise ValueError("trajectory sample period differs from the control period")
        path = directory / "trajectories.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != m["sha256"]:
            raise ValueError("trajectory library bytes changed")
        with np.load(path) as d:
            poses, levels = d["poses"], d["levels"]
        if poses.ndim != 3 or poses.shape[-1] != 7 or not np.isfinite(poses).all():
            raise ValueError("invalid trajectory poses")
        self.manifest = m
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.device = torch.device(device)
        self.poses = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        self.poses[..., 3:] = quat_normalize(self.poses[..., 3:])
        self.levels = torch.as_tensor(levels, dtype=torch.long, device=self.device)
        self.count, self.length = poses.shape[:2]


class TrajectorySchedule:
    """Vectorised Ta/Tb trajectory commands for a batch of environments."""

    def __init__(self, library, cfg, follow_frame, num_envs, device, generator=None):
        if cfg.segment_steps + 1 > library.length:
            raise ValueError("trajectory.segment_steps exceeds the library trajectory length")
        self.lib, self.cfg, self.frame = library, cfg, follow_frame
        self.device = torch.device(device)
        self.generator = generator
        n = num_envs
        z = lambda *shape, dtype=torch.long: torch.zeros(*shape, dtype=dtype, device=self.device)
        self.active = z(n, dtype=torch.bool)
        self.traj, self.index, self.age = z(n), z(n), z(n)
        self.anchor_pos = z(n, 3, dtype=torch.float32)
        self.anchor_quat = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(n, 1)
        self.poses = library.poses.to(self.device)
        self.levels = library.levels.to(self.device)

    def _rand(self, *shape):
        g = self.generator
        return torch.rand(*shape, device=self.device if g is None else g.device, generator=g).to(self.device)

    def _randint(self, high, shape):
        g = self.generator
        return torch.randint(high, shape, device=self.device if g is None else g.device, generator=g).to(self.device)

    def _pool_choice(self, n, level):
        """Trajectory ids: current level, or with some probability any lower level."""
        available = self.levels <= level
        if not available.any():
            available = self.levels == self.levels.min()
        current = (self.levels == min(level, int(self.levels.max()))) & available
        pool_all, pool_cur = available.nonzero().squeeze(-1), current.nonzero().squeeze(-1)
        lower = self._rand(n) < self.cfg.lower_level_fraction
        ids = pool_cur[self._randint(len(pool_cur), (n,))]
        alt = pool_all[self._randint(len(pool_all), (n,))]
        return torch.where(lower, alt, ids)

    def _body_pose(self, state, world):
        fp, fq = following_frame(state, self.frame)
        local = quat_apply_inverse(fq, world[..., :3] - fp)
        return local, quat_normalize(quat_mul(quat_conjugate(fq), world[..., 3:7]))

    def _sample_segments(self, ids, state, tcp_pose, level):
        k = max(1, int(self.cfg.continuity_candidates))
        n = len(ids)
        traj = self._pool_choice(n * k, level).view(n, k)
        start = self._randint(self.lib.length - self.cfg.segment_steps, (n, k))
        candidate = self.poses[traj, start]
        local, rel = self._body_pose(state[ids], tcp_pose[ids])
        distance = (candidate[..., :3] - local[:, None]).norm(dim=-1)
        distance = distance + .1 * quat_angle(quat_error(candidate[..., 3:], rel[:, None].expand_as(candidate[..., 3:])))
        best = distance.argmin(-1)
        rows = torch.arange(n, device=self.device)
        self.traj[ids] = traj[rows, best]
        self.index[ids] = start[rows, best]
        self.age[ids] = 0

    def _set_anchor(self, ids, state):
        fp, fq = following_frame(state[ids], self.frame)
        self.anchor_pos[ids] = fp
        self.anchor_quat[ids] = fq

    def _write(self, ids, state, goals):
        if not len(ids):
            return goals
        body = self.poses[self.traj[ids], self.index[ids]]
        fp, fq = following_frame(state[ids], self.frame)
        if goals.shape[-1] >= 11:
            moving = goals[ids, 10:11] > .5
            fp = torch.where(moving, fp, self.anchor_pos[ids])
            fq = torch.where(moving, fq, self.anchor_quat[ids])
        else:
            fp, fq = self.anchor_pos[ids], self.anchor_quat[ids]
        goals = goals.clone()
        goals[ids, :3] = fp + quat_apply(fq, body[:, :3])
        goals[ids, 3:7] = quat_normalize(quat_mul(fq, body[:, 3:]))
        return goals

    def start(self, ids, state, goals, level, tcp_pose):
        """New episode for ``ids``: choose followers, a segment and write targets."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1)
        self.active[ids] = self._rand(len(ids)) < self.cfg.fraction
        follow = ids[self.active[ids]]
        if len(follow):
            self._sample_segments(follow, state, tcp_pose, level)
            self._set_anchor(follow, state)
            goals = self._write(follow, state, goals)
        return goals

    def refresh(self, ids, state, goals):
        """Command mode/velocity changed: re-anchor and rewrite the target."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1)
        follow = ids[self.active[ids]]
        if len(follow):
            self._set_anchor(follow, state)
            goals = self._write(follow, state, goals)
        return goals

    def step(self, state, goals, level, tcp_pose, alive=None):
        """Advance one control step (after the environment step and goal transport)."""
        active = self.active if alive is None else self.active & alive
        self.age += active.long()
        self.index = torch.where(active, (self.index + 1).clamp_max(self.lib.length - 1), self.index)
        new_segment = (active & (self.age >= self.cfg.segment_steps)).nonzero().squeeze(-1)
        if len(new_segment):
            self._sample_segments(new_segment, state, tcp_pose, level)
            self._set_anchor(new_segment, state)
        update = (active & (self.age % self.cfg.update_steps == 0)).nonzero().squeeze(-1)
        return self._write(update, state, goals)

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in ("active", "traj", "index", "age", "anchor_pos", "anchor_quat")}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v.to(self.device))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--per-level", type=int, default=256)
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--duration-s", type=float, default=16.)
    p.add_argument("--seed", type=int, default=990000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--min-segment-s", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=.05, help="joint-limit margin (rad)")
    p.add_argument("--clearance", type=float, default=.01, help="planning collision clearance (m)")
    a = p.parse_args()
    from .evaluate import config_from_dict
    raw = json.loads(Path(a.config).read_text())
    cfg = config_from_dict(raw)
    cfg.validate()
    if any(not 0 <= x <= 3 for x in a.levels):
        p.error("levels must be in 0..3")
    generate(cfg, raw, a.output, per_level=a.per_level, levels=sorted(set(a.levels)), seed=a.seed,
             workers=a.workers, duration_s=a.duration_s, min_segment_s=a.min_segment_s,
             margin=a.margin, clearance=a.clearance)


if __name__ == "__main__":
    main()
