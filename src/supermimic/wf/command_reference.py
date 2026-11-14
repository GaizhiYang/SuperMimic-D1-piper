"""Physically screened references for the velocity-command + TCP-pose task.

Each clip pairs ONE fixed 11-D task command with real actuator-driven MuJoCo
transitions:

    goal = [x, y, z, qw, qx, qy, qz, vx, vy, wz, mode]

* ``mode=1`` (driving manipulation): the base tracks the body-frame velocity
  command (vx, 0, wz) while the TCP reaches and holds a pose that is fixed in
  the following frame (``reward.goal_follow_frame``), i.e. it moves with the
  base exactly as ``WholeBodyTask.transport_goal`` moves it during training.
* ``mode=0`` (stationary reach): zero base command, world-fixed TCP goal.

The offline teacher (used ONLY here, never at deployment) is:

* wheels: geometric differential-drive mapping (from the wheel joint axes and
  positions of the settled robot) plus a PI loop on the measured body velocity;
* legs: hold the nominal standing targets;
* arm: a collision-checked joint corridor (``ArmPathPlanner``) followed by
  damped-Jacobian TCP correction toward the transported goal.

A request is executed twice: nominal (friction 1.0) and an independent replay
with joint noise and friction in [0.85, 1.15]. Only the replay is stored, and
only if BOTH pass ``screen_reference`` (final 50-step hold of TCP pose, TCP
velocity and base-velocity tolerances, no failure, wheel contact and slip).
Rejected requests are logged and never become training data.

Usage (CPU, parallel):
    python -m supermimic.wf.command_reference --config configs/examples/wf_cmd_gpu4096.json \
        --output outputs/wf_cmd_refs --count 512 --workers 32
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

from .agent_references import ArmPathPlanner
from .evaluate import config_from_dict, physical_failure
from .goal_reference import rotation_error
from .reference import screen_reference
from .simulator import BatchedSimulator
from .task import WholeBodyTask

NONWHEEL = np.r_[0:12, 16:22]
ARM_SPREAD = np.array([.60, .45, .50, .60, .50, .70])
LEVEL_AMPLITUDE = (.15, .35, .60, 1.0)


def wheel_kinematics(sim):
    """Body-x speed per wheel rate and lateral wheel offsets, from geometry.

    A wheel spinning at rate w about unit axis a, rolling without slip at the
    contact point p_c = center - R z, moves its center at w R (a x z). For a
    planar body twist (vx, wz) the wheel at lateral offset y must move at
    vx - wz y, hence w_k = (vx - wz y_k) / f_k with f_k = R (a_k x z) . x.
    """
    s, m, d = sim.spec, sim.model, sim.data[0]
    rotation = d.xmat[s.base_body].reshape(3, 3)
    forward, lateral = np.zeros(4), np.zeros(4)
    for k in range(4):
        joint = m.joint(s.joint_names[12 + k]).id
        axis = rotation.T @ d.xaxis[joint]
        forward[k] = s.wheel_radius * np.cross(axis, [0., 0., 1.])[0]
        lateral[k] = (rotation.T @ (d.xpos[s.wheel_body_ids[k]] - d.xpos[s.base_body]))[1]
    if np.any(np.abs(forward) < .5 * s.wheel_radius):
        raise ValueError("wheel axes are not horizontal in the settled pose; cannot map base velocity")
    return forward, lateral


def pitch_roll(state):
    w, x, y, z = state[3:7] / np.linalg.norm(state[3:7])
    return np.arcsin(np.clip(2*(w*y-z*x), -1, 1)), np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))


FRONT = np.array([1., 1., -1., -1.])
LEFT = np.array([1., -1., 1., -1.])


def leg_posture_targets(spec, delta):
    """Nominal leg targets with per-leg crouch delta (thigh +d, calf -2d)."""
    target = spec.q0[:12].copy()
    target[np.arange(4)*3+1] += delta
    target[np.arange(4)*3+2] -= 2*delta
    return target


def posture_sensitivity(sim, settle_steps, probe=.04, steps=100):
    """Identify d(pitch)/d(front crouch) and d(roll)/d(left crouch) in physics.

    Joint sign conventions differ between assets (mirrored or not), so the
    offline stabiliser measures them on the actual model instead of assuming.
    """
    result = []
    for pattern, index in ((np.zeros(4), None), (probe*FRONT, 0), (probe*LEFT, 1)):
        sim.reset(); sim.settle(steps=settle_steps)
        action = sim.initial_action.copy()
        action[0, :12] = (leg_posture_targets(sim.spec, pattern) - sim.spec.q0[:12]) / sim.spec.action_scale[:12]
        for _ in range(steps):
            state = sim.step(action)[0]
        result.append(pitch_roll(state))
    (p0, r0), (p1, _), (_, r2) = result
    gains = np.array([(p1 - p0) / probe, (r2 - r0) / probe])
    if np.any(np.abs(gains) < .2):
        raise ValueError(f"leg posture has no measurable pitch/roll authority: {gains}")
    return gains


def body_velocity(state):
    w, x, y, z = state[3:7] / np.linalg.norm(state[3:7])
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    v = r.T @ state[29:32]
    return v[0], state[34]


class CommandTeacher:
    """Offline Markov-in-(state, previous, goal, PI memory) teacher."""

    def __init__(self, sim, task, goal, path, *, kinematics, movement_steps=250, ramp_steps=50,
                 arm_lower, arm_upper, gains=(1.0, 2.0, 1.0, 2.0), max_turn=.2, posture=None):
        self.sim, self.task = sim, task
        self.goal0 = torch.as_tensor(goal, dtype=torch.float32)[None]
        self.path = np.asarray(path)
        lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=-1)
        knots = np.r_[0, np.cumsum(lengths)]
        self.knots = knots / max(knots[-1], 1e-8)
        self.forward, self.lateral = kinematics
        self.movement_steps, self.ramp_steps = movement_steps, ramp_steps
        self.arm_lower, self.arm_upper = arm_lower, arm_upper
        self.kp_v, self.ki_v, self.kp_w, self.ki_w = gains
        self.max_turn = float(max_turn)
        self.iv = self.iw = 0.
        self.posture = posture  # (dpitch/dfront, droll/dleft) or None
        self.ip = self.ir = 0.
        self.reference_attitude = None
        self.initial = None

    def current_goal(self, state):
        g = self.task.transport_goal(self.initial, torch.as_tensor(state, dtype=torch.float32)[None], self.goal0)
        return g[0].numpy()

    def arm_correction(self, goal, previous):
        s, m, d = self.sim.spec, self.sim.model, self.sim.data[0]
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, d.site_xmat[s.tcp_site])
        error = goal[:3] - d.site_xpos[s.tcp_site]
        angle = rotation_error(goal[3:7], quat)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, s.tcp_site)
        cols = s.dof_indices[16:]
        jac = np.vstack((jp[:, cols], .3 * jr[:, cols]))
        desired = np.r_[error, .3 * angle]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .025**2 * np.eye(6), desired)
        q = (s.q0 + s.action_scale * previous)[16:]
        step = np.clip(.08 * dq - .03 * d.qvel[cols], -.006, .006)
        return np.clip(q + step, self.arm_lower, self.arm_upper)

    def command(self, step, state, previous):
        s = self.sim.spec
        goal = self.current_goal(state)
        vx_cmd, wz_cmd, mode = goal[7], goal[9], goal[10]
        out = previous.copy()
        # Legs: nominal standing targets plus a pitch/roll PI stabiliser that
        # holds the settled attitude (wheel reaction torques and arm motion
        # otherwise make the body pitch over during long drives).
        delta = np.zeros(4)
        if self.posture is not None:
            pitch, roll = pitch_roll(state)
            p0, r0 = self.reference_attitude
            dt = self.sim.dt
            ep, er = pitch - p0, roll - r0
            self.ip = np.clip(self.ip + ep * dt, -.3, .3)
            self.ir = np.clip(self.ir + er * dt, -.3, .3)
            dp = -(.8 * ep + 1.5 * self.ip + .05 * state[33]) / self.posture[0]
            dr = -(.8 * er + 1.5 * self.ir + .05 * state[32]) / self.posture[1]
            delta = np.clip(dp * FRONT + dr * LEFT, -.12, .12)
        legs = (leg_posture_targets(s, delta) - s.q0[:12]) / s.action_scale[:12]
        out[:12] = previous[:12] + np.clip(legs - previous[:12], -.004, .004)
        # Wheels: ramped feed-forward + PI on measured body velocity.
        ramp = min(1., (step + 1) / self.ramp_steps)
        v_ref, w_ref = ramp * vx_cmd * mode, ramp * wz_cmd * mode
        vx, wz = body_velocity(state)
        dt = self.sim.dt
        self.iv = np.clip(self.iv + (v_ref - vx) * dt, -.3, .3)
        self.iw = np.clip(self.iw + (w_ref - wz) * dt, -.5, .5)
        v = v_ref + self.kp_v * (v_ref - vx) + self.ki_v * self.iv
        w = w_ref + self.kp_w * (w_ref - wz) + self.ki_w * self.iw
        # Skid steering scrubs the wheels sideways. A bounded left/right
        # differential avoids pitching the body over when yaw saturates.
        turn = np.clip(w * self.lateral / self.forward / s.action_scale[12:16],
                       -self.max_turn, self.max_turn)
        wheel = v / self.forward / s.action_scale[12:16] - turn
        out[12:16] = previous[12:16] + np.clip(wheel - previous[12:16], -.01, .01)
        # Arm: collision-checked corridor, then Cartesian correction.
        if step < self.movement_steps:
            u = (step + 1) / self.movement_steps
            u = u**3 * (10 - 15*u + 6*u*u)
            seg = min(len(self.path) - 2, max(0, int(np.searchsorted(self.knots, u) - 1)))
            local = (u - self.knots[seg]) / max(1e-8, self.knots[seg+1] - self.knots[seg])
            target = self.path[seg] + np.clip(local, 0, 1) * (self.path[seg+1] - self.path[seg])
            desired = (target - s.q0[16:]) / s.action_scale[16:]
            out[16:] = previous[16:] + np.clip(desired - previous[16:], -.008, .008)
        else:
            q = self.arm_correction(goal, previous)
            out[16:] = (q - s.q0[16:]) / s.action_scale[16:]
        return np.clip(out, -1., 1.)


@torch.no_grad()
def execute(sim, task, goal, path, *, kinematics, steps, settle_steps, arm_lower, arm_upper,
            joint_noise=0., seed=0, movement_steps=250, max_turn=.2, posture=None):
    sim.reset()
    if joint_noise:
        state = sim.read_state()
        state[:, 7 + NONWHEEL] += np.random.default_rng(seed).normal(0, joint_noise, (1, 18))
        state[:, 7 + NONWHEEL] = np.clip(state[:, 7 + NONWHEEL], sim.spec.low[NONWHEEL], sim.spec.high[NONWHEEL])
        sim.write_state(state)
    sim.settle(steps=settle_steps)
    snapshot = sim.snapshot()
    initial_action = sim.applied_action[0].copy()
    previous = initial_action.copy()
    teacher = CommandTeacher(sim, task, goal, path, kinematics=kinematics, movement_steps=movement_steps,
                             arm_lower=arm_lower, arm_upper=arm_upper, max_turn=max_turn, posture=posture)
    first = sim.read_state()[0].copy()
    teacher.initial = torch.as_tensor(first, dtype=torch.float32)[None]
    teacher.reference_attitude = pitch_roll(first)
    states = [first]
    data = {k: [] for k in ("actions", "safe", "contacts", "slip", "torque_peak", "failed")}
    for step in range(steps):
        command = teacher.command(step, states[-1], previous)
        sn = sim.step(command[None])[0].copy()
        previous = sim.applied_action[0].copy()
        metrics = sim.metrics()
        states.append(sn)
        data["actions"].append(previous.copy()); data["safe"].append(bool(metrics["safe"][0]))
        data["contacts"].append(metrics["wheel_contact"][0]); data["slip"].append(metrics["slip"][0])
        data["torque_peak"].append(metrics["torque_peak"][0])
        failed = bool(physical_failure(task, torch.as_tensor(sn[None], dtype=torch.float32), metrics)[0])
        data["failed"].append(failed)
        if failed:
            break
    return dict(states=np.asarray(states), **{k: np.asarray(v) for k, v in data.items()},
                goal=np.asarray(goal, dtype=np.float64), xi=sim.xi[0].copy(), initial_action=initial_action,
                initial_mj_state=snapshot["integration_state"][0], initial_state_flag=snapshot["state_flag"])


# ----------------------------------------------------------------- workers
_W: dict = {}


def _init_worker(raw_config, options):
    torch.set_num_threads(1)
    cfg = config_from_dict(raw_config)
    c = cfg.sim
    sim = BatchedSimulator(num_envs=1, seed=0, friction=1., dt=c.dt, timestep=c.timestep,
        asset_dir=cfg.asset_dir, leg_action_scale=c.leg_action_scale, arm_action_scale=c.arm_action_scale,
        wheel_action_scale=c.wheel_action_scale, arm_initial_position=c.arm_initial_position)
    task = WholeBodyTask(sim.model, sim.spec, reward_cfg=cfg.reward)
    s = sim.spec
    lower, upper = s.low.copy(), s.high.copy()
    if cfg.policy.action_lower:
        lower = np.maximum(lower, s.q0 + s.action_scale * np.asarray(cfg.policy.action_lower))
        upper = np.minimum(upper, s.q0 + s.action_scale * np.asarray(cfg.policy.action_upper))
    lower = np.maximum(lower, s.q0 - s.action_scale)[16:]
    upper = np.minimum(upper, s.q0 + s.action_scale)[16:]
    sim.reset(); sim.settle(steps=c.reset_settle_steps)
    kinematics = wheel_kinematics(sim)
    posture = posture_sensitivity(sim, c.reset_settle_steps) if options.get("stabilise_posture", True) else None
    _W.update(cfg=cfg, sim=sim, task=task, lower=lower, upper=upper, kinematics=kinematics,
              posture=posture, options=options)


def _attempt(seed):
    cfg, sim, task, o = _W["cfg"], _W["sim"], _W["task"], _W["options"]
    lower, upper = _W["lower"], _W["upper"]
    rng = np.random.default_rng(seed)
    c = cfg.sim
    stationary = rng.random() < o["stationary_fraction"]
    level = int(rng.integers(o["min_level"], o["max_level"] + 1))
    if stationary:
        vx = wz = 0.; mode = 0.
    else:
        mode = 1.
        vx = rng.uniform(-1, 1) * o["vx_max"]
        wz = rng.uniform(-1, 1) * o["wz_max"]
        if rng.random() < o["straight_fraction"]:
            wz = 0.
        if rng.random() < o["zero_command_fraction"]:
            vx = wz = 0.
    sim.friction_range = (1., 1.)
    sim.reset(); sim.settle(steps=c.reset_settle_steps)
    initial = sim.read_state()[0].copy()
    margin = .05
    planner = ArmPathPlanner(sim, initial, lower + margin, upper - margin, clearance=o["clearance"])
    start_arm = initial[23:29]
    target = None
    for _ in range(300):
        candidate = np.clip(start_arm + rng.uniform(-1, 1, 6) * ARM_SPREAD * LEVEL_AMPLITUDE[level],
                            lower + margin, upper - margin)
        if not planner.valid(candidate):
            continue
        d = planner.data
        if d.site_xpos[sim.spec.tcp_site, 2] < cfg.reward.minimum_goal_height + .05:
            continue
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, d.site_xmat[sim.spec.tcp_site])
        pose = np.r_[d.site_xpos[sim.spec.tcp_site].copy(), quat]
        target = candidate
        break
    request = dict(seed=int(seed), vx=float(vx), wz=float(wz), mode=int(mode), level=level)
    if target is None:
        return dict(request, accepted=False, reason="no collision-free arm target")
    path = planner.plan(target, rng)
    if path is None:
        return dict(request, accepted=False, reason="no collision-free joint corridor")
    goal = np.r_[pose, vx, 0., wz, mode]
    common = dict(kinematics=_W["kinematics"], steps=c.episode_steps, settle_steps=c.reset_settle_steps,
                  arm_lower=lower, arm_upper=upper, movement_steps=o["movement_steps"],
                  max_turn=o["max_turn"], posture=_W["posture"])
    nominal = execute(sim, task, goal, path, **common)
    check = screen_reference(nominal, task, goal, max_mean_slip=o["max_slip"])
    if not check["accepted"] or len(nominal["actions"]) != c.episode_steps:
        return dict(request, accepted=False, reason="nominal physical screen failed", screen=check)
    sim.friction_range = (.85, 1.15)
    clip = execute(sim, task, goal, path, joint_noise=.005, seed=seed + 500_000, **common)
    check = screen_reference(clip, task, goal, max_mean_slip=o["max_slip"])
    if not check["accepted"] or len(clip["actions"]) != c.episode_steps:
        return dict(request, accepted=False, reason="independent replay screen failed", screen=check)
    # Requested TCP offset in the base frame, used for curriculum gating.
    tcp0 = task.tcp_pose(torch.as_tensor(clip["states"][:1], dtype=torch.float32))[0, :3].numpy()
    reach = float(np.linalg.norm(goal[:3] - tcp0))
    radii = cfg.reward.goal_translation_radii
    goal_level = next((i for i, r in enumerate(radii) if reach <= r), len(radii) - 1)
    return dict(request, accepted=True, screen=check, clip=clip, path=[p.tolist() for p in path],
                reach_m=reach, goal_level=int(goal_level))


def generate(cfg, raw_config, output, *, count, seed, workers, max_attempts=None, **options):
    from .train import model_manifest
    if not cfg.reward.base_command_conditioning:
        raise ValueError("command references require reward.base_command_conditioning=true")
    output = Path(output)
    library = output / "library"
    if (library / "manifest.json").exists():
        raise FileExistsError("reference library already exists; select a new output directory")
    library.mkdir(parents=True, exist_ok=True)
    probe = BatchedSimulator(num_envs=1, friction=1., dt=cfg.sim.dt, timestep=cfg.sim.timestep, asset_dir=cfg.asset_dir,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale, arm_initial_position=cfg.sim.arm_initial_position)
    provenance = model_manifest(probe.spec)
    probe.close()
    manifest = dict(format="wf_physical_references_v1", complete=False, provenance=provenance, dt=cfg.sim.dt,
        seed=seed, level=options["max_level"], requested_clips=count, config=asdict(cfg), clips=[], rejected=[],
        goal_layout="world TCP xyz, quaternion wxyz, body vx/vy/wz, mode (1 = drive with goal following the base)",
        goal_follow_frame=cfg.reward.goal_follow_frame, options=options,
        method="offline geometric wheel PI + collision-checked arm corridor + Jacobian TCP correction; "
               "nominal and independent noisy-friction replay both physically screened",
        generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    attempts = max_attempts or count * 6
    started = time.monotonic()
    accepted = 0
    ctx = mp.get_context("fork" if os.name == "posix" else "spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(raw_config, options)) as pool, \
            (output / "attempts.jsonl").open("w") as log:
        for result in pool.imap(_attempt, range(seed, seed + attempts), chunksize=1):
            clip = result.pop("clip", None)
            log.write(json.dumps({k: v for k, v in result.items() if k != "path"}) + "\n"); log.flush()
            if not result["accepted"]:
                manifest["rejected"].append({k: v for k, v in result.items() if k != "path"})
                continue
            i = len(manifest["clips"])
            name = f"clip_{i:05d}"
            np.savez_compressed(library / (name + ".npz"), **clip)
            (library / (name + "_program.json")).write_text(json.dumps(dict(
                world_goal=clip["goal"].tolist(), joint_corridor=result["path"],
                command=dict(vx=result["vx"], vy=0., wz=result["wz"], mode=result["mode"]),
                teacher="offline only; not available to the deployed Actor"), indent=2) + "\n")
            manifest["clips"].append(dict(file=name + ".npz", keyframes=name + "_program.json",
                family="command_drive_reach" if result["mode"] else "stationary_reach",
                seed=result["seed"], level=result["goal_level"], split="validation" if i % 5 == 4 else "train",
                command=[result["vx"], 0., result["wz"], result["mode"]], reach_m=result["reach_m"],
                screen=result["screen"], sha256=hashlib.sha256((library / (name + ".npz")).read_bytes()).hexdigest()))
            accepted = i + 1
            if accepted % 10 == 0 or accepted == count:
                print(json.dumps(dict(accepted=accepted, attempts=result["seed"] - seed + 1,
                                      elapsed_s=round(time.monotonic() - started, 1))), flush=True)
                (library / "manifest.partial.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if accepted >= count:
                pool.terminate()
                break
    manifest["complete"] = accepted == count
    manifest["acceptance_rate"] = accepted / max(1, accepted + len(manifest["rejected"]))
    (library / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    summary = dict(clips=accepted, requested=count, rejected=len(manifest["rejected"]),
                   command_clips=sum(c["family"] == "command_drive_reach" for c in manifest["clips"]),
                   stationary_clips=sum(c["family"] == "stationary_reach" for c in manifest["clips"]),
                   library=str(library), elapsed_s=round(time.monotonic() - started, 1))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not manifest["complete"]:
        raise RuntimeError(f"only {accepted}/{count} references passed physical screening; "
                           "see attempts.jsonl and raise --max-attempts or relax the command ranges")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="training config (sim/reward/policy must match training)")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--seed", type=int, default=880000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-attempts", type=int)
    p.add_argument("--vx-max", type=float, help="default: reward.base_command_ranges[0]")
    p.add_argument("--wz-max", type=float, help="default: reward.base_command_ranges[2]")
    p.add_argument("--stationary-fraction", type=float, default=.2)
    p.add_argument("--straight-fraction", type=float, default=.25)
    p.add_argument("--zero-command-fraction", type=float, default=.1)
    p.add_argument("--min-level", type=int, default=0, choices=range(4))
    p.add_argument("--max-level", type=int, default=2, choices=range(4))
    p.add_argument("--movement-steps", type=int, default=250)
    p.add_argument("--clearance", type=float, default=.01)
    p.add_argument("--max-slip", type=float, default=.10)
    p.add_argument("--max-turn", type=float, default=.2,
                   help="bound on the normalized left/right wheel differential of the offline teacher")
    a = p.parse_args()
    if a.count < 5:
        p.error("at least five clips are needed for separate training/validation")
    if a.min_level > a.max_level:
        p.error("--min-level must not exceed --max-level")
    raw = json.loads(Path(a.config).read_text())
    cfg = config_from_dict(raw)
    cfg.validate()
    ranges = cfg.reward.base_command_ranges
    options = dict(vx_max=ranges[0] if a.vx_max is None else a.vx_max,
                   wz_max=ranges[2] if a.wz_max is None else a.wz_max,
                   stationary_fraction=a.stationary_fraction, straight_fraction=a.straight_fraction,
                   zero_command_fraction=a.zero_command_fraction, min_level=a.min_level, max_level=a.max_level,
                   movement_steps=a.movement_steps, clearance=a.clearance, max_slip=a.max_slip,
                   max_turn=a.max_turn)
    manifest = generate(cfg, raw, a.output, count=a.count, seed=a.seed, workers=a.workers,
                        max_attempts=a.max_attempts, **options)
    print(json.dumps(dict(clips=len(manifest["clips"]), rejected=len(manifest["rejected"]),
                          manifest=str(Path(a.output) / "library/manifest.json"))))


if __name__ == "__main__":
    main()
