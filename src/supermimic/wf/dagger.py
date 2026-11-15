"""DAgger relabelling with the offline command teacher.

Pure behaviour cloning of the reference clips is accurate on the teacher's own
states (action error ~1e-3) but drifts in closed loop: the learner never saw
how the teacher recovers from its own small errors. DAgger rolls the learner
(mixed with the teacher at rate ``beta``) out in CPU MuJoCo from training clip
start states, and at every visited state asks the same offline teacher that
produced the clips (``command_reference.CommandTeacher``) for its command.

The teacher is Markov in (state, previous executed command, goal, its own PI
integrators, step index), so one teacher instance per world runs alongside the
rollout and is updated from the states actually visited. Labels are the
teacher's executable [-1, 1] commands, i.e. the same quantity as the clips'
BC targets. Only training clips are used; validation clips stay held out for
physical BC selection. The teacher remains offline-only: it never runs at
deployment and never enters world-model replay.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .command_reference import CommandTeacher, pitch_roll, teacher_setup
from .evaluate import physical_failure
from .simulator import BatchedSimulator


class TeacherOracle:
    """Offline teacher configuration for one reference library."""

    def __init__(self, cfg, model, spec, demonstrations):
        manifest = demonstrations.manifest
        options = manifest.get("options") or {}
        if not str(manifest.get("method", "")).startswith("offline geometric wheel PI"):
            raise ValueError("DAgger relabelling requires a command_reference library (CommandTeacher clips)")
        self.cfg, self.model, self.spec = cfg, model, spec
        probe = BatchedSimulator(model, spec, num_envs=1, seed=0, dt=cfg.sim.dt, friction=1.,
                                 arm_initial_position=cfg.sim.arm_initial_position)
        self.lower, self.upper, self.kinematics, self.posture = teacher_setup(
            cfg, probe, options.get("stabilise_posture", True))
        probe.close()
        self.movement_steps = int(options.get("movement_steps", 250))
        self.max_turn = float(options.get("max_turn", .2))
        directory = demonstrations.directory
        self.paths = []
        for clip in manifest["clips"]:
            program = json.loads((directory / clip["keyframes"]).read_text())
            self.paths.append(np.asarray(program["joint_corridor"], dtype=np.float64))

    def make_teacher(self, sim, world, task, goal, initial_state, clip):
        """A fresh teacher reading MuJoCo data of world ``world`` of ``sim``."""
        view = SimpleNamespace(data=[sim.data[world]], spec=sim.spec, model=sim.model, dt=sim.dt)
        teacher = CommandTeacher(view, task, goal, self.paths[clip], kinematics=self.kinematics,
                                 movement_steps=self.movement_steps, arm_lower=self.lower,
                                 arm_upper=self.upper, max_turn=self.max_turn, posture=self.posture)
        teacher.initial = torch.as_tensor(initial_state, dtype=torch.float32)[None]
        teacher.reference_attitude = pitch_roll(initial_state)
        return teacher


@torch.no_grad()
def collect_corrections(actor, task, oracle, demonstrations, rows, *, beta, seed, steps=None,
                        friction_range=(.85, 1.15)):
    """Roll the beta-mixed learner from clip starts; return teacher labels.

    ``actor`` and ``task`` must be on the CPU (MuJoCo state lives there).
    Returns CPU tensors ``state, previous, goal, action`` (one row per visited
    alive state) plus rollout statistics of this round.
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    cfg = oracle.cfg
    rows = torch.as_tensor(rows, dtype=torch.long).cpu()
    n = len(rows)
    steps = int(steps or cfg.sim.episode_steps)
    rng = np.random.default_rng(seed)
    sim = BatchedSimulator(oracle.model, oracle.spec, num_envs=n, seed=seed, dt=cfg.sim.dt, friction=1.,
                           arm_initial_position=cfg.sim.arm_initial_position)
    sim.reset()
    sim.set_friction(np.arange(n), rng.uniform(*friction_range, n))
    initial = demonstrations.states[rows.to(demonstrations.device), 0].cpu()
    sim.write_state(initial.numpy())
    goal0 = demonstrations.goal[rows.to(demonstrations.device)].cpu()
    previous = demonstrations.initial_action[rows.to(demonstrations.device)].cpu().clone()
    sim.ablation_action[:] = previous.numpy()
    teachers = [oracle.make_teacher(sim, i, task, goal0[i].numpy(), initial[i].numpy(), int(rows[i]))
                for i in range(n)]
    alive = torch.ones(n, dtype=torch.bool)
    out = {key: [] for key in ("state", "previous", "goal", "action")}
    learner_steps = 0
    gap = torch.zeros(3)
    gap_count = 0
    input_mode = actor.training
    actor.eval()
    try:
        for step in range(steps):
            state = torch.as_tensor(sim.read_state(), dtype=torch.float32)
            goal = task.transport_goal(initial, state, goal0)
            learner = actor(task.actor_input(state, previous, goal)).clamp(-1, 1)
            label = torch.as_tensor(np.stack([
                teachers[i].command(step, state[i].numpy().astype(np.float64), previous[i].numpy().astype(np.float64))
                if alive[i] else previous[i].numpy() for i in range(n)]), dtype=torch.float32)
            live = alive.nonzero().squeeze(-1)
            for key, value in (("state", state), ("previous", previous), ("goal", goal), ("action", label)):
                out[key].append(value[live].clone())
            error = (learner - label)[live].abs()
            gap += torch.stack((error[:, :12].mean(-1).sum(), error[:, 12:16].mean(-1).sum(), error[:, 16:].mean(-1).sum()))
            gap_count += len(live)
            use_teacher = torch.as_tensor(rng.random(n) < beta)
            learner_steps += int((~use_teacher & alive).sum())
            command = torch.where(use_teacher[:, None], label, learner)
            command[~alive] = previous[~alive]
            sn = torch.as_tensor(sim.step(command.numpy()), dtype=torch.float32)
            previous = torch.as_tensor(sim.applied_action, dtype=torch.float32).clone()
            failed = physical_failure(task, sn, sim.metrics()).cpu() & alive
            alive &= ~failed
            if not alive.any():
                break
    finally:
        actor.train(input_mode)
        sim.close()
    data = {key: torch.cat(value) for key, value in out.items()}
    stats = dict(episodes=n, beta=float(beta), labels=int(len(data["state"])),
                 learner_step_fraction=learner_steps / max(1, int(len(data["state"]))),
                 failure_rate=float((~alive).float().mean()),
                 learner_teacher_gap=dict(zip(("legs", "wheels", "arm"), (gap / max(1, gap_count)).tolist())))
    return data, stats


def save_queries(path, rounds):
    """Persist all aggregated rounds (needed again after a resume)."""
    path = Path(path)
    keys = ("state", "previous", "goal", "action")
    arrays = {key: torch.cat([r[key] for r in rounds]).numpy() for key in keys}
    arrays["round"] = np.concatenate([np.full(len(r["state"]), i, dtype=np.int32) for i, r in enumerate(rounds)])
    np.savez_compressed(path, **arrays)
    return path


def load_queries(path):
    with np.load(path, allow_pickle=False) as d:
        return {key: torch.as_tensor(d[key]) for key in ("state", "previous", "goal", "action")}
