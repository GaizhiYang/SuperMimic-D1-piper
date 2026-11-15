"""DAgger relabelling: config, label aggregation and the real teacher loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.wf.dagger import load_queries, save_queries
from supermimic.wf.demonstrations import Demonstrations

ROOT = Path(__file__).resolve().parents[1]


def raw_smoke():
    return json.loads((ROOT / "configs/examples/wf_cmd_smoke.json").read_text())


@pytest.mark.parametrize("patch", [dict(dagger_rounds=3, dagger_interval=10),  # 30 >= 20 updates
                                   dict(query_file="q.npz"), dict(dagger_beta=1.5), dict(query_fraction=1.)])
def test_dagger_config_validation(patch):
    from supermimic.wf.evaluate import config_from_dict
    raw = raw_smoke()
    raw["demonstrations"].update(patch)
    with pytest.raises(ValueError):
        config_from_dict(raw).validate()


def test_configs_enable_dagger():
    from supermimic.wf.evaluate import config_from_dict
    for name in ("wf_cmd_gpu4096.json", "wf_cmd_smoke.json"):
        cfg = config_from_dict(json.loads((ROOT / "configs/examples" / name).read_text()))
        cfg.validate()
        assert cfg.demonstrations.dagger_rounds > 0
        # Rate-limited teacher labels make "copy the previous command" a near-perfect
        # BC fit that random-walks in closed loop; the actor must not see it.
        assert cfg.policy.ignore_previous_action and cfg.policy.action_mode == "absolute"


def test_labels_join_bc_with_clip_scale(tmp_path):
    demos = Demonstrations.__new__(Demonstrations)
    demos.device = torch.device("cpu"); demos.query_inputs = None; demos.query_scale = None
    task = SimpleNamespace(actor_input=lambda s, p, g: torch.cat((s, p, g), -1), command_goal=lambda g: g)
    n = demos.add_queries(task, torch.zeros(3, 2), torch.zeros(3, 22), torch.zeros(3, 11), torch.full((3, 22), .5))
    n = demos.add_queries(task, torch.ones(2, 2), torch.zeros(2, 22), torch.zeros(2, 11), torch.zeros(2, 22))
    assert n == 5 and demos.query_scale is None and len(demos.query_training) == 5
    with pytest.raises(ValueError):
        demos.add_queries(task, torch.zeros(1, 2), torch.zeros(1, 22), torch.zeros(1, 11), torch.full((1, 22), 2.))


def test_query_file_round_trip(tmp_path):
    rounds = [{k: torch.randn(4, d) for k, d in (("state", 57), ("previous", 22), ("goal", 11), ("action", 22))}
              for _ in range(2)]
    path = save_queries(tmp_path / "q.npz", rounds)
    loaded = load_queries(path)
    assert torch.equal(loaded["state"], torch.cat([r["state"] for r in rounds]))
    with np.load(path) as d:
        assert d["round"].tolist() == [0] * 4 + [1] * 4


@pytest.mark.private_assets
def test_teacher_relabels_learner_states(tmp_path):
    """With beta=1 the rollout is the teacher itself: labels equal executed clips."""
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf import command_reference as cr
    from supermimic.wf.dagger import TeacherOracle, collect_corrections
    from supermimic.wf.evaluate import config_from_dict
    from supermimic.wf.networks import Actor
    from supermimic.wf.simulator import BatchedSimulator
    from supermimic.wf.task import WholeBodyTask
    from supermimic.wf.train import model_manifest
    raw = raw_smoke()
    cfg = config_from_dict(raw)
    ranges = cfg.reward.base_command_ranges
    cr.generate(cfg, raw, tmp_path / "refs", count=5, seed=880000, workers=2, vx_max=ranges[0], wz_max=ranges[2],
                stationary_fraction=.2, straight_fraction=.25, zero_command_fraction=.1, min_level=0, max_level=1,
                movement_steps=250, clearance=.01, max_slip=.1, max_turn=.2)
    probe = BatchedSimulator(num_envs=1, dt=cfg.sim.dt, timestep=cfg.sim.timestep,
                             leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
                             wheel_action_scale=cfg.sim.wheel_action_scale)
    task = WholeBodyTask(probe.model, probe.spec, "cpu", cfg.reward)
    demos = Demonstrations(tmp_path / "refs/library", task, model_manifest(probe.spec)["physics_hash"])
    oracle = TeacherOracle(cfg, probe.model, probe.spec, demos)
    actor = Actor(cfg.policy)
    data, stats = collect_corrections(actor, task, oracle, demos, demos.training[:2], beta=1., seed=3,
                                      steps=40, friction_range=(1., 1.))
    assert stats["labels"] == 80 and stats["learner_step_fraction"] == 0 and stats["failure_rate"] == 0
    # Same teacher, same start, friction 1: labels track the stored clip commands
    # (clips are a noisy-friction replay, so only approximately).
    stored = demos.actions[demos.training[:2], :40].reshape(-1, 22)
    order = torch.arange(80).view(40, 2).T.reshape(-1)  # data is step-major
    assert (data["action"][order] - stored).abs().mean() < .02
    probe.close()
