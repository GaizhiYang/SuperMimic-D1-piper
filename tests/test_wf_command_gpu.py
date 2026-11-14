"""Velocity-command + TCP-pose pipeline: configs, asset gains, Warp parity, references."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from supermimic.wf.evaluate import config_from_dict

ROOT = Path(__file__).resolve().parents[1]


def _assets():
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")


@pytest.mark.parametrize("name", ["wf_cmd_gpu4096.json", "wf_cmd_smoke.json"])
def test_command_configs_validate(name):
    raw = json.loads((ROOT / "configs/examples" / name).read_text())
    cfg = config_from_dict(raw)
    cfg.validate()
    assert cfg.policy.base_command_conditioning and cfg.reward.base_command_conditioning
    assert cfg.sim.backend == "warp" and cfg.sim.reset_bank_batches > 0
    assert cfg.demonstrations.directory
    schedule = cfg.base_speed_curriculum
    assert schedule[-1][1] <= cfg.reward.base_command_ranges[0]


@pytest.mark.private_assets
def test_build_model_uses_asset_actuator_parameters():
    _assets()
    from supermimic.wf.robot import build_model
    root = Path(os.environ["WFARM_ASSET_DIR"])
    config = json.loads((root / "config/robot.json").read_text())
    params = config.get("actuator_parameters")
    if not params:
        pytest.skip("asset has no actuator_parameters")
    model, spec = build_model(None, arm_action_scale=None)
    np.testing.assert_allclose(spec.kp[:12], np.tile(params["leg_kp"], 4))
    np.testing.assert_allclose(spec.kd[12:16], params["wheel_kd"])
    np.testing.assert_allclose(spec.kp[16:], params["arm_kp"])
    if params.get("arm_effort_limit_nm") is not None:
        assert spec.effort[16:].max() <= params["arm_effort_limit_nm"] + 1e-9
        assert np.abs(model.actuator_ctrlrange[spec.actuator_ids[16:]]).max() <= params["arm_effort_limit_nm"] + 1e-9
    np.testing.assert_allclose(spec.wheel_radius, model.geom_size[spec.wheel_geom_ids, 0])


@pytest.mark.private_assets
def test_warp_backend_matches_cpu_mujoco():
    _assets()
    pytest.importorskip("mujoco_warp")
    from supermimic.wf.simulator import BatchedSimulator
    from supermimic.wf.simulator_warp import WarpBatchedSimulator
    kw = dict(num_envs=2, seed=0, arm_action_scale=None)
    cpu = BatchedSimulator(**kw)
    gpu = WarpBatchedSimulator(model=cpu.model, spec=cpu.spec, num_envs=2, seed=0, device="cpu")
    friction = np.array([.7, 1.1])
    cpu.set_friction(None, friction); gpu.set_friction(None, friction)
    cpu.settle(steps=20); gpu.settle(steps=20)
    action = np.clip(cpu.initial_action + np.random.default_rng(0).normal(0, .12, (2, 22)), -1, 1)
    for _ in range(10):
        cpu.step(action); gpu.step(action)
    assert np.abs(cpu.read_state() - gpu.read_state()).max() < 1e-3
    assert np.abs(cpu.applied_action - gpu.applied_action).max() < 1e-6
    for key in ("wheel_contact", "forbidden_contact", "safe"):
        assert (cpu.metrics()[key] == gpu.metrics()[key]).all(), key
    np.testing.assert_allclose(cpu.metrics()["slip"], gpu.metrics()["slip"], atol=1e-3)
    before = gpu.read_state()
    gpu.step(action, ids=[1])
    after = gpu.read_state()
    assert np.array_equal(before[0], after[0]) and not np.array_equal(before[1], after[1])
    snapshot = gpu.snapshot()
    gpu.step(action)
    gpu.restore(snapshot)
    assert np.array_equal(gpu.read_state(), after)


@pytest.mark.private_assets
def test_command_reference_teacher_produces_screened_clip():
    _assets()
    from supermimic.wf import command_reference as cr
    raw = json.loads((ROOT / "configs/examples/wf_cmd_smoke.json").read_text())
    options = dict(vx_max=.2, wz_max=0., stationary_fraction=0., straight_fraction=1., zero_command_fraction=0.,
                   min_level=0, max_level=0, movement_steps=250, clearance=.01, max_slip=.1, max_turn=.2)
    cr._init_worker(raw, options)
    forward, lateral = cr._W["kinematics"]
    radius = cr._W["sim"].spec.wheel_radius
    assert np.all(np.abs(np.abs(forward) - radius) < .1 * radius)
    assert np.all(np.sign(lateral) == np.array([1, -1, 1, -1]))
    results = [cr._attempt(seed) for seed in range(880000, 880004)]
    accepted = [r for r in results if r["accepted"]]
    assert accepted, [r.get("reason") for r in results]
    clip = accepted[0]["clip"]
    assert clip["goal"].shape == (11,) and clip["goal"][10] == 1.
    assert clip["states"].shape == (raw["sim"]["episode_steps"] + 1, 57)
    assert not clip["failed"].any() and clip["safe"][-50:].all()
