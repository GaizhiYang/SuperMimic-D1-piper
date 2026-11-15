"""Moving trajectory recording preserves the real policy evaluation rollout."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.private_assets
@pytest.mark.parametrize("presentation", [False, True])
def test_visual_recording_follows_moving_library_without_changing_physics(monkeypatch, presentation):
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf import play
    from supermimic.wf.config import PolicyConfig, TrajectoryConfig
    from supermimic.wf.evaluate import evaluate_policy
    from supermimic.wf.math import following_frame, quat_apply_inverse
    from supermimic.wf.networks import Actor
    from supermimic.wf.robot import build_model
    from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask
    from supermimic.wf.train import model_manifest

    model, spec = build_model(arm_action_scale=None)
    task = WholeBodyTask(model, spec, "cpu", WholeBodyRewardCfg(
        base_command_conditioning=True, base_command_fraction=1.,
    ))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actor = Actor(PolicyConfig(hidden=(16, 16), latent_dim=8,
                                   base_command_conditioning=True))
    # A deliberately moving local TCP command makes this test distinguish
    # library playback from a static goal merely transported with the chassis.
    poses = torch.zeros(1, 32, 7)
    poses[0, :, 0] = .45 + .001 * torch.arange(32)
    poses[0, :, 2] = .15
    poses[0, :, 3] = 1.
    library = SimpleNamespace(poses=poses, levels=torch.tensor([0]), count=1,
                              length=32, fingerprint="integration-trajectory")
    schedule = TrajectoryConfig(segment_steps=16, update_steps=2,
        continuity_candidates=1, lower_level_fraction=0., tracking_grace_steps=0)
    options = dict(episodes=1, steps=10, hold_steps=1, seed=123,
        settle_steps=20, goal_seed_device="cpu", trajectories=library,
        trajectory_cfg=schedule)

    inputs = []
    original_input = task.actor_input

    def record_input(state, previous, goals, **kwargs):
        inputs.append(tuple(x.detach().clone() for x in (state, previous, goals)))
        return original_input(state, previous, goals, **kwargs)

    monkeypatch.setattr(task, "actor_input", record_input)
    before_hash = model_manifest(spec)["physics_hash"]
    before_offscreen = model.vis.global_.offwidth, model.vis.global_.offheight
    baseline = evaluate_policy(actor, task, model, spec, **options)
    baseline_inputs = inputs[:]
    inputs.clear()
    render_updates = []

    class FakeRenderer:
        def __init__(self, render_model, *, height, width):
            self.model = render_model
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
            self.scene = SimpleNamespace(flags=np.zeros(int(play.mujoco.mjtRndFlag.mjNRNDFLAG), dtype=bool))

        def update_scene(self, data, *, camera, scene_option=None):
            render_updates.append((data, scene_option))

        def render(self):
            return self.frame.copy()

        def close(self):
            pass

    monkeypatch.setattr(play.mujoco, "Renderer", FakeRenderer)
    written = []
    recorder = play.VideoRecorder(model, spec,
        SimpleNamespace(append_data=lambda picture: written.append(picture.copy())),
        width=320, height=240, fps=50, dt=.02, steps=10,
        iteration=1, level=0, episodes=1, distance=2.8, presentation=presentation,
        hud="compact" if presentation else "full", shadows=presentation)
    # Goal marker rendering is covered separately; avoid constructing an OpenGL
    # scene while retaining the actual visual-model sync and recorder callback.
    monkeypatch.setattr(recorder, "_markers", lambda *args: None)
    recorder.begin_episode(1)
    callback_goals = []

    def record(sim, goals, step, metrics):
        state = torch.as_tensor(sim.read_state(), dtype=torch.float32)
        fp, fq = following_frame(state, task.cfg.goal_follow_frame)
        callback_goals.append(quat_apply_inverse(fq, goals[:, :3] - fp).clone())
        recorder(sim, goals, step, metrics)

    try:
        observed = evaluate_policy(actor, task, model, spec,
                                   frame_callback=record, **options)
    finally:
        recorder.close()

    assert observed == baseline
    assert observed["goal_source"] == "trajectory_library"
    assert observed["trajectory_library_fingerprint"] == library.fingerprint
    assert len(callback_goals) == len(written) == 10
    # The first Tb update writes the pose two samples ahead in the trajectory.
    torch.testing.assert_close(callback_goals[1] - callback_goals[0],
                               torch.tensor([[.002, 0., 0.]]), atol=1e-6, rtol=0.)
    for expected, actual in zip(baseline_inputs, inputs):
        for before, after in zip(expected, actual):
            torch.testing.assert_close(after, before, atol=0., rtol=0.)
    assert len(inputs) == len(baseline_inputs) == 10
    assert recorder.visuals.model is not model
    assert all(data is recorder.visuals.data and option is recorder.visuals.option
               for data, option in render_updates)
    assert model_manifest(spec)["physics_hash"] == before_hash
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == before_offscreen
