"""Recording observes policy evaluation without changing its physical rollout."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("step, dt, fps, expected", [
    (0, .02, 30., 0),
    (1, .02, 30., 0),
    (2, .02, 30., 1),
    (5, .02, 30., 3),
    (600, .02, 30., 360),
    (600, .02, 60., 720),
])
def test_frame_schedule_keeps_requested_video_duration(step, dt, fps, expected):
    from supermimic.wf.play import frame_count

    assert frame_count(step, dt, fps) == expected


def test_terminal_frame_is_recorded_between_scheduled_frames():
    from PIL import Image, ImageDraw, ImageFont

    from supermimic.wf.play import VideoRecorder

    written = []
    # Exercise frame scheduling and annotation without opening a GL context.
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder.Image, recorder.ImageDraw = Image, ImageDraw
    recorder.font = ImageFont.load_default()
    recorder.renderer = SimpleNamespace(update_scene=lambda *args, **kwargs: None,
        render=lambda: np.zeros((240, 320, 3), dtype=np.uint8))
    recorder._markers = lambda *args: None
    recorder.writer = SimpleNamespace(append_data=lambda picture: written.append(picture.copy()))
    recorder.spec = SimpleNamespace(base_body=0)
    recorder.camera = SimpleNamespace(lookat=np.zeros(3))
    recorder.fps, recorder.dt, recorder.steps = 30, .02, 10
    recorder.iteration, recorder.level, recorder.episodes = 3000, 0, 1
    recorder.total_frames = 1
    recorder.begin_episode(1)
    recorder.frames = 1  # A frame was already written after control step 2.
    sim = SimpleNamespace(data=[SimpleNamespace(xpos=np.array([[0., 0., .5]]))])
    goal = torch.tensor([[.5, 0., .6, 1., 0., 0., 0.]])
    metrics = dict(failed=torch.tensor([True]), held_success=torch.tensor([False]),
                   position_error_m=torch.tensor([.1]), orientation_error_rad=torch.tensor([.2]))

    recorder(sim, goal, 3, metrics)  # 0.06 s is between the 30 FPS frame boundaries.

    assert len(written) == 1
    assert recorder.frames == recorder.total_frames == 2
    assert recorder.last_frame is not None


@pytest.fixture
def physical_policy():
    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.config import PolicyConfig
    from supermimic.wf.networks import Actor
    from supermimic.wf.robot import build_model
    from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask

    model, spec = build_model(arm_action_scale=None)
    task = WholeBodyTask(model, spec, "cpu", WholeBodyRewardCfg(
        base_command_conditioning=True, base_command_fraction=1.,
    ))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actor = Actor(PolicyConfig(hidden=(16, 16), latent_dim=8,
                                   base_command_conditioning=True))
    return actor, task, model, spec


@pytest.mark.private_assets
def test_frame_callback_preserves_states_actions_goals_and_scores(physical_policy, monkeypatch):
    from supermimic.wf.evaluate import evaluate_policy

    actor, task, model, spec = physical_policy
    inputs = []
    original_input = task.actor_input

    def record_input(state, previous, goals, **kwargs):
        inputs.append(tuple(value.detach().clone() for value in (state, previous, goals)))
        return original_input(state, previous, goals, **kwargs)

    monkeypatch.setattr(task, "actor_input", record_input)
    options = dict(episodes=1, steps=10, hold_steps=1, seed=123,
                   settle_steps=20, goal_seed_device="cpu")
    baseline = evaluate_policy(actor, task, model, spec, **options)
    baseline_inputs = inputs[:]
    inputs.clear()
    frames = []

    def observe(sim, goals, step, metrics):
        frames.append((sim.read_state().copy(), sim.applied_action.copy(),
                       goals.detach().clone(), step))

    observed = evaluate_policy(actor, task, model, spec, frame_callback=observe, **options)

    assert observed == baseline
    assert actor.training  # Evaluation restores the caller's original mode.
    assert len(frames) == len(inputs) == len(baseline_inputs) == baseline["records"][0]["steps"]
    for before, after in zip(baseline_inputs, inputs):
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    # The next inference uses exactly the scene, executed command history and
    # transported goal observed by the preceding frame callback.
    for frame, next_input in zip(frames, inputs[1:]):
        state, executed, goals, _ = frame
        torch.testing.assert_close(next_input[0], torch.as_tensor(state, dtype=torch.float32))
        torch.testing.assert_close(next_input[1], torch.as_tensor(executed, dtype=torch.float32))
        torch.testing.assert_close(next_input[2], goals, rtol=0., atol=0.)
    assert [frame[3] for frame in frames] == list(range(1, len(frames) + 1))


@pytest.mark.private_assets
def test_frame_callback_observes_terminal_state_before_reset(physical_policy, monkeypatch):
    from supermimic.wf import evaluate

    actor, task, model, spec = physical_policy
    events = []
    last_step = []
    original_reset = evaluate.BatchedSimulator.reset
    original_step = evaluate.BatchedSimulator.step

    def reset(sim, *args, **kwargs):
        events.append("reset")
        return original_reset(sim, *args, **kwargs)

    def step(sim, *args, **kwargs):
        state = original_step(sim, *args, **kwargs)
        last_step[:] = [state.copy()]
        return state

    monkeypatch.setattr(evaluate.BatchedSimulator, "reset", reset)
    monkeypatch.setattr(evaluate.BatchedSimulator, "step", step)
    monkeypatch.setattr(evaluate, "physical_failure", lambda task, state, metrics:
                        torch.ones(len(state), dtype=torch.bool, device=state.device))
    terminal = []

    def observe(sim, goals, step, metrics):
        events.append("frame")
        assert step == 1
        assert bool(metrics["failed"][0])
        state = sim.read_state().copy()
        np.testing.assert_array_equal(state, last_step[0])
        terminal.append((sim, state))

    result = evaluate.evaluate_policy(actor, task, model, spec, episodes=1,
        steps=10, hold_steps=1, seed=123, frame_callback=observe)

    assert result["records"][0]["failed"]
    assert result["records"][0]["steps"] == 1
    assert len(terminal) == 1
    assert events[-2:] == ["frame", "reset"]
    sim, terminal_state = terminal[0]
    assert not np.array_equal(terminal_state, sim.read_state())


@pytest.mark.private_assets
def test_command_fixture_preserves_program_context_and_follow_target(physical_policy, monkeypatch):
    from supermimic.wf.config import PolicyConfig, TrainConfig
    from supermimic.wf.evaluate import evaluate_policy
    from supermimic.wf.math import quat_apply_inverse
    from supermimic.wf.networks import Actor
    from supermimic.wf.play import command_goal
    from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask

    _, _, model, spec = physical_policy
    cfg = TrainConfig()
    cfg.sim.reset_settle_steps = 20
    cfg.policy = PolicyConfig(hidden=(16, 16), architecture="goal_program", action_mode="target",
        full_goal_orientation=True, base_command_conditioning=True, program_context=True)
    cfg.reward = WholeBodyRewardCfg(full_goal_orientation=True, base_command_conditioning=True,
                                   goal_program=True, program_context=True)
    task = WholeBodyTask(model, spec, "cpu", cfg.reward)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actor = Actor(cfg.policy)
    goals = command_goal(task, model, spec, cfg, seed=123, level=0,
                         command=[.1, 0., .02], friction=1.)
    initial = []
    terminal = []
    original_input = task.actor_input

    def inspect_input(state, previous, goal, **kwargs):
        initial.append((state.clone(), goal.clone()))
        return original_input(state, previous, goal, **kwargs)

    def observe(sim, goal, step, metrics):
        terminal.append((torch.as_tensor(sim.read_state(), dtype=torch.float32), goal.clone()))

    monkeypatch.setattr(task, "actor_input", inspect_input)
    evaluate_policy(actor, task, model, spec, episodes=1, steps=1, hold_steps=1,
        seed=123, goals=goals, settle_steps=cfg.sim.reset_settle_steps, frame_callback=observe)

    before, initial_goal = initial[0]
    after, final_goal = terminal[0]
    # Goal sampling and evaluation must use the same settled physical origin;
    # otherwise a 37-dimensional policy silently starts with stale context.
    assert initial_goal.shape == (1, 37)
    torch.testing.assert_close(initial_goal, goals, rtol=0., atol=0.)
    torch.testing.assert_close(initial_goal[:, 12:19], before[:, :7], rtol=0., atol=0.)
    torch.testing.assert_close(task.initialize_goal(before, initial_goal), initial_goal)
    torch.testing.assert_close(initial_goal[:, 7:11], torch.tensor([[.1, 0., .02, 1.]]))
    # A command goal follows the actual base motion while its program origin
    # stays fixed. Check this downstream behavior, including program advancement.
    local_before = quat_apply_inverse(before[:, 3:7], initial_goal[:, :3] - before[:, :3])
    local_after = quat_apply_inverse(after[:, 3:7], final_goal[:, :3] - after[:, :3])
    torch.testing.assert_close(local_after, local_before, atol=1e-6, rtol=1e-6)
    assert not torch.equal(final_goal[:, :3], initial_goal[:, :3])
    torch.testing.assert_close(final_goal[:, 12:], initial_goal[:, 12:], rtol=0., atol=0.)
    assert final_goal[0, 11] > initial_goal[0, 11]
