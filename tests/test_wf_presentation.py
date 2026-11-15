"""Presentation framing and overlays remain observations of physical state."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _recorder(*, camera_mode="follow", hud="none"):
    from PIL import Image, ImageDraw, ImageFont

    from supermimic.wf.play import VideoRecorder

    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder.Image, recorder.ImageDraw = Image, ImageDraw
    recorder.font = recorder.caption_font = ImageFont.load_default()
    # An asymmetric image makes unexpected replacement, cropping, or flipping
    # visible without requiring any graphics driver or private robot assets.
    frame = np.full((240, 640, 3), [210, 225, 240], dtype=np.uint8)
    frame[120:, 320:] = [25, 70, 120]
    scenes, written = [], []
    recorder.renderer = SimpleNamespace(
        update_scene=lambda data, **kwargs: scenes.append(
            (data, kwargs["camera"].lookat.copy())),
        render=lambda: frame.copy(),
    )
    recorder.visuals = SimpleNamespace(sync=lambda data: data, option=None)
    recorder._markers = lambda *args: None
    recorder.writer = SimpleNamespace(append_data=lambda picture: written.append(picture.copy()))
    recorder.spec = SimpleNamespace(base_body=0)
    recorder.camera = SimpleNamespace(lookat=np.zeros(3))
    recorder.goal_source = "trajectory_library"
    recorder.presentation, recorder.hud, recorder.camera_mode = True, hud, camera_mode
    recorder.title = "Policy demonstration"
    recorder.fps, recorder.dt, recorder.steps = 50, .02, 100
    recorder.iteration, recorder.level, recorder.episodes = 1, 0, 1
    recorder.total_frames = 0
    recorder.begin_episode(1)
    return recorder, frame, scenes, written


def _rollout_inputs():
    sim = SimpleNamespace(data=[SimpleNamespace(
        xpos=np.array([[.2, -.1, .5]]), qpos=np.array([.2, -.1, .5, 1., 0., 0., 0.]))])
    goal = torch.tensor([[.5, 0., .6, 1., 0., 0., 0.]])
    metrics = dict(failed=torch.tensor([False]), held_success=torch.tensor([False]),
                   position_error_m=torch.tensor([.02]), orientation_error_rad=torch.tensor([.1]))
    return sim, goal, metrics


@pytest.mark.parametrize("camera_mode", ["fixed", "follow"])
def test_presentation_camera_observes_motion_without_altering_rollout(camera_mode):
    recorder, frame, scenes, written = _recorder(camera_mode=camera_mode)
    sim, goal, metrics = _rollout_inputs()
    qpos_before, goal_before = sim.data[0].qpos.copy(), goal.clone()

    recorder(sim, goal, 1, metrics)
    first_camera = recorder.camera.lookat.copy()
    # Emulate subsequent physical motion, independent of camera movement.
    sim.data[0].xpos[0] += [1., .5, 0.]
    state_before = sim.data[0].xpos.copy()
    recorder(sim, goal, 2, metrics)

    np.testing.assert_array_equal(first_camera, [.2, -.1, .85])
    if camera_mode == "fixed":
        np.testing.assert_array_equal(recorder.camera.lookat, first_camera)
    else:
        # Studio follow eases toward the new position instead of jumping.
        target = state_before[0] + [0., 0., .35]
        assert np.all(recorder.camera.lookat[:2] > first_camera[:2])
        assert np.all(recorder.camera.lookat[:2] < target[:2])
        assert recorder.camera.lookat[2] == first_camera[2]
    np.testing.assert_array_equal(sim.data[0].xpos, state_before)
    np.testing.assert_array_equal(sim.data[0].qpos, qpos_before)
    torch.testing.assert_close(goal, goal_before, atol=0., rtol=0.)
    assert len(scenes) == len(written) == 2
    assert all(scene_data is sim.data[0] for scene_data, _ in scenes)
    np.testing.assert_array_equal(written[-1], frame)


def test_new_episode_recenters_fixed_camera_on_new_physical_start():
    recorder, _, _, _ = _recorder(camera_mode="fixed")
    sim, goal, metrics = _rollout_inputs()
    recorder(sim, goal, 1, metrics)
    sim.data[0].xpos[0] = [3., 2., .4]

    recorder.begin_episode(2)
    recorder(sim, goal, 1, metrics)

    np.testing.assert_allclose(recorder.camera.lookat, [3., 2., .75], atol=1e-12)
    assert recorder.episode == 2
    assert recorder.frames == 1


def test_compact_hud_keeps_center_of_image_visible_and_preserves_input_frame():
    recorder, frame, _, written = _recorder(hud="compact")
    sim, goal, metrics = _rollout_inputs()
    original_frame = frame.copy()
    recorder(sim, goal, 1, metrics)

    picture = written[0]
    assert picture.shape == frame.shape
    # Compact title and progress bar leave the main robot viewing area intact.
    np.testing.assert_array_equal(picture[100:220], original_frame[100:220])
    assert not np.array_equal(picture[:90], original_frame[:90])
    assert not np.array_equal(picture[-15:], original_frame[-15:])
    np.testing.assert_array_equal(frame, original_frame)
