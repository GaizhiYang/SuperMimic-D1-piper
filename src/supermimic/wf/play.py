"""Record a saved WF policy in real MuJoCo physics to an annotated MP4.

Example: python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu --episodes 3
Uses EGL by default; set MUJOCO_GL before starting to select another backend.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

# MuJoCo chooses the GL backend at import time. Physics and policy inference
# stay on CPU by default; EGL provides rendering without a desktop window.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from .evaluate import config_from_dict, evaluate_policy
from .networks import Actor
from .play_trajectory import load_playback_trajectories, sequential_trajectory_playback
from .robot import build_model
from .simulator import BatchedSimulator
from .task import WholeBodyRewardCfg, WholeBodyTask
from .visuals import RobotVisuals


def frame_count(step: int, dt: float, fps: int) -> int:
    """Resample control steps to video frames while preserving playback speed."""
    return math.floor(step * dt * fps + 1e-9)


def command_goal(task, model, spec, cfg, seed, level, command, friction):
    """Use the same settled fixture and seed as evaluate_policy, then set velocity."""
    sim = BatchedSimulator(model, spec, num_envs=1, seed=seed, dt=cfg.sim.dt,
                           friction=friction, arm_initial_position=cfg.sim.arm_initial_position)
    try:
        sim.reset()
        sim.settle(steps=cfg.sim.reset_settle_steps)
        state = torch.as_tensor(sim.read_state(), dtype=torch.float32, device=task.device)
        sample_task = task if task.device.type == "cpu" else WholeBodyTask(model, spec, "cpu", task.cfg)
        goals = sample_task.sample_goals(state.cpu(), level=level,
                                        generator=torch.Generator().manual_seed(seed)).to(task.device)
        goals[..., 7:10] = goals.new_tensor(command)
        goals[..., 10] = 1.
        return goals
    finally:
        sim.close()


class VideoRecorder:
    def __init__(self, model, spec, writer, *, width, height, fps, dt, steps,
                 iteration, level, episodes, distance, goal_source="trajectory_library",
                 presentation=False, hud="full", title="Whole-body Manipulation",
                 azimuth=135., elevation=-22., camera_mode="follow", shadows=False,
                 reflections=False, msaa=4):
        from PIL import Image, ImageDraw, ImageFont

        self.Image, self.ImageDraw = Image, ImageDraw
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", max(14, height // 38))
            self.caption_font = ImageFont.truetype("DejaVuSans.ttf", max(12, height // 54))
        except OSError:
            self.font = ImageFont.load_default()
            self.caption_font = self.font
        self.visuals = RobotVisuals(model, spec, presentation=presentation,
                                   shadows=shadows, reflections=reflections, msaa=msaa)
        render_model = self.visuals.model
        render_model.vis.global_.offwidth = max(render_model.vis.global_.offwidth, width)
        render_model.vis.global_.offheight = max(render_model.vis.global_.offheight, height)
        if not presentation and not shadows:
            render_model.vis.headlight.ambient[:] = .35
            render_model.vis.headlight.diffuse[:] = .65
            render_model.vis.headlight.specular[:] = .15
        self.renderer = mujoco.Renderer(render_model, height=height, width=width)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflections
        self.spec, self.writer = spec, writer
        self.goal_source = goal_source
        self.presentation, self.hud, self.title = presentation, hud, title
        self.camera_mode = camera_mode
        self.fps, self.dt, self.steps = fps, dt, steps
        self.iteration, self.level, self.episodes = iteration, level, episodes
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, azimuth, elevation
        self.total_frames = 0

    def begin_episode(self, episode):
        self.episode, self.frames = episode, 0
        self.last_frame = None
        self.camera_initialized = False
        self.camera_step = 0

    def _geom(self, kind, size, position, color):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("render scene has no space for goal markers")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float),
                           np.asarray(position, dtype=float), np.eye(3).reshape(-1),
                           np.asarray(color, dtype=np.float32))
        scene.ngeom += 1
        return geom

    def _markers(self, data, goal):
        target = goal[:3]
        tcp = data.site_xpos[self.spec.tcp_site]
        self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.022, 0, 0], target, [0.1, 1., .25, .8])
        self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.012, 0, 0], tcp, [1., .6, .05, 1.])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(goal[3:7], dtype=np.float64))
        for axis, color in zip(rotation.reshape(3, 3).T,
                               ([1., .1, .1, 1.], [.1, 1., .1, 1.], [.1, .4, 1., 1.])):
            geom = self._geom(mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], target, color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .006, target, target + .12 * axis)

    def __call__(self, sim, goals, step, metrics):
        failed = bool(metrics["failed"][0])
        desired = frame_count(step, self.dt, self.fps)
        # A failure between scheduled frames must still show its terminal state.
        if failed:
            desired = max(desired, self.frames + 1)
        if desired <= self.frames:
            return
        data = sim.data[0]
        goal = goals[0].detach().cpu().numpy().astype(float)
        target = data.xpos[self.spec.base_body] + [0, 0, .35 if self.presentation else .25]
        if not self.camera_initialized:
            self.camera.lookat[:] = target
            self.camera_initialized = True
        elif self.camera_mode == "follow":
            alpha = 1. - math.exp(-(step - self.camera_step) * self.dt / .18) if self.presentation else 1.
            self.camera.lookat[:] += alpha * (target - self.camera.lookat)
        self.camera_step = step
        render_data = self.visuals.sync(data)
        self.renderer.update_scene(render_data, camera=self.camera, scene_option=self.visuals.option)
        self._markers(data, goal)
        picture = self.Image.fromarray(self.renderer.render())
        picture = self._annotate(picture, goal, step, metrics, failed)
        self.last_frame = np.asarray(picture)
        while self.frames < desired:
            self.writer.append_data(self.last_frame)
            self.frames += 1
            self.total_frames += 1

    def _annotate(self, picture, goal, step, metrics, failed):
        if self.hud == "none":
            return picture
        if self.hud == "compact":
            overlay = self.Image.new("RGBA", picture.size, (0, 0, 0, 0))
            draw = self.ImageDraw.Draw(overlay)
            pad = max(12, picture.height // 55)
            line = max(24, picture.height // 30)
            detail = (f"{step * self.dt:.1f} / {self.steps * self.dt:.1f} s  |  "
                      f"Position error {float(metrics['position_error_m'][0]) * 100:.1f} cm  |  "
                      f"Orientation error {math.degrees(float(metrics['orientation_error_rad'][0])):.1f} deg")
            if failed:
                detail += "  |  FAILED"
            available = picture.width - 4 * pad
            if draw.textbbox((0, 0), detail, font=self.caption_font)[2] > available:
                detail = (f"{step * self.dt:.1f}/{self.steps * self.dt:.1f}s | "
                          f"TCP {float(metrics['position_error_m'][0]) * 100:.1f}cm | "
                          f"rot {math.degrees(float(metrics['orientation_error_rad'][0])):.1f}deg"
                          + (" | FAILED" if failed else ""))
            title = self.title
            while title and draw.textbbox((0, 0), title, font=self.font)[2] > available:
                title = title[:-2].rstrip() + "…" if len(title) > 2 else ""
            box_width = min(picture.width - 2 * pad, max(
                draw.textbbox((0, 0), title, font=self.font)[2],
                draw.textbbox((0, 0), detail, font=self.caption_font)[2]) + 2 * pad)
            draw.rounded_rectangle((pad, pad, pad + box_width, pad * 3 + line * 2),
                                   radius=pad // 2, fill=(20, 28, 40, 215))
            draw.text((pad * 2, pad * 2), title, font=self.font, fill=(245, 248, 252, 255))
            draw.text((pad * 2, pad * 2 + line), detail, font=self.caption_font,
                      fill=(255, 160, 130, 255) if failed else (213, 225, 238, 255))
            progress = min(1., step / self.steps)
            y, thickness = picture.height - pad, max(3, picture.height // 200)
            draw.rectangle((pad, y - thickness, picture.width - pad, y), fill=(25, 35, 48, 90))
            draw.rectangle((pad, y - thickness, pad + progress * (picture.width - 2 * pad), y),
                           fill=(45, 147, 194, 220))
            return self.Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")
        draw = self.ImageDraw.Draw(picture)
        status = ("FAILED" if failed else "TRAJECTORY TRACKING" if self.goal_source == "trajectory_library"
                  else "HELD SUCCESS" if bool(metrics["held_success"][0]) else "TRACKING")
        lines = [f"Policy iteration {self.iteration} | episode {self.episode}/{self.episodes} | level {self.level}",
                 f"t={step * self.dt:.2f}/{self.steps * self.dt:.2f}s | {status}",
                 f"TCP error: {float(metrics['position_error_m'][0]) * 100:.1f} cm | "
                 f"orientation: {math.degrees(float(metrics['orientation_error_rad'][0])):.1f} deg"]
        if len(goal) >= 11:
            lines.append(f"Base command: vx={goal[7]:+.3f} vy={goal[8]:+.3f} m/s "
                         f"wz={goal[9]:+.3f} rad/s | {'follow' if goal[10] > .5 else 'world target'}")
        lines.append("Green sphere: target | orange: TCP | RGB arrows: target orientation")
        line_height = max(22, picture.height // 30)
        draw.rectangle((0, 0, picture.width, len(lines) * line_height + 12), fill=(18, 23, 32))
        for i, line in enumerate(lines):
            draw.text((12, 6 + i * line_height), line, font=self.font, fill=(235, 240, 245))
        return picture

    def end_episode(self):
        # Pause on the last state so failures/episode transitions are easy to see.
        if self.last_frame is not None:
            for _ in range(self.fps):
                self.writer.append_data(self.last_frame)
                self.total_frames += 1

    def close(self):
        self.renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint file or training directory")
    parser.add_argument("--video", help="output MP4; default: CHECKPOINT_DIR/videos/preview_iterNNNNNNN_levelN_trajectory.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--level", type=int, choices=range(4), help="default: checkpoint curriculum level, or 0")
    parser.add_argument("--steps", type=int, help="control steps per episode; defaults to training episode length")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="force a base velocity command in m/s, m/s, rad/s while tracking the TCP goal")
    parser.add_argument("--friction", type=float, default=1.)
    parser.add_argument("--device", default="cpu", help="policy inference device; physics stays on CPU")
    parser.add_argument("--asset-dir")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--trajectory-library", help="directory containing manifest.json and trajectories.npz; "
                         "default: checkpoint library, or a compatible library under outputs/ for older checkpoints")
    targets.add_argument("--static-goals", action="store_true", help="sample static TCP goals instead of library trajectories")
    parser.add_argument("--full-trajectory", action="store_true", help="play a complete library trajectory from its first sample, "
                        "updating every control step; default episode length becomes the trajectory duration")
    parser.add_argument("--presentation", action="store_true", help="presentation preset: 1080p/50 FPS, studio lights, "
                        "shadows, 4x MSAA, closer smooth-follow camera and compact HUD; individual flags override defaults")
    parser.add_argument("--hud", choices=("full", "compact", "none"), help="default: full; presentation: compact")
    parser.add_argument("--title", default="Whole-body Manipulation", help="title shown in compact HUD")
    parser.add_argument("--shadows", action=argparse.BooleanOptionalAction, default=None,
                        help="enable cast shadows; enabled by presentation preset")
    parser.add_argument("--reflections", action=argparse.BooleanOptionalAction, default=False,
                        help="subtle planar floor reflections; disabled by default")
    parser.add_argument("--msaa", type=int, choices=(0, 2, 4, 8), default=4,
                        help="offscreen multisample anti-aliasing (default: 4)")
    parser.add_argument("--fps", type=int, help="default: 30; presentation: 50 (matches 20 ms control steps)")
    parser.add_argument("--width", type=int, help="default: 1280; presentation: 1920")
    parser.add_argument("--height", type=int, help="default: 720; presentation: 1080")
    parser.add_argument("--camera-distance", type=float, help="default: 2.8 m; presentation: 2.5 m")
    parser.add_argument("--camera-azimuth", type=float, default=135., help="camera yaw in degrees (default: 135)")
    parser.add_argument("--camera-elevation", type=float, help="camera elevation in degrees; default: -22; presentation: -18")
    parser.add_argument("--camera-mode", choices=("follow", "fixed"), default="follow")
    args = parser.parse_args()
    args.fps = args.fps if args.fps is not None else 50 if args.presentation else 30
    args.width = args.width if args.width is not None else 1920 if args.presentation else 1280
    args.height = args.height if args.height is not None else 1080 if args.presentation else 720
    args.camera_distance = args.camera_distance if args.camera_distance is not None else 2.5 if args.presentation else 2.8
    args.camera_elevation = args.camera_elevation if args.camera_elevation is not None else -18. if args.presentation else -22.
    args.hud = args.hud if args.hud is not None else "compact" if args.presentation else "full"
    args.shadows = args.shadows if args.shadows is not None else args.presentation
    if min(args.episodes, args.fps, args.width, args.height) < 1:
        parser.error("episodes, fps and frame dimensions must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("MP4 width and height must be even")
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("camera distance must be finite and positive")
    if not all(math.isfinite(value) for value in (args.camera_azimuth, args.camera_elevation)):
        parser.error("camera angles must be finite")
    if not -89 <= args.camera_elevation <= 89:
        parser.error("camera elevation must be in [-89, 89] degrees")
    if args.command and not all(math.isfinite(x) for x in args.command):
        parser.error("velocity command must be finite")
    if args.full_trajectory and args.static_goals:
        parser.error("--full-trajectory requires trajectory goals")
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 - validate the encoder before simulation
    except ImportError:
        parser.error("MP4 dependencies missing: python -m pip install 'imageio>=2.30,<3' 'imageio-ffmpeg>=0.5,<1'")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "ckpt_latest.pt"
    torch.set_num_threads(4)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "wf_goal_bptt_v1":
        raise ValueError("expected a WF policy checkpoint")
    cfg = config_from_dict(ck["config"])
    level = args.level if args.level is not None else ck.get("level", 0)
    steps = args.steps if args.steps is not None else cfg.sim.episode_steps
    if steps < cfg.curriculum.hold_steps:
        parser.error(f"steps must be >= success hold length ({cfg.curriculum.hold_steps})")
    model, spec = build_model(args.asset_dir or cfg.asset_dir, cfg.sim.timestep,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale)
    from .train import model_manifest
    if model_manifest(spec)["physics_hash"] != ck["manifest"]["physics_hash"]:
        raise ValueError("checkpoint physics differs; use its original assets/configuration")
    task = WholeBodyTask(model, spec, args.device, WholeBodyRewardCfg(**ck["reward_config"]))
    actor = Actor(cfg.policy).to(args.device)
    actor.load_state_dict(ck["actor"])
    if args.command and not task.cfg.base_command_conditioning:
        parser.error("this checkpoint does not support base velocity commands")
    trajectories, trajectory_cfg = None, None
    if not args.static_goals:
        trajectories, trajectory_cfg = load_playback_trajectories(
            cfg, task, ck["manifest"]["physics_hash"], checkpoint,
            explicit=args.trajectory_library,
            expected_fingerprint=ck.get("trajectory_fingerprint"), level=level)
        if args.full_trajectory:
            trajectories, trajectory_cfg = sequential_trajectory_playback(trajectories, trajectory_cfg)
            if args.steps is None:
                steps = trajectories.length - 2
            if steps < cfg.curriculum.hold_steps:
                parser.error(f"trajectory duration must be >= success hold length ({cfg.curriculum.hold_steps} steps)")
        print(json.dumps(dict(goal_source="trajectory_library", trajectory_library=trajectory_cfg.library,
            trajectory_library_fingerprint=trajectories.fingerprint,
            segment_seconds=trajectory_cfg.segment_steps * cfg.sim.dt,
            goal_update_seconds=trajectory_cfg.update_steps * cfg.sim.dt)), flush=True)
    goal_source = "static" if args.static_goals else "trajectory_library"
    source_suffix = "static" if args.static_goals else "trajectory"
    if args.presentation:
        source_suffix += "_presentation"
    video = (Path(args.video).expanduser().resolve() if args.video else
             checkpoint.parent / "videos" / f"preview_iter{ck['iteration']:07d}_level{level}_{source_suffix}.mp4")
    if video.suffix.lower() != ".mp4":
        parser.error("--video must have an .mp4 extension")
    video.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with imageio.get_writer(str(video), fps=args.fps, codec="libx264", quality=9 if args.presentation else 8,
                            macro_block_size=1, ffmpeg_log_level="error") as writer:
        recorder = VideoRecorder(model, spec, writer, width=args.width, height=args.height,
            fps=args.fps, dt=cfg.sim.dt, steps=steps, iteration=ck["iteration"], level=level,
            episodes=args.episodes, distance=args.camera_distance, goal_source=goal_source,
            presentation=args.presentation, hud=args.hud, title=args.title,
            azimuth=args.camera_azimuth, elevation=args.camera_elevation, camera_mode=args.camera_mode,
            shadows=args.shadows, reflections=args.reflections, msaa=args.msaa)
        try:
            for episode in range(args.episodes):
                seed = args.seed + episode
                goals = (command_goal(task, model, spec, cfg, seed, level, args.command, args.friction)
                         if args.command is not None else None)
                recorder.begin_episode(episode + 1)
                result = evaluate_policy(actor, task, model, spec, episodes=1, level=level,
                    seed=seed, steps=steps, hold_steps=cfg.curriculum.hold_steps, dt=cfg.sim.dt,
                    friction=args.friction, goals=goals, settle_steps=cfg.sim.reset_settle_steps,
                    goal_seed_device=cfg.curriculum.goal_seed_device,
                    arm_initial_position=cfg.sim.arm_initial_position, frame_callback=recorder,
                    trajectories=trajectories, trajectory_cfg=trajectory_cfg)
                recorder.end_episode()
                results.append(result)
                print(json.dumps(dict(episode=episode + 1, seed=seed,
                    failed=result["records"][0]["failed"],
                    final_held_success=result["records"][0]["final_held_success"],
                    tracking_success_rate=result.get("tracking_success_rate"),
                    final_position_error_m=result["records"][0]["final_position_error_m"])), flush=True)
        finally:
            recorder.close()
    report = dict(checkpoint=str(checkpoint), checkpoint_iteration=ck["iteration"],
        video=str(video), fps=args.fps, frames=recorder.total_frames,
        duration_s=recorder.total_frames / args.fps, episode_end_pause_s=1.,
        goal_source=goal_source, trajectory_schedule=asdict(trajectory_cfg) if trajectory_cfg else None,
        full_trajectory=args.full_trajectory,
        trajectory_library_fingerprint=trajectories.fingerprint if trajectories else None,
        appearance="URDF visual meshes; white/gray checkerboard floor",
        rendering=dict(presentation=args.presentation, width=args.width, height=args.height,
            hud=args.hud, title=args.title, shadows=args.shadows, reflections=args.reflections,
            msaa=args.msaa, camera_distance=args.camera_distance, camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation, camera_mode=args.camera_mode),
        command=args.command, physics_hash=ck["manifest"]["physics_hash"], episodes=results)
    video.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(video=str(video), report=str(video.with_suffix('.json')),
                          checkpoint_iteration=ck["iteration"], duration_s=report["duration_s"])), flush=True)


if __name__ == "__main__":
    main()
