"""Record a saved WF policy in real MuJoCo physics to an annotated MP4.

Example: python -m supermimic.wf.play --checkpoint outputs/wf_cmd_gpu --episodes 3
Uses EGL by default; set MUJOCO_GL before starting to select another backend.
"""
from __future__ import annotations

import argparse
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
from .robot import build_model
from .simulator import BatchedSimulator
from .task import WholeBodyRewardCfg, WholeBodyTask


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
                 iteration, level, episodes, distance, appearance="visual", visual_scene=None):
        from PIL import Image, ImageDraw, ImageFont
        from .visuals import create_visual_scene

        if appearance not in ("visual", "collision"):
            raise ValueError("appearance must be visual or collision")
        self.visual_scene = visual_scene or create_visual_scene(model, spec)
        self.visual_scene.option.geomgroup[2] = appearance == "visual"
        self.visual_scene.option.geomgroup[3] = appearance == "collision"
        render_model = self.visual_scene.model
        self.Image, self.ImageDraw = Image, ImageDraw
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", max(14, height // 38))
        except OSError:
            self.font = ImageFont.load_default()
        render_model.vis.global_.offwidth = max(render_model.vis.global_.offwidth, width)
        render_model.vis.global_.offheight = max(render_model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(render_model, height=height, width=width)
        self.spec, self.writer = spec, writer
        self.fps, self.dt, self.steps = fps, dt, steps
        self.iteration, self.level, self.episodes = iteration, level, episodes
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, 135., -22.
        self.total_frames = 0

    def begin_episode(self, episode):
        self.episode, self.frames = episode, 0
        self.last_frame = None

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
        self.camera.lookat[:] = data.xpos[self.spec.base_body] + getattr(self, "camera_offset", [0, 0, .25])
        visual_scene = getattr(self, "visual_scene", None)
        if visual_scene is not None:
            visual_scene.update(data)
            self.renderer.update_scene(visual_scene.data, camera=self.camera,
                                       scene_option=visual_scene.option)
        else:
            self.renderer.update_scene(data, camera=self.camera)
        self._markers(data, goal)
        picture = self.Image.fromarray(self.renderer.render())
        draw = self.ImageDraw.Draw(picture)
        lines = self.annotation_lines(goal, step, metrics)
        line_height = max(22, picture.height // 30)
        draw.rectangle((0, 0, picture.width, len(lines) * line_height + 12), fill=(18, 23, 32))
        for i, line in enumerate(lines):
            draw.text((12, 6 + i * line_height), line, font=self.font, fill=(235, 240, 245))
        self.last_frame = np.asarray(picture)
        while self.frames < desired:
            self.writer.append_data(self.last_frame)
            self.frames += 1
            self.total_frames += 1

    def annotation_lines(self, goal, step, metrics):
        status = "FAILED" if bool(metrics["failed"][0]) else "HELD SUCCESS" if bool(metrics["held_success"][0]) else "TRACKING"
        lines = [f"Policy iteration {self.iteration} | episode {self.episode}/{self.episodes} | level {self.level}",
                 f"t={step * self.dt:.2f}/{self.steps * self.dt:.2f}s | {status}",
                 f"TCP error: {float(metrics['position_error_m'][0]) * 100:.1f} cm | "
                 f"orientation: {math.degrees(float(metrics['orientation_error_rad'][0])):.1f} deg"]
        if len(goal) >= 11:
            lines.append(f"Base command: vx={goal[7]:+.3f} vy={goal[8]:+.3f} m/s "
                         f"wz={goal[9]:+.3f} rad/s | {'follow' if goal[10] > .5 else 'world target'}")
        lines.append("Green sphere: target | orange: TCP | RGB arrows: target orientation")
        return lines

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
    parser.add_argument("--video", help="output MP4; defaults to CHECKPOINT_DIR/videos/preview_iterNNNNNNN_levelN.mp4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--level", type=int, choices=range(4), help="default: checkpoint curriculum level, or 0")
    parser.add_argument("--steps", type=int, help="control steps per episode; defaults to training episode length")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="force a base velocity command in m/s, m/s, rad/s while tracking the TCP goal")
    parser.add_argument("--friction", type=float, default=1.)
    parser.add_argument("--device", default="cpu", help="policy inference device; physics stays on CPU")
    parser.add_argument("--asset-dir")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--appearance", choices=("visual", "collision"), default="visual",
                        help="URDF surface meshes (default), or collision geometry")
    args = parser.parse_args()
    if min(args.episodes, args.fps, args.width, args.height) < 1:
        parser.error("episodes, fps and frame dimensions must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("MP4 width and height must be even")
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("camera distance must be finite and positive")
    if args.command and not all(math.isfinite(x) for x in args.command):
        parser.error("velocity command must be finite")
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
    video = (Path(args.video).expanduser().resolve() if args.video else
             checkpoint.parent / "videos" / f"preview_iter{ck['iteration']:07d}_level{level}.mp4")
    if video.suffix.lower() != ".mp4":
        parser.error("--video must have an .mp4 extension")
    video.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with imageio.get_writer(str(video), fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_log_level="error") as writer:
        recorder = VideoRecorder(model, spec, writer, width=args.width, height=args.height,
            fps=args.fps, dt=cfg.sim.dt, steps=steps, iteration=ck["iteration"], level=level,
            episodes=args.episodes, distance=args.camera_distance, appearance=args.appearance)
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
                    arm_initial_position=cfg.sim.arm_initial_position, frame_callback=recorder)
                recorder.end_episode()
                results.append(result)
                print(json.dumps(dict(episode=episode + 1, seed=seed,
                    failed=result["records"][0]["failed"],
                    final_held_success=result["records"][0]["final_held_success"],
                    final_position_error_m=result["records"][0]["final_position_error_m"])), flush=True)
        finally:
            recorder.close()
    report = dict(checkpoint=str(checkpoint), checkpoint_iteration=ck["iteration"],
        video=str(video), fps=args.fps, frames=recorder.total_frames,
        duration_s=recorder.total_frames / args.fps, episode_end_pause_s=1.,
        command=args.command, appearance=args.appearance, floor="white_light_gray_grid",
        physics_hash=ck["manifest"]["physics_hash"], episodes=results)
    video.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(video=str(video), report=str(video.with_suffix('.json')),
                          checkpoint_iteration=ck["iteration"], duration_s=report["duration_s"])), flush=True)


if __name__ == "__main__":
    main()
