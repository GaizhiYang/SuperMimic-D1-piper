# SuperMimic-D1-Piper

[简体中文](README_CN.md) · **English** · [License](LICENSE) · [Attribution](NOTICE.md)

Learning wheel-legged mobile manipulation with a differentiable world model.
This project adapts the SuperMimic learning approach to a D1 platform with four
legs, four wheels and a six-axis Piper arm. A unified policy controls all 22
actuators while following end-effector poses and, when enabled, chassis velocity
commands.

The main workflow covers reference generation, continuous TCP trajectory
commands, CPU or GPU simulation, world-model training, policy optimization,
physical evaluation and video recording. Fixed-base arm experiments and the
inherited G1 implementation are also available as separate research branches.
This is an independent research adaptation; it does not reproduce or claim the
original papers' benchmark results or hardware validation.

## What is included

| Component | Implementation |
|---|---|
| Whole-body control | One Actor outputs 12 leg position commands, 4 wheel velocity commands and 6 arm position commands |
| Differentiable dynamics | Learn simulator state transitions, then optimize the Actor through short imagined rollouts using BPTT |
| Reference generation | Motor-driven MuJoCo rollouts with physical screening and optional imitation guidance |
| Moving TCP commands | Collision-checked joint splines converted to position and orientation trajectories |
| Simulation backends | Native CPU MuJoCo and batched MuJoCo Warp collection for GPU training |
| Evaluation | Pose reaching, trajectory tracking, physical failures, friction changes and actuator ablations |
| Recording | Full continuous trajectories, original robot visual meshes, a bright white/gray floor, target and actual TCP trails |
| Export | TorchScript inference wrapper with the Actor and observation transforms |

The control period in the example WF configurations is 20 ms (50 Hz), with a
5 ms physics timestep. Position commands use the configured actuator scales;
they are not joint torques. Quaternion inputs use **wxyz** order.

## How learning works

1. Generate reference motions and execute their motor commands in MuJoCo.
   Store accepted state/action transitions separately from TCP target paths.
2. Collect additional transitions by running the current policy in physics,
   then fit a differentiable dynamics model to replay windows.
3. Freeze the world-model parameters during policy updates and backpropagate
   task rewards through imagined rollouts. Optional behavior cloning and
   reference-tracking losses guide the Actor.
4. Evaluate the resulting Actor in MuJoCo. Curriculum and guard checks use
   physical rollouts to assess progress and candidate updates.

A reference motion that a teacher can execute is different from a task that the
learned Actor can complete. The TCP trajectory library provides commands and
nominal kinematic feasibility; it is not a collection of successful policy
rollouts. The learned world model and offline teacher are not needed for Actor
inference.

## Installation

Use Linux with Python **3.11 or 3.12**. The recording and GPU workflows have been
checked with MuJoCo **3.16**, PyTorch **2.7** and Python **3.11**. Start with an
isolated environment:

```bash
git clone https://github.com/GaizhiYang/SuperMimic-D1-piper.git
cd SuperMimic-D1-piper
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,media]' 'mujoco~=3.16.0'
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build appropriate for
your driver, then add the GPU backend:

```bash
python -m pip install -e '.[dev,media,gpu]'
```

| Extra | Purpose |
|---|---|
| `dev` | pytest |
| `media` | MP4 encoding, image annotations and plots |
| `gpu` | MuJoCo 3.16, MuJoCo Warp and Warp |
| `teacher` | Optional OSQP/quadprog teachers used by other WBC experiments |
| `g1` | Separate inherited G1 stack; not required for D1/Piper training |

The main command-reference and TCP-trajectory generators do not require the
`teacher` extra. Native CPU MuJoCo handles evaluation and recording even when
training collection uses Warp. Headless recording defaults to EGL; the host
must provide a working OpenGL/EGL runtime.

## Robot model

The current source tree includes `wf-arm-tbot/`, containing the URDF, referenced
collision and visual meshes, robot configuration and TCP definition. The asset
resolver finds this directory automatically. To select it explicitly, or use an
external copy:

```bash
export WFARM_ASSET_DIR="$PWD/wf-arm-tbot"
python tools/check_assets.py
```

Expected layout:

```text
wf-arm-tbot/
  urdf/wf-arm-tbot.urdf
  meshes/
  config/robot.json
  config/robot_calibration.json
  config/operating_limits.json     # also used by the fixed-base arm adapter
```

An explicit `--asset-dir` or configuration `asset_dir` overrides the environment
variable. Evaluation checks the model's physics fingerprint against the
checkpoint, so use the same assets and actuator configuration as training.
Asset availability and redistribution rights are separate; see [licensing](#licensing-and-acknowledgements).

## Quick start

Run commands from the repository root with the environment activated.

### Check the code and CPU training loop

```bash
# Core tests without loading the robot model.
python -m pytest -q -m 'not private_assets' tests/test_release_core.py

# Small integration run: 4 environments, 64 iterations.
python -m supermimic.wf.train --smoke --device cpu --output outputs/wf_smoke
```

The smoke run checks the training loop, not policy convergence. For a longer CPU
experiment, use or copy `configs/examples/wf_train.json` and set a fresh output
directory:

```bash
python -m supermimic.wf.train --config configs/examples/wf_train.json
```

### Train with chassis commands and moving TCP targets

The GPU pipeline performs these stages:

| Stage | Work |
|---|---|
| `0` | Dependency, asset, CUDA and Warp/CPU consistency checks |
| `1` | Generate physically screened references and TCP trajectory libraries |
| `2` | Reference pretraining, online collection, dynamics fitting and Actor/Critic updates |
| `3` | MuJoCo evaluation across curriculum levels, reference validation, friction and ablations |
| `4` | TorchScript export |

For a small GPU integration run:

```bash
CONFIG=configs/examples/wf_cmd_smoke.json \
REF_COUNT=10 TRAJ_PER_LEVEL=4 WORKERS=4 \
bash scripts/wf_cmd_pipeline.sh 0 4
```

For the full preset:

```bash
CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/wf_cmd_pipeline.sh 0 4
```

The default configuration is named `wf_cmd_gpu4096.json`, but currently specifies
**8192 environments and 30000 iterations**. Adjust environment counts, batch
sizes and generation workers for your hardware before starting a full run.
The small command preset also requires a CUDA GPU; use `train --smoke` for the
CPU check.

Including stage `1` creates matching timestamped reference and training
directories under `outputs/`. The script prints their paths and saves
`pipeline_config.json` in the training directory. Use that saved configuration
to continue the same run:

```bash
# Replace this value with the training directory printed by the pipeline.
RUN_DIR=outputs/wf_cmd_gpu_YYYYMMDD_HHMMSS_SUFFIX
CONFIG="$RUN_DIR/pipeline_config.json" bash scripts/wf_cmd_pipeline.sh 2 3
```

### Evaluate a checkpoint

Use a checkpoint **file** for the evaluation and export modules:

```bash
python -m supermimic.wf.evaluate \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --level 3 --episodes 64 \
  --output "$RUN_DIR/eval_level3.json"

python -m supermimic.wf.evaluate \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --static-goals --level 3 \
  --output "$RUN_DIR/eval_static.json"
```

Evaluation uses the checkpoint's TCP trajectory library by default when one is
configured. `--static-goals` selects reaching targets instead. Other options
include `--ablation all`, `--friction`, and `--reference-library` with
`--reference-split validation`. Keep the referenced libraries available when
moving a training directory.

Report trajectory tracking success, position/orientation RMS errors and physical
failure rate together. Tracking statistics count unexecuted steps after a
failure as misses. A low dynamics-model error, a successful teacher or an
attractive recording alone does not establish policy performance. Use separate
held-out requests to assess generalization.

### Record continuous trajectories

For a command-conditioned checkpoint with a generated TCP trajectory library:

```bash
# Record stationary, forward and gentle-turning scenarios.
bash scripts/wf_record_trajectories.sh "$RUN_DIR"

# Set chassis forward/lateral speeds and yaw rate: m/s, m/s, rad/s.
bash scripts/wf_record_trajectories.sh "$RUN_DIR" \
  --scenario turn --command 0.2 0 0.03

# Slow the TCP path and select three distinct large-range paths.
bash scripts/wf_record_trajectories.sh "$RUN_DIR" \
  --scenario stationary --speed 0.4 --clips 3
```

The recorder selects complete paths by spatial extent, adds a smooth approach
and updates the target every control step. It does not randomly switch trajectory
windows during a clip. The defaults are level 3, 0.75× path speed, 1280×720 and
30 FPS. Videos play at simulation time, independent of rendering speed.

`--command VX VY WZ` overrides the chosen scenarios' base commands. `--speed`
controls TCP trajectory playback separately. The default `--frame follow`
transports targets with the configured base-following frame;
`--scenario stationary --frame world` anchors the path to the initial world pose
and requires zero base velocity commands.

Original URDF visual meshes/materials and a bright white/light-gray grid floor
are enabled by default. Use `--appearance collision` to inspect collision
geometry. Rendering uses a separate model and leaves policy physics unchanged.
Cyan shows the complete target path, green the command trail and orange the
actual TCP trail.

Each recording saves MP4, JSON metrics, a control-rate NPZ trace and a PNG preview
under `"$RUN_DIR/videos/trajectories_<timestamp>/"`. These previews reuse the
training trajectory library; they do not measure unseen-trajectory performance.
For static-target recordings, use `python -m supermimic.wf.play --help`.
See the [recording guide](docs/WF_COMMAND_TRAINING_CN.md#录制连续末端轨迹视频) for more options.

### Export inference

```bash
python -m supermimic.wf.export \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" --output "$RUN_DIR/policy_inference.pt"
```

For the command-conditioned WF policy, the wrapper accepts the canonical
57-dimensional state, 22 previous executed actions and an 11-dimensional world
goal: TCP position, quaternion wxyz, chassis vx/vy/wz and a command-mode flag.
The adjacent export JSON describes the actual interface, joint order, actuator
scales and physics fingerprint. The caller must maintain executed-action history
and target-frame transport. Export does not include a simulator or hardware I/O.
Exports refuse to overwrite an existing model or metadata file. The full
pipeline already creates `policy.pt`; choose a fresh filename when exporting
again after continuation.

## Other experiments and source layout

```text
src/supermimic/wf/        Whole-body model, simulation, training, evaluation and recording
src/supermimic/mimicagent/ Task/plan/program reference frontend
src/supermimic/world_model/, imagination/, kinematics/
                         Shared learning and differentiable-kinematics utilities
frameworks/fixed_base_arm/ Six-axis baseline and structured-controller variant
configs/examples/        CPU and GPU experiment presets
scripts/                 Pipeline and recording wrappers
tests/                   Core, physics, trajectory and rendering checks
wf-arm-tbot/             Robot descriptions, meshes and model configuration
outputs/                 Generated runs; ignored by Git
docs/                    Detailed workflow and algorithm notes
```

The fixed-base arm baseline has a separate entry point:

```bash
python tools/check_assets.py --fixed-arm
PYTHONPATH=src:frameworks/fixed_base_arm python -m arm_smc.train \
  --config configs/examples/fixed_arm_smoke.json
```

The structured fixed-arm variant combines learned components with planning and
analytic Jacobian feedback; it is separate from the unified whole-body Actor.
See the [fixed-arm overview](frameworks/fixed_base_arm/README.md),
[algorithm notes](docs/PROJECT_ANALYSIS_CN.md) and [script index](scripts/README.md).
The inherited G1 code requires its own environment and data.

To run the robot-dependent WF tests explicitly:

```bash
WFARM_ASSET_DIR="$PWD/wf-arm-tbot" python -m pytest -q tests
```

## Licensing and acknowledgements

Project code is distributed under the [MIT license](LICENSE), retaining the
upstream copyright notice. This work extends the independent
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
implementation by Yuxi Liu / CuboiLeo and draws on the SuperMimic learning and
MimicAgent reference-generation approaches. Author attribution and research
references are listed in [NOTICE](NOTICE.md).

The code license does not grant additional rights to third-party robot
resources. The bundled D1 description currently contains an unfilled license
declaration in [package.xml](wf-arm-tbot/d1_description/package.xml); confirm the
model and mesh redistribution terms with their source providers. Preserve
separate notices, including the Apache-2.0 notice in the articulation adapter.

For bug reports, use [Issues](https://github.com/GaizhiYang/SuperMimic-D1-piper/issues)
and include the code revision, software versions, configuration and a short
reproduction. Generated references, logs, checkpoints and videos belong under
`outputs/`, which is excluded from Git tracking.
