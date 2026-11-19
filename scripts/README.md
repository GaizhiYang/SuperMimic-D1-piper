# Command-line entry points

Run these from the repository root after installing the package.

| Entry | Purpose |
|---|---|
| `python -m supermimic.wf.train` | Whole-body collection, dynamics fitting and BPTT |
| `python -m supermimic.wf.evaluate` | Physical task evaluation |
| `python -m supermimic.wf.play` | Annotated MP4 recording of a saved policy in real MuJoCo physics |
| `bash scripts/wf_record_trajectories.sh TRAINING_DIR [options]` | Continuous, wide-range TCP trajectory videos for stationary, forward and turning scenarios |
| `bash scripts/wf_cmd_pipeline.sh 1 4` | Fresh reference/training run with matching timestamped directories |
| `python -m supermimic.wf.export` | Local policy export; exported files remain private |
| `wf_mimicagent_pilot.py` | Reference generation and paired physical screening |
| `wf_mimicagent_smoke.py` | Connect screened references to the learning loop |
| `eval_benchmark.py`, `eval_push.py`, `diag_drift.py` | Optional inherited G1 evaluation |
| `data/` | Optional G1 motion preprocessing |

Installation, required private assets and command examples are documented in
[English](../README.md) and [中文](../README_CN.md).

The WF pipeline saves `pipeline_config.json` in each new training directory.
Continue that run with `CONFIG=outputs/<run>/pipeline_config.json bash scripts/wf_cmd_pipeline.sh 2 4`.

Record the requested policy with `conda activate supermimic-wfarm`, then
`bash scripts/wf_record_trajectories.sh outputs/wf_cmd_gpu_20261116_190019_821523`.
The default records one full level-3 trajectory per scenario at 0.75 speed;
MP4s, metrics and traces go to a new `videos/trajectories_<timestamp>/` folder.
Set base motion with `--scenario forward --command 0.2 0 0.03`: body-frame
forward/lateral speeds are in m/s and yaw rate is in rad/s. `--command VX VY WZ`
overrides the selected scenarios' velocities; `--frame world` requires zero base commands.
The default `--appearance visual` renders original URDF visual meshes/materials on a bright
white/light-gray grid floor. Use `--appearance collision` to inspect collision geometry;
appearance changes affect rendering only.
See the [recording examples](../docs/WF_COMMAND_TRAINING_CN.md#录制连续末端轨迹视频) for slower playback and options.
