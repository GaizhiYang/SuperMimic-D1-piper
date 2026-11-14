# Command-line entry points

Run these from the repository root after installing the package.

| Entry | Purpose |
|---|---|
| `python -m supermimic.wf.train` | Whole-body collection, dynamics fitting and BPTT |
| `python -m supermimic.wf.evaluate` | Physical task evaluation |
| `python -m supermimic.wf.play` | Annotated MP4 recording of a saved policy in real MuJoCo physics |
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
