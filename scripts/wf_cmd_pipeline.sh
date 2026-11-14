#!/usr/bin/env bash
# Full WF velocity-command + TCP-pose pipeline on a GPU server.
#
#   stage 0  environment / asset / Warp-vs-CPU checks
#   stage 1  physically screened command references (CPU, parallel)
#   stage 2  training: reference world-model + BC pretraining, then online
#            4096-env MuJoCo Warp collection with world-model BPTT
#   stage 3  held-out MuJoCo evaluation (random, explicit commands, ablations)
#   stage 4  TorchScript export
#
# Usage:  bash scripts/wf_cmd_pipeline.sh [first_stage] [last_stage]
# Env:    CONFIG (default configs/examples/wf_cmd_gpu4096.json), REFS (outputs/wf_cmd_refs),
#         REF_COUNT (512), WORKERS (nproc-1), CUDA_VISIBLE_DEVICES, WFARM_ASSET_DIR
set -euo pipefail
cd "$(dirname "$0")/.."

FIRST=${1:-0}
LAST=${2:-4}
CONFIG=${CONFIG:-configs/examples/wf_cmd_gpu4096.json}
REFS=${REFS:-outputs/wf_cmd_refs}
REF_COUNT=${REF_COUNT:-512}
WORKERS=${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
OUT=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output'])" "$CONFIG")
LIB=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['demonstrations']['directory'])" "$CONFIG")
export WFARM_ASSET_DIR=${WFARM_ASSET_DIR:-$PWD/wf-arm-tbot}

run() { echo "+ $*" >&2; "$@"; }

if (( FIRST <= 0 && 0 <= LAST )); then
  run python -m pytest -q tests/test_release_core.py
  run python tools/check_assets.py
  run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name())"
  run python -m supermimic.wf.simulator_warp --device cuda:0 --num-envs 4096 --steps 100
fi

if (( FIRST <= 1 && 1 <= LAST )); then
  if [[ "$LIB" != "$REFS/library" ]]; then
    echo "config demonstrations.directory ($LIB) must equal \$REFS/library ($REFS/library)" >&2; exit 1
  fi
  run python -m supermimic.wf.command_reference --config "$CONFIG" --output "$REFS" \
      --count "$REF_COUNT" --workers "$WORKERS"
fi

if (( FIRST <= 2 && 2 <= LAST )); then
  mkdir -p "$OUT"
  if [[ -f "$OUT/ckpt_latest.pt" ]]; then
    run python -m supermimic.wf.train --config "$CONFIG" --resume "$OUT/ckpt_latest.pt" 2>&1 | tee -a "$OUT/train.log"
  else
    run python -m supermimic.wf.train --config "$CONFIG" 2>&1 | tee -a "$OUT/train.log"
  fi
fi

if (( FIRST <= 3 && 3 <= LAST )); then
  CK="$OUT/ckpt_latest.pt"
  for level in 0 1 2 3; do
    run python -m supermimic.wf.evaluate --checkpoint "$CK" --level "$level" --episodes 64 \
        --output "$OUT/eval_level${level}.json"
  done
  run python -m supermimic.wf.evaluate --checkpoint "$CK" --reference-library "$LIB" \
      --reference-split validation --output "$OUT/eval_reference_validation.json"
  run python -m supermimic.wf.evaluate --checkpoint "$CK" --level 2 --episodes 32 --ablation all \
      --output "$OUT/eval_ablation.json"
  for friction in 0.6 1.2; do
    run python -m supermimic.wf.evaluate --checkpoint "$CK" --level 2 --episodes 32 --friction "$friction" \
        --output "$OUT/eval_friction_${friction}.json"
  done
fi

if (( FIRST <= 4 && 4 <= LAST )); then
  run python -m supermimic.wf.export --checkpoint "$OUT/ckpt_latest.pt" --output "$OUT/policy.pt"
fi
