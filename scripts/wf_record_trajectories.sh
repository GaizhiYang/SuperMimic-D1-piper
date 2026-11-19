#!/usr/bin/env bash
# Record continuous TCP trajectories using a saved WF policy.
# Usage: bash scripts/wf_record_trajectories.sh TRAINING_DIR [play_trajectories options]
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/wf_record_trajectories.sh TRAINING_DIR [options]" >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "usage: bash scripts/wf_record_trajectories.sh TRAINING_DIR [options]"
  echo "Options are passed to python -m supermimic.wf.play_trajectories."
  exit 0
fi

TRAINING_DIR=$1
shift
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WFARM_ASSET_DIR=${WFARM_ASSET_DIR:-$PWD/wf-arm-tbot}

exec "$PYTHON" -m supermimic.wf.play_trajectories --checkpoint "$TRAINING_DIR" "$@"
