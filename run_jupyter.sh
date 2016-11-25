#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JUPYTER="${JUPYTER:-jupyter}"
JUPYTER_CONFIG="$HOME/jupyter-configs/caffe-excitation-bp-config.py"

export PYTHONPATH="$SCRIPT_DIR/lib/:$PYTHONPATH"

cd "$SCRIPT_DIR"
"$JUPYTER" notebook --config="$JUPYTER_CONFIG"
