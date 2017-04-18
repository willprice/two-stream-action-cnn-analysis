#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JUPYTER="${JUPYTER:-jupyter}"
JUPYTER_CONFIG="$HOME/jupyter-configs/caffe-excitation-bp-config.py"

export PYTHONPATH="$HOME/src/Caffe-ExcitationBP/python:$SCRIPT_DIR/net_configs:$SCRIPT_DIR/lib:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

CUDNN_ROOT="$HOME/lib/cudnn-8.0-v5.1/"
export LD_LIBRARY_PATH="$CUDNN_ROOT/cuda/lib64:$LD_LIBRARY_PATH"

cd "$SCRIPT_DIR"
"$JUPYTER" notebook --config="$JUPYTER_CONFIG"
