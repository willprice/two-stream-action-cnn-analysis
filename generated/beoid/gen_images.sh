#!/usr/bin/env bash
set -e # exit on error
export PYTHONPATH="$HOME/thesis/lib:$HOME/thesis/net_configs:$HOME/src/Caffe-ExcitationBP/python:$PYTHONPATH"

SCRIPT_ROOT="$HOME/thesis/scripts"
DATA_ROOT="$HOME/thesis/generated/beoid/test-1"
ARGS='--desaturate'
TEMPORAL_ARGS='--overlay-offset=10'

"${SCRIPT_ROOT}/generate_spatial_excitation_maps.py" $ARGS\
    "${DATA_ROOT}" \
    "${DATA_ROOT}/spatial_contrastive_excitation_maps.pickle" \
    "${DATA_ROOT}/spatial_contrastive"

"${SCRIPT_ROOT}/generate_spatial_excitation_maps.py" $ARGS\
    "${DATA_ROOT}" \
    "${DATA_ROOT}/spatial_non_contrastive_excitation_maps.pickle" \
    "${DATA_ROOT}/spatial_non_contrastive"

"${SCRIPT_ROOT}/generate_temporal_excitation_maps.py" $TEMPORAL_ARGS $ARGS\
    "${DATA_ROOT}" \
    "${DATA_ROOT}/temporal_contrastive_excitation_maps.pickle" \
    "${DATA_ROOT}/temporal_contrastive"

"${SCRIPT_ROOT}/generate_temporal_excitation_maps.py" $TEMPORAL_ARGS  $ARGS\
    "${DATA_ROOT}" \
    "${DATA_ROOT}/temporal_non_contrastive_excitation_maps.pickle" \
    "${DATA_ROOT}/temporal_non_contrastive"
