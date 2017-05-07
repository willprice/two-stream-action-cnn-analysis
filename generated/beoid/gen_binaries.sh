#!/usr/bin/env bash
export PYTHONPATH="$HOME/thesis/lib:$HOME/thesis/net_configs:$HOME/src/Caffe-ExcitationBP/python:$PYTHONPATH"

SCRIPT_ROOT="$HOME/thesis/scripts"
DATA_ROOT="$HOME/thesis/generated/beoid/test-1"
#DATA_ROOT="$HOME/thesis/generated/beoid/test-1-subset"
SPATIAL_CONFIG=beoid_spatial_config
TEMPORAL_CONFIG=beoid_temporal_config

"${SCRIPT_ROOT}/generate_spatial_excitation_maps_binary.py" \
    --contrastive \
    "$SPATIAL_CONFIG" \
    "$DATA_ROOT" \
    "${DATA_ROOT}/spatial_contrastive_excitation_maps.pickle"

"${SCRIPT_ROOT}/generate_spatial_excitation_maps_binary.py" \
    --no-contrastive \
    "$SPATIAL_CONFIG" \
    "$DATA_ROOT" \
    "${DATA_ROOT}/spatial_non_contrastive_excitation_maps.pickle"

"${SCRIPT_ROOT}/generate_temporal_excitation_maps_binary.py" \
    --contrastive \
    "$TEMPORAL_CONFIG" \
    "$DATA_ROOT" \
    "${DATA_ROOT}/temporal_contrastive_excitation_maps.pickle"

"${SCRIPT_ROOT}/generate_temporal_excitation_maps_binary.py" \
    --no-contrastive \
    "$TEMPORAL_CONFIG" \
    "$DATA_ROOT" \
    "${DATA_ROOT}/temporal_non_contrastive_excitation_maps.pickle"
