#!/usr/bin/env bash

FFMPEG=ffmpeg
FLOW_TYPE="${1:-u}"
FLOW_ROOT="flow/$FLOW_TYPE"
FLOW_VIDEOS_ROOT="flow_${FLOW_TYPE}_videos"

mkdir -p "$FLOW_VIDEOS_ROOT"
for vid_dir in $(find "${FLOW_ROOT}" -type d); do
    "$FFMPEG" -f image2 \
        -r 24 \
        -pattern_type glob \
        -i "$vid_dir/*.jpg" \
        -y \
        -c:v libx264 \
        "${FLOW_VIDEOS_ROOT}/$(basename "$vid_dir").mp4"
done
