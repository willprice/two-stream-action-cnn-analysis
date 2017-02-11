#!/usr/bin/env bash
FFMPEG=ffmpeg
EXCITATION_MAPS_ROOT=excitation_maps
EXCITATION_VIDEOS_ROOT=excitation_videos

for video_excitation_maps in "${EXCITATION_MAPS_ROOT}"/*; do
    "$FFMPEG" -f image2 \
        -r 24 \
        -pattern_type glob \
        -i "$video_excitation_maps/*.jpg"\
        -y \
        "${EXCITATION_VIDEOS_ROOT}/$(basename "$video_excitation_maps").avi"
done
