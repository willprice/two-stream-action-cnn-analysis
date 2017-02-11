#!/usr/bin/env bash
SCRIPT_DIR="$(dirname $(realpath $0))"
# Config options for run
SRC_FLOW_U=${SCRIPT_DIR}/flow/u
SRC_FLOW_V=${SCRIPT_DIR}/flow/v
SRC_FRAMES=${SCRIPT_DIR}/frames

DEST_EXCITATION_MAPS=excitation_maps/contrastive
DEST_EXCITATION_VIDEOS=excitation_videos/contrastive

NET_CONFIG=ucf101_temporal_config

GEN_OPTIONS="--colormap hot --desaturate --contrastive"

# Config for scripts
export PYTHONPATH="$HOME/thesis/lib:$HOME/thesis/net_configs"
GEN_TEMPORAL_EXCITATION_MAPS_SCRIPT="$HOME/thesis/scripts/generate_temporal_excitation_maps.py"
STITCH_VIDEO_SCRIPT="$HOME/thesis/scripts/stitch_video.py"

gen_excitation_maps() {
    if [[ -d "$DEST_EXCITATION_MAPS" ]]; then
        echo "Destination for excitation maps: $DEST_EXCITATION_MAPS already exists"
        read -p "Continue and overwrite? " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    mkdir -p "$DEST_EXCITATION_MAPS" || exit 1
    for video_path in  $SRC_FRAMES/*; do
        local video_name=$(basename "$video_path")
        local destination="${DEST_EXCITATION_MAPS}/$video_name"
        mkdir -p "$destination" || exit 1
        "$GEN_TEMPORAL_EXCITATION_MAPS_SCRIPT" \
            $GEN_OPTIONS \
            "$NET_CONFIG" \
            "$SRC_FLOW_U" \
            "$SRC_FLOW_V" \
            "$SRC_FRAMES" \
            "$video_name" \
            "$destination" \
        || exit 1
    done

}

gen_excitation_videos() {
    pushd .
    mkdir -p "$DEST_EXCITATION_VIDEOS"
    for src_frame_path in "$DEST_EXCITATION_MAPS/"*; do
        ffmpeg -f image2 \
            -framerate 24 \
            -pattern_type glob \
            -i "${src_frame_path}/frame*.jpg" \
            -c:v libx264 \
            -y \
            "$DEST_EXCITATION_VIDEOS/$(basename "${src_frame_path}").mp4"
    done
    popd
}

main() {
    #gen_excitation_maps
    gen_excitation_videos
}

main
