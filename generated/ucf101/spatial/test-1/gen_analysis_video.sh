#!/usr/bin/env bash

VIDEO_NAME="${1:-v_BoxingPunchingBag_g01_c03}"
VIDEO="videos/${VIDEO_NAME}.avi"
FLOW_U_VIDEO="flow_u_videos/${VIDEO_NAME}.mp4"
FLOW_V_VIDEO="flow_v_videos/${VIDEO_NAME}.mp4"
EXCITATION_CONTRASTIVE_VIDEO="excitation_videos/contrastive/${VIDEO_NAME}.mp4"
EXCITATION_NON_CONTRASTIVE_VIDEO="excitation_videos/non-contrastive/${VIDEO_NAME}.mp4"
TARGET_VIDEO="${2:-${VIDEO_NAME}_analysis.mp4}"

scale_filter="scale=height=256:width=-1"
drawtext_abstract_filter="drawtext=\
fontsize=20:\
fontcolor=white@0.9:\
borderw=2:\
bordercolor=black@0.7:\
x=(w-text_w)/2:\
y=(h-text_h-line_h)"

FFMPEG_FILTER="\
[0:v]${scale_filter},${drawtext_abstract_filter}:text='Source'[video];\
[1:v]${drawtext_abstract_filter}:text='Flow U'[flow_u];\
[2:v]${drawtext_abstract_filter}:text='Flow V'[flow_v];\
[3:v]${drawtext_abstract_filter}:text='Contrastive EBP'[contrastive_ebp];\
[4:v]${drawtext_abstract_filter}:text='EBP'[ebp];\
[video][flow_u][flow_v][contrastive_ebp][ebp]hstack=inputs=5[v]\
"

#    ffplay \
echo "$FFMPEG_FILTER"
ffmpeg \
    -i "$VIDEO" \
    -i "$FLOW_U_VIDEO" \
    -i "$FLOW_V_VIDEO" \
    -i "$EXCITATION_CONTRASTIVE_VIDEO" \
    -i "$EXCITATION_NON_CONTRASTIVE_VIDEO" \
    -filter_complex "$FFMPEG_FILTER" \
    -map "[v]" \
    -an \
    -y \
    "$TARGET_VIDEO"
