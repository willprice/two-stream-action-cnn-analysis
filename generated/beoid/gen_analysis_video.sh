#!/usr/bin/env bash

VIDEO_NAME="${1:-00_Desk2_pick-up_plug_334-366}"
TEMPORAL_ROOT=temporal/
SPATIAL_ROOT=spatial/
VIDEO="$TEMPORAL_ROOT/videos/${VIDEO_NAME}.avi"
FLOW_U_VIDEO="$TEMPORAL_ROOT/flow_u_videos/${VIDEO_NAME}.mp4"
FLOW_V_VIDEO="$TEMPORAL_ROOT/flow_v_videos/${VIDEO_NAME}.mp4"
TEMPORAL_EXCITATION_CONTRASTIVE_VIDEO="$TEMPORAL_ROOT/excitation_videos/contrastive/${VIDEO_NAME}.mp4"
TEMPORAL_EXCITATION_NON_CONTRASTIVE_VIDEO="$TEMPORAL_ROOT/excitation_videos/non-contrastive/${VIDEO_NAME}.mp4"
SPATIAL_EXCITATION_CONTRASTIVE_VIDEO="$SPATIAL_ROOT/excitation_videos/contrastive/${VIDEO_NAME}.mp4"
SPATIAL_EXCITATION_NON_CONTRASTIVE_VIDEO="$SPATIAL_ROOT/excitation_videos/non-contrastive/${VIDEO_NAME}.mp4"
TARGET_VIDEO="${2:-${VIDEO_NAME}_analysis.mp4}"
FPS=12

scale_filter="scale=height=256:width=-2"
drawtext_abstract_filter="drawtext=\
fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
fontsize=20:\
fontcolor=white@0.9:\
borderw=2:\
bordercolor=black@0.7:\
x=(w-text_w)/2:\
y=(h-text_h-line_h)"
timestamp_filter="drawtext=\
fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
fontsize=20:\
fontcolor=white@0.9:\
borderw=2:\
bordercolor=black@0.7:\
x=0:\
y=10:\
text='%{pts\:flt}'\
"

SKIP=0.2
FFMPEG_FILTER="\
[0:v]scale=height=960:width=-2,${drawtext_abstract_filter}:text='Source'[video];\
[1:v]scale=height=480:width=-2,${drawtext_abstract_filter}:text='Flow U'[flow_u];\
[2:v]scale=height=480:width=-2,${drawtext_abstract_filter}:text='Flow V',$timestamp_filter[flow_v];\
[flow_u][flow_v]vstack[flow];\
[3:v]${drawtext_abstract_filter}:text='Temporal Contrastive EBP'[temporal_contrastive_ebp];\
[4:v]${drawtext_abstract_filter}:text='Temporal EBP'[temporal_ebp];\
[temporal_contrastive_ebp][temporal_ebp]vstack[temporal];\
[5:v]${drawtext_abstract_filter}:text='Spatial Contrastive EBP'[spatial_contrastive_ebp];\
[6:v]${drawtext_abstract_filter}:text='Spatial EBP'[spatial_ebp];\
[spatial_contrastive_ebp][spatial_ebp]vstack[spatial];\
[video][spatial][temporal][flow]hstack=inputs=4[v]\
"

#    ffplay \
echo "$FFMPEG_FILTER"
ffmpeg \
    -i "$VIDEO" \
    -i "$FLOW_U_VIDEO" \
    -i "$FLOW_V_VIDEO" \
    -i "$TEMPORAL_EXCITATION_CONTRASTIVE_VIDEO" \
    -i "$TEMPORAL_EXCITATION_NON_CONTRASTIVE_VIDEO" \
    -i "$SPATIAL_EXCITATION_CONTRASTIVE_VIDEO" \
    -i "$SPATIAL_EXCITATION_NON_CONTRASTIVE_VIDEO" \
    -r "$FPS" \
    -filter_complex "$FFMPEG_FILTER" \
    -map "[v]" \
    -an \
    -y \
    "$TARGET_VIDEO"
