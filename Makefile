DATA_ROOT := data
GENERATED_DATA_ROOT := generated
BEOID_DATA_ROOT := $(DATA_ROOT)/beoid/k-fold-1/test
# BEOID_DATA_ROOT should have the following directory structure
#
# .
# ├── spatial
# │   ├── 00_Desk2_pick-up_plug_334-366
# │   ├── 00_Desk2_pick-up_tape_1070-1099
# │   ├── ...
# ├── temporal
# │   ├── u
# │       ├── 00_Desk2_pick-up_plug_334-366
# │       ├── 00_Desk2_pick-up_tape_1070-1099
# │       ├── ...
# │   ├── v
# │       ├── 00_Desk2_pick-up_plug_334-366
# │       ├── 00_Desk2_pick-up_tape_1070-1099
# │       ├── ...
#
# where each directory containing frames should follow the BEOID 
# naming format:
#
# <beoid-video-name> ::= <video-set> '_' <location> '_' <action> '_' <objects> '_' <frame-range>
# <video-set> ::= \d{2}
# <location> ::= <clause>
# <action> ::= <clause>
# <objects> ::= <object> ('+' <object>)*
# <object> ::= <clause>
# <clause> ::= <word> ('-' <word>)*
# <word> :: [a-Z0-9]+
NET_ROOT := /home/will/nets/
NET_SPATIAL_ROOT := $(NET_ROOT)/dual-stream/spatial
NET_SPATIAL_CAFFEMODEL := $(NET_SPATIAL_ROOT)/kfold1borders25.caffemodel
NET_SPATIAL_PROTOTXT := $(NET_SPATIAL_ROOT)/deploy.prototxt
NET_TEMPORAL_ROOT := $(NET_ROOT)/dual-stream/temporal
NET_TEMPORAL_CAFFEMODEL := $(NET_TEMPORAL_ROOT)/kfold1temporal.caffemodel
NET_TEMPORAL_PROTOTXT := $(NET_TEMPORAL_ROOT)/deploy.prototxt

BEOID_SPATIAL_ROOT := $(BEOID_DATA_ROOT)/spatial
BEOID_TEMPORAL_ROOT := $(BEOID_DATA_ROOT)/temporal
BEOID_TEMPORAL_U_ROOT := $(BEOID_TEMPORAL_ROOT)/u
BEOID_TEMPORAL_V_ROOT := $(BEOID_TEMPORAL_ROOT)/v

BEOID_VIDEOS := $(notdir $(wildcard $(BEOID_SPATIAL_ROOT)/*))

GENERATED_SPATIAL_ROOT := $(GENERATED_DATA_ROOT)/spatial
GENERATED_SPATIAL_FRAMES_ROOT := $(GENERATED_SPATIAL_ROOT)/frames
GENERATED_SPATIAL_VIDEOS_ROOT := $(GENERATED_SPATIAL_ROOT)/videos
GENERATED_SPATIAL_EXCITATION_FRAMES := $(addprefix $(GENERATED_SPATIAL_FRAMES_ROOT)/,$(BEOID_VIDEOS))
GENERATED_SPATIAL_EXCITATION_VIDEOS := $(addprefix $(GENERATED_SPATIAL_VIDEOS_ROOT)/,$(addsuffix .mp4,$(BEOID_VIDEOS)))

GENERATED_TEMPORAL_ROOT := $(GENERATED_DATA_ROOT)/temporal
GENERATED_TEMPORAL_FRAMES_ROOT := $(GENERATED_TEMPORAL_ROOT)/frames
GENERATED_TEMPORAL_VIDEOS_ROOT := $(GENERATED_TEMPORAL_ROOT)/videos
GENERATED_TEMPORAL_EXCITATION_FRAMES := $(addprefix $(GENERATED_TEMPORAL_FRAMES_ROOT)/,$(BEOID_VIDEOS))
GENERATED_TEMPORAL_EXCITATION_VIDEOS := $(addprefix $(GENERATED_TEMPORAL_VIDEOS_ROOT)/,$(addsuffix .mp4,$(BEOID_VIDEOS)))

$(GENERATED_SPATIAL_FRAMES_ROOT)/%: $(BEOID_SPATIAL_ROOT)/%
	./scripts/generate_spatial_excitation_maps.py \
		$< \
		$@ \
		--caffemodel $(NET_SPATIAL_CAFFEMODEL) \
		--prototxt $(NET_SPATIAL_PROTOTXT)

$(GENERATED_TEMPORAL_FRAMES_ROOT)/%: $(BEOID_TEMPORAL_U_ROOT)/% $(BEOID_TEMPORAL_V_ROOT)/%
	./scripts/generate_temporal_excitation_maps.py \
		$< \
		$@ \
		--caffemodel $(NET_TEMPORAL_CAFFEMODEL) \
		--prototxt $(NET_TEMPORAL_PROTOTXT)


$(GENERATED_SPATIAL_VIDEOS_ROOT)/%.mp4: $(GENERATED_SPATIAL_FRAMES_ROOT)/%
	./scripts/stitch_video.py \
		$< \
		$@

$(GENERATED_TEMPORAL_VIDEOS_ROOT)/%.mp4: $(GENERATED_TEMPORAL_FRAMES_ROOT)/%
	./scripts/stitch_video.py \
		$< \
		$@

.PHONY: all
all: spatial_excitation_map_videos temporal_excitation_map_videos

.PHONY: spatial_excitation_map_videos 
spatial_excitation_map_videos: $(GENERATED_SPATIAL_EXCITATION_VIDEOS)

.PHONY: temporal_excitation_map_videos
temporal_excitation_map_videos: $(GENERATED_TEMPORAL_EXCITATION_VIDEOS)

