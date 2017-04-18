import os

_script_path = os.path.realpath(__file__)
_script_dir = os.path.dirname(_script_path)
_beoid_gaze_root = os.path.join(_script_dir, "..",
                                "data/beoid-download/Bristol_Egocentric_Object_Interactions_Dataset_2014",
                                "TrackedGaze_2DAnd3D"
)
_beoid_root = os.path.join(_script_dir, "..", "generated/beoid/test-1")
_ucf101_root = os.path.join(_script_dir, "..", "generated/ucf101/test-1")

_spatial_contrastive_name = 'spatial_contrastive_excitation_maps.pickle'
_spatial_non_contrastive_name = 'spatial_non_contrastive_excitation_maps.pickle'
_temporal_contrastive_name = 'temporal_contrastive_excitation_maps.pickle'
_temporal_non_contrastive_name = 'temporal_non_contrastive_excitation_maps.pickle'


path_attention_map_spatial_contrastive_beoid = os.path.join(_beoid_root, _spatial_contrastive_name)
path_attention_map_spatial_non_contrastive_beoid = os.path.join(_beoid_root, _spatial_non_contrastive_name)
path_attention_map_temporal_contrastive_beoid = os.path.join(_beoid_root, _temporal_contrastive_name)
path_attention_map_temporal_non_contrastive_beoid = os.path.join(_beoid_root, _temporal_non_contrastive_name)

path_attention_map_spatial_contrastive_ucf101 = os.path.join(_ucf101_root, _spatial_contrastive_name)
path_attention_map_spatial_non_contrastive_ucf101 = os.path.join(_ucf101_root, _spatial_non_contrastive_name)
path_attention_map_temporal_contrastive_ucf101 = os.path.join(_ucf101_root, _temporal_contrastive_name)
path_attention_map_temporal_non_contrastive_ucf101 = os.path.join(_ucf101_root, _temporal_non_contrastive_name)

gaze_file_suffix = "_3DGazeData.txt"
beoid_gaze_filenames = [os.path.join(_beoid_gaze_root, filename) for
                        filename in os.listdir(_beoid_gaze_root)
                        if filename.endswith(gaze_file_suffix)]


