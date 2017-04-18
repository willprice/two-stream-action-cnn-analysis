import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from beoid.filename_parser import *
from beoid.gaze import *

from . import data

spatial_contrastive_beoid = pd.read_pickle(data.path_attention_map_spatial_contrastive_beoid)
spatial_non_contrastive_beoid = pd.read_pickle(data.path_attention_map_spatial_non_contrastive_beoid)
temporal_contrastive_beoid = pd.read_pickle(data.path_attention_map_temporal_contrastive_beoid)
temporal_non_contrastive_beoid = pd.read_pickle(data.path_attention_map_temporal_non_contrastive_beoid)

gaze = dict()
for filename in data.beoid_gaze_filenames:
    gaze[filename[:-len(data.gaze_file_suffix)]] = read_gaze(filename)