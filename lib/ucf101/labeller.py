import pandas as pd
import os
from . import filename_parser

_ucf101_dir = os.path.dirname(os.path.abspath(__file__))
label_to_id = pd.read_csv(os.path.join(_ucf101_dir, 'class-labels.csv'))
label_count = len(label_to_id)

def labeller(filename):
    """
    Get the label ID from the filename of a BEOID video.

    :param filename: The filename of the video being processed (e.g. v_WritingOnBoard_g04_c04.avi)
    :return: id: The neuron ID in the final layer representing the class specified by the filename
    """
    action = filename_parser.ucf101_filename_parser(filename).action
    return get_label_id(action)

def get_label_id(label):
    row = label_to_id.loc[label_to_id.label == label]
    return int(row.id)
