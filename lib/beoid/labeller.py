import pandas as pd
from pypeg2 import parse
from . import filename_parser

import os

_beoid_dir = os.path.dirname(os.path.abspath(__file__))
label_to_id = pd.read_csv(os.path.join(_beoid_dir, 'class-labels.csv'))
label_count = len(label_to_id)


def labeller(filename):
    """
    Get the label ID from the filename of a BEOID video.

    :param filename: The filename of the video being processed (e.g. 00_Desk2_pick-up_plug_334-366)
    :return: id: The neuron ID in the final layer representing the class specified by the filename
    """
    parsed_filename = parse(filename, filename_parser.ActionVideo)
    label = parsed_filename.action.name
    label += "_" + parsed_filename.objects[0]
    for object in parsed_filename.objects[1:]:
        label += "+{}" + object.name
    return get_label_id(label)

def get_label_id(label):
    row = label_to_id.loc[label_to_id.label == label]
    return int(row.id)
