#!/usr/bin/env python3

import importlib
import logging
import os

import caffe
import click
import pandas as pd
import re

from dataset import ActionRecognitionDataSet
from two_stream_excitation_backprop import generate_spatial_excitation_maps_for_dataset


@click.command()
@click.argument('net-config-module',
              type=click.STRING)
@click.argument('data-root',
                type=click.Path(exists=True))
@click.argument('output-file',
                type=click.Path(exists=False))
@click.option('--contrastive/--no-contrastive',
              default=True,
              help='Use contrastive excitation backprop, or just excitation backprop')
def generate_spatial_excitation_maps(net_config_module, data_root,
                                     output_file, contrastive):
    net_config = importlib.import_module(net_config_module)

    net = caffe.Net(net_config.net_prototxt_path,
                    net_config.net_caffemodel_path,
                    caffe.TRAIN)

    new_size = (224, 224)
    net.blobs['data'].reshape(1, 3, new_size[0], new_size[1])

    dataset = ActionRecognitionDataSet(data_root)


    attention_maps = []

    frame_number_re = re.compile(r'.*?(\d+)')
    def get_frame_number(frame_filename):
        matches = frame_number_re.match(frame_filename)
        # group 0 is the full regexp match, we want the first subgroup
        return int(matches.group(1))

    def attention_map_callback(video_name, image_filename, image, attention_map):
        attention_maps.append({
            "video": video_name,
            "frame": get_frame_number(image_filename),
            "type": "spatial",
            "attention_map": attention_map,
            "contrastive": contrastive
        })

    generate_spatial_excitation_maps_for_dataset(net, net_config, dataset, attention_map_callback, contrastive)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(attention_maps).to_pickle(output_file)


if __name__ == '__main__':
    caffe.set_mode_gpu()
    logging.basicConfig(level=logging.INFO)
    generate_spatial_excitation_maps()
