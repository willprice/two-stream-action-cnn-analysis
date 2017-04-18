#!/usr/bin/env python3
import importlib
import logging
import os

import caffe
import click
import pandas as pd

from dataset import ActionRecognitionDataSet
from two_stream_excitation_backprop import generate_temporal_excitation_maps_for_dataset




caffe.set_mode_gpu()


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
def generate_temporal_excitation_maps(net_config_module,
                                      data_root,
                                      output_file,
                                      contrastive):
    net_config = importlib.import_module(net_config_module)
    dataset = ActionRecognitionDataSet(data_root)

    batch_size = 10

    net = caffe.Net(net_config.net_prototxt_path,
                    net_config.net_caffemodel_path,
                    caffe.TEST)

    attention_maps = []

    def attention_map_callback(video_name, starting_frame_index, attention_map):
        attention_maps.append({
            "video": video_name,
            "starting_frame":  starting_frame_index + 1,
            "type": "temporal",
            "attention_map": attention_map,
            "contrastive": contrastive
        })
    generate_temporal_excitation_maps_for_dataset(net, net_config, dataset, attention_map_callback, batch_size, contrastive)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(attention_maps).to_pickle(output_file)



if __name__ == '__main__':
    caffe.set_mode_gpu()
    logging.basicConfig(level=logging.INFO)
    generate_temporal_excitation_maps()
