#!/usr/bin/env python3

from prelude import *
import os
import click
import logging
import importlib
from pypeg2 import parse
from scipy.misc import toimage
from skimage import transform
import numpy as np
import caffe

import cnn_utils

logger = logging.getLogger()



caffe.set_mode_gpu()


@click.command()
@click.argument('net-config-module',
              type=click.STRING)
@click.argument('frame-source-dir',
                type=click.Path(exists=True))
@click.argument('output-dir',
                type=click.Path(exists=False))
@click.option('--label',
              help='Label you wish to generate excitation maps from')
def generate_spatial_excitation_maps(net_config_module, frame_source_dir, output_dir, label):
    """
    Create an image from each frame in FRAME_SOURCE_DIR by running excitation
    backprop from the LABEL neuron in the last layer.
    """
    net_config = importlib.import_module(net_config_module)
    video_name = os.path.basename(os.path.normpath(frame_source_dir))
    label_id = net_config.labeller(video_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    net = caffe.Net(net_config.net_prototxt_path,
                    net_config.net_caffemodel_path,
                    caffe.TRAIN)
    eb = cnn_utils.ExcitationBackprop(net, net_config.top_blob, net_config.second_top_blob, net_config.bottom_blob)

    transformer = cnn_utils.imagenet_transformer(net)

    new_size = (224, 224)
    net.blobs['data'].reshape(1, 3, new_size[0], new_size[1])

    print("Generating excitation maps for neuron id: {}".format(label_id))
    for frame_filename in os.listdir(frame_source_dir):
        image_path = os.path.join(frame_source_dir, frame_filename)
        print("Generating attention map for '{}'".format(image_path))
        image = caffe.io.load_image(image_path)
        preprocessed_image = transformer.preprocess('data', image)
        eb.prop(preprocessed_image)
        attention_map = eb.backprop(label_id)
        if not np.any(attention_map):
            print("WARNING: Attention map is all zeroes")
        attention_map_overlaid_image = toimage(cnn_utils.overlay_attention_map(image, attention_map))
        attention_map_overlaid_image.save(os.path.join(output_dir, frame_filename))


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    generate_spatial_excitation_maps()
