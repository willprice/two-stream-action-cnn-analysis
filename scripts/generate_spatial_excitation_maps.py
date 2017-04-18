#!/usr/bin/env python3

import importlib
import logging
import os

import caffe
import pandas as pd
import click
from scipy.misc import toimage
from skimage.color import rgb2gray, gray2rgb

import visualisation
from dataset import ActionRecognitionDataSet
from two_stream_excitation_backprop import generate_spatial_excitation_maps_for_dataset

logger = logging.getLogger()


@click.command()
@click.argument('data-root',
                type=click.Path(exists=True))
@click.argument('excitation-maps-pickle',
                type=click.Path(exists=True))
@click.argument('root-output-dir',
                type=click.Path(exists=False))
@click.option('--desaturate/--no-desaturate',
              default=False,
              help='Desaturate the underlay before overlaying excitation map')
@click.option('--colormap',
              default='hot',
              help='matplotlib color map used to color attention map')
def generate_spatial_excitation_maps(data_root,
                                     excitation_maps_pickle,
                                     root_output_dir,
                                     desaturate, colormap):
    """
    Overlay the excitation maps from the pickle onto the source frame
    """

    dataset = ActionRecognitionDataSet(data_root)
    excitation_maps = pd.read_pickle(excitation_maps_pickle)

    def attention_map_callback(video_name, frame, attention_map):
        underlay = caffe.io.load_image(dataset.get_frame(video_name, frame))
        if desaturate:
            underlay = gray2rgb(rgb2gray(underlay))
        attention_map_overlaid_image = toimage(visualisation.overlay_attention_map(underlay, attention_map, cmap=colormap))

        output_dir = os.path.join(root_output_dir, video_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        attention_map_overlaid_image.save(os.path.join(output_dir, os.path.join(
            output_dir, "frame{:06d}.jpg".format(frame)
        )))

    excitation_maps.apply(
        lambda row: attention_map_callback(row['video'], row['frame'], row['attention_map']), axis=1
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_spatial_excitation_maps()
