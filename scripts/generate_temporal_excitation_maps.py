#!/usr/bin/env python3
import importlib
import logging
import os
import pandas as pd
import caffe
import click
from scipy.misc import toimage
from skimage import color

import visualisation
from dataset import ActionRecognitionDataSet
from two_stream_excitation_backprop import generate_temporal_excitation_maps_for_dataset

logger = logging.getLogger(__name__)



@click.command()
@click.argument('data-root',
                type=click.Path(exists=True))
@click.argument('excitation-maps-pickle',
                type=click.Path(exists=True))
@click.argument('root-output-dir',
                type=click.Path(exists=False))
@click.option('--overlay-offset',
              help='Index of frame from start of flow batch to overlay',
              default=5)
@click.option('--desaturate/--no-desaturate',
              default=True,
              help='Desaturate underlay image before overlaying excitation map')
@click.option('--colormap',
              default='hot',
              help='matplotlib color map to colorize excitation map')
def draw_temporal_excitation_maps(data_root,
                                  excitation_maps_pickle,
                                  root_output_dir,
                                  overlay_offset,
                                  desaturate, colormap
                                  ):
    """
    Create an image from each optical flow frame pair (u, v) by forward
    propagating a batch of (u,v) pairs and then running excitation
    backprop from the LABEL neuron in the last layer.
    """
    dataset = ActionRecognitionDataSet(data_root)
    # attention_maps.append({
    #     "video": video_name,
    #     "starting_frame":  starting_frame_index + 1,
    #     "type": "temporal",
    #     "attention_map": attention_map,
    #     "contrastive": contrastive
    # })
    batch_size = 10

    numerical_excitation_maps = pd.read_pickle(excitation_maps_pickle)

    def attention_map_callback(video_name, starting_frame_index, attention_map):
        underlay = caffe.io.load_image(dataset.get_frame(video_name,
        starting_frame_index + 1 + overlay_offset))
        if desaturate:
            underlay = color.rgb2gray(underlay)
        attention_map_overlaid_image = toimage(
            visualisation.overlay_attention_map(underlay, attention_map, cmap=colormap)
        )
        output_dir = os.path.join(root_output_dir, video_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        attention_map_overlaid_image.save(
            os.path.join(output_dir,
                         "frame{:06d}-{:06d}.jpg".format(starting_frame_index + 1,
                                                         starting_frame_index + batch_size + 1)))

    numerical_excitation_maps.apply(
        lambda row: attention_map_callback(row['video'], row['starting_frame'] - 1, row['attention_map']), axis=1)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    draw_temporal_excitation_maps()
