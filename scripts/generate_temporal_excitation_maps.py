#!/usr/bin/env python3

from prelude import *
import os
import re
import click
import logging
import importlib
from pypeg2 import parse
from scipy.misc import toimage
import caffe
import numpy as np
from skimage import transform, color

import cnn_utils

logger = logging.getLogger()



caffe.set_mode_gpu()


@click.command()
@click.argument('net-config-module',
                type=click.STRING)
@click.argument('optical-flow-u-dataset-root',
                type=click.Path(exists=True))
@click.argument('optical-flow-v-dataset-root',
                type=click.Path(exists=True))
@click.argument('spatial-root',
                type=click.Path(exists=True))
@click.argument('video-name')
@click.argument('output-dir',
                type=click.Path(exists=False))
@click.option('--label',
              help='Label you wish to generate excitation maps from')
@click.option('--to-layer',
              help='Layer to exictation backprop to',
              default='pool3')
@click.option('--overlay-offset',
              help='Index of frame from start of flow batch to overlay',
              default=10)
@click.option('--contrastive/--no-contrastive',
              default=True,
              help='Use contrastive excitation backprop, or just excitation backprop')
@click.option('--desaturate/--no-desaturate',
              default=False,
              help='Desaturate underlay image before overlaying excitation map')
@click.option('--colormap',
              default='hot',
              help='matplotlib color map to colorize excitation map')
def generate_spatial_excitation_maps(net_config_module,
                                     optical_flow_u_dataset_root,
                                     optical_flow_v_dataset_root,
                                     spatial_root, video_name,
                                     output_dir, label, to_layer,
                                     overlay_offset,
                                     contrastive,
                                     desaturate, colormap
                                     ):
    """
    Create an image from each optical flow frame pair (u, v) by forward
    propagating a batch of (u,v) pairs and then running excitation
    backprop from the LABEL neuron in the last layer.
    """
    #if not label:
    #    parsed_filename = parse(video_name, ActionVideo)
    #    label = parsed_filename.action.name
    #    label += "_" + parsed_filename.objects[0]
    #    for object in parsed_filename.objects[1:]:
    #        label += "+{}" + object.name
    #label_id = int(beoid.get_label_id(label))
    net_config = importlib.import_module(net_config_module)
    label_id = net_config.labeller(video_name)
    optical_flow_u_root = os.path.join(optical_flow_u_dataset_root, video_name)
    optical_flow_v_root = os.path.join(optical_flow_v_dataset_root, video_name)

    if not os.path.exists(optical_flow_u_root):
        raise RuntimeError("Optical flow U directory for {} does not exist".format(optical_flow_u_root))

    if not os.path.exists(optical_flow_v_root):
        raise RuntimeException("Optical flow V directory for {} does not\
                               exist".format(optical_flow_v_root))

    if not os.path.exists(os.path.join(optical_flow_u_root, "frame000010.jpg")):
        print("This source directory has less than 10 frames, can't generate video")
        return 1


    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    net = caffe.Net(net_config.net_prototxt_path,
                    net_config.net_caffemodel_path,
                    caffe.TEST)
    eb = cnn_utils.ExcitationBackprop(net, net_config.top_blob, net_config.second_top_blob, net_config.bottom_blob)
    spatial_video_root = os.path.join(spatial_root, video_name)

    transformer = net_config.transformer(net)
    frame_batches = np.array(
        get_frame_batches(optical_flow_u_root, optical_flow_v_root, video_name, transformer))
    for starting_frame_index, frame_batch in enumerate(frame_batches):
        frame_batch = frame_batch.reshape(1, 20, 224, 224)
        print("Generating excitation maps for frame {}-{}".format(starting_frame_index, starting_frame_index + 20))
        # mean normalisation
        #frame_batch_average = np.mean(frame_batch, axis=1)[0]
        ##toimage(frame_batch_average).save(os.path.join(output_dir,
        ##                 "average_frame{:06d}-{:06d}.jpg".format(starting_frame_index + 1,
        ##                                                 starting_frame_index + 11)))
        # end mean normalisation
        eb.prop(frame_batch)
        attention_map = eb.backprop(label_id, contrastive=contrastive)
        if not np.any(attention_map):
            print("WARNING: Attention map is all zeroes")
        underlay = caffe.io.load_image(
            os.path.join(spatial_video_root,
                         net_config.frame_pattern.format(int(starting_frame_index
                                                             +
                                                             overlay_offset
                                                             +
                                                             1))))
        if desaturate:
            underlay = color.rgb2gray(underlay)
        attention_map_overlaid_image = toimage(
            cnn_utils.overlay_attention_map(underlay, attention_map,
                                            cmap=colormap)
        )
        attention_map_overlaid_image.save(
            os.path.join(output_dir,
                         "frame{:06d}-{:06d}.jpg".format(starting_frame_index + 1,
                                                         starting_frame_index + 11)))

def get_frame_batches(optical_flow_u_root, optical_flow_v_root, video_name, transformer, batch_size=10):
    frame_count = get_frame_count(optical_flow_u_root)
    batches = []
    for batch_index in range(0, frame_count - batch_size):
        batch = get_frame_batch(optical_flow_u_root, optical_flow_v_root, transformer, start_frame=batch_index,
                                batch_size=batch_size)
        batches.append(batch)
    return batches

def get_frame_count(optical_flow_u_root):
    frame_files = os.listdir(optical_flow_u_root)
    return len(frame_files)

def get_frame_batch(optical_flow_u_root, optical_flow_v_root, transformer, start_frame=0, batch_size=10):
    """
    u and v represent the different directions of optical flow
    :param transformer:
    """
    u_frames = list(os.listdir(optical_flow_u_root))
    v_frames = list(os.listdir(optical_flow_v_root))

    u_frames = sorted(u_frames, key=frame_number)
    v_frames = sorted(v_frames, key=frame_number)

    u_frames = [os.path.join(optical_flow_u_root, file) for file in  u_frames]
    v_frames = [os.path.join(optical_flow_v_root, file) for file in  v_frames]

    last_frame = int(start_frame + batch_size)
    selected_u_frames = u_frames[start_frame:last_frame]
    selected_v_frames = v_frames[start_frame:last_frame]

    frames = []
    for i in range(len(selected_u_frames)):
        frames.append(selected_u_frames[i])
        frames.append(selected_v_frames[i])

    raw_frames = np.array([caffe.io.load_image(frame, color=False) for frame in frames])
    preprocessed_frames = np.array([transformer.preprocess('data', raw_frame) for raw_frame in raw_frames])
    return preprocessed_frames

def frame_number(frame_filename):
    """
    Parses frame of format frame%06d.jpg to get the number from the
    filename
    """
    pattern = re.compile(r'.*?(\d+)')
    matches = pattern.match(frame_filename)
    return int(matches.groups()[0])

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    generate_spatial_excitation_maps()
