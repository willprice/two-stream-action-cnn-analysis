#!/usr/bin/env python3

from prelude import *
import os
import click
import logging
from pypeg2 import parse
from scipy.misc import toimage
import caffe

import cnn_utils
import beoid
from beoid.filename_parser import ActionVideo

logger = logging.getLogger()



caffe.set_mode_gpu()


@click.command()
@click.argument('frame-source-dir',
                type=click.Path(exists=True))
@click.argument('output-dir',
                type=click.Path(exists=False))
@click.option('--label',
              help='Label you wish to generate excitation maps from')
@click.option('--caffemodel',
              default="/home/will/nets/dual-stream/spatial/kfold1borders25.caffemodel",
              type=click.Path(exists=True),
              help='convnet caffemodel')
@click.option('--prototxt',
              default="/home/will/nets/dual-stream/spatial/deploy.prototxt",
              type=click.Path(exists=True),
              help='convnet caffemodel')
def generate_spatial_excitation_maps(frame_source_dir, output_dir, label,
                                     caffemodel, prototxt):
    """
    Create an image from each frame in FRAME_SOURCE_DIR by running excitation
    backprop from the LABEL neuron in the last layer.
    """
    video_name = os.path.basename(os.path.normpath(frame_source_dir))
    if not label:
        parsed_filename = parse(video_name, ActionVideo)
        label = parsed_filename.action.name
        label += "_" + parsed_filename.objects[0]
        for object in parsed_filename.objects[1:]:
            label += "+{}" + object.name
    label_id = int(beoid.get_label_id(label))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    net = caffe.Net(prototxt,
                    caffemodel,
                    caffe.TRAIN)
    eb = cnn_utils.ExcitationBackprop(net, 'fc8', 'fc7', 'pool3')

    transformer = cnn_utils.imagenet_transformer(net)

    new_size = (224, 224)
    net.blobs['data'].reshape(1, 3, new_size[0], new_size[1])

    for frame_filename in os.listdir(frame_source_dir):
        image_path = os.path.join(frame_source_dir, frame_filename)
        print("Generating attention map for '{}'".format(image_path))
        image = caffe.io.load_image(image_path)
        preprocessed_image = transformer.preprocess('data', image)
        eb.prop(preprocessed_image)
        attention_map = eb.backprop(label_id)
        attention_map_overlaid_image = toimage(cnn_utils.overlay_attention_map(image, attention_map))
        attention_map_overlaid_image.save(os.path.join(output_dir, frame_filename))


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    generate_spatial_excitation_maps()
