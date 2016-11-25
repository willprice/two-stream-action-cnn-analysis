#!/usr/bin/env python3

import sys
import os
import click
import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from pypeg2 import parse

caffe_root = '../'
sys.path.insert(0, caffe_root + "python")
import caffe

import cnn_utils
import beoid
from beoid.filename_parser import ActionVideo

logger = logging.getLogger()



caffe.set_mode_gpu()


@click.command()
@click.option('--base-path', default='/home/will/data/beoid/spatial', help="Root of the frame directories")
@click.option('--video', default='04_Printer2_push_drawer_241-293', help='Video name to generate video from')
@click.option('--label', default='label', help='Label you wish to generate excitation maps from')
@click.option('--output-path', help='Directory in which to save the output')
def generate_video(base_path, video, label, output_path):
    if label == 'label':
        parsed_filename = parse(video, ActionVideo)
        label = parsed_filename.action.name
        label += "_" + parsed_filename.objects[0]
        for object in parsed_filename.objects[1:]:
            label += "+{}" + object.name
    label_id = int(beoid.get_label_id(label))
    if not output_path:
        output_path = video + "excitation-maps"

    model_root_path = "/home/will/nets/dual-stream/spatial"
    model_name = "kfold1borders25"

    deploy_prototxt_path = os.path.join(model_root_path, "deploy.prototxt")
    caffemodel_path = os.path.join(model_root_path, model_name + ".caffemodel")

    net = caffe.Net(deploy_prototxt_path,
                    caffemodel_path,
                    caffe.TRAIN)
    eb = cnn_utils.ExcitationBackprop(net, 'fc8', 'fc7', 'pool3')

    transformer = cnn_utils.imagenet_transformer(net)

    new_size = (224, 224)
    net.blobs['data'].reshape(1, 3, new_size[0], new_size[1])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    frames_root_path = os.path.join(base_path, video)
    for frame_filename in os.listdir(frames_root_path):
        image_path = os.path.join(frames_root_path, frame_filename)
        print("Generating attention map for '{}'".format(image_path))
        image = caffe.io.load_image(image_path)
        preprocessed_image = transformer.preprocess('data', image)
        eb.prop(preprocessed_image)
        attention_map = eb.backprop(label_id)

        fig = plt.figure()
        attention_map -= attention_map.min()
        if attention_map.max() > 0:
            attention_map /= attention_map.max()
        attention_map = transform.resize(attention_map, (image.shape[:2]), order = 3, mode = 'nearest')
#        if blur:
#            attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
#            attMap -= attMap.min()
#            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attention_map)
        attMapV = np.delete(attMapV, 3, 2)
        attention_map = 1*(1-attention_map**0.8).reshape(attention_map.shape + (1,))*image + (attention_map**0.8).reshape(attention_map.shape+(1,)) * attMapV;
        plt.imshow(attention_map, interpolation = 'bicubic')
        fig.savefig(os.path.join(output_path, frame_filename))
        plt.close(fig)



if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    generate_video()
