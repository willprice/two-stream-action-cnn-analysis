import os

import caffe
import numpy as np
import logging

import excitation_backprop

logger = logging.getLogger(__name__)


def generate_temporal_excitation_maps_for_dataset(net, net_config, dataset, attention_map_callback, batch_size=10, contrastive=True):
    for video_name in dataset.get_videos():
        logger.info("Generating temporal excitation maps for video {}".format(video_name))
        generate_temporal_excitation_maps(net, net_config, dataset, video_name, attention_map_callback,
                                          contrastive, batch_size)


def generate_temporal_excitation_maps(net, net_config, dataset, video_name, attention_map_callback,
                                      contrastive, batch_size):
    transformer = net_config.transformer(net)

    def load_and_transform(image_path):
        image = caffe.io.load_image(image_path, color=False)
        return transformer.preprocess('data', image)

    eb = excitation_backprop.ExcitationBackprop(net, net_config.top_blob, net_config.second_top_blob,
                                                net_config.bottom_blob)

    label_id = net_config.labeller(video_name)
    flow_batches_paths = dataset.get_flow_batches(video_name, batch_size=batch_size)

    for starting_frame_index, flow_batch_paths in enumerate(flow_batches_paths):
        flow_batch = np.array([load_and_transform(path) for path in flow_batch_paths])
        flow_batch = flow_batch.reshape(1, 20, 224, 224)
        logger.info("Generating temporal excitation maps for frame {}-{}".format(
            starting_frame_index, starting_frame_index + 20
        ))
        eb.prop(flow_batch)
        attention_map = eb.backprop(label_id, contrastive=contrastive)
        if not np.any(attention_map):
            logger.warning(
                "Temporal attention map is all zeroes for video: {}, with batch size: {}, starting at index {}".format(
                    video_name, batch_size, starting_frame_index
                ))
        attention_map_callback(video_name, starting_frame_index, attention_map)


def generate_spatial_excitation_maps_for_dataset(net, net_config, dataset, attention_map_callback, contrastive=True):
    transformer = net_config.transformer(net)

    for video_name in dataset.get_videos():
        logger.info("Generating spatial excitation maps for video {}".format(video_name))
        generate_spatial_excitation_map(net, net_config, dataset, transformer, video_name, attention_map_callback, contrastive)


def generate_spatial_excitation_map(net, net_config, dataset, transformer, video_name, attention_map_callback, contrastive=True):
    eb = excitation_backprop.ExcitationBackprop(net, net_config.top_blob, net_config.second_top_blob,
                                                net_config.bottom_blob)
    label_id = net_config.labeller(video_name)
    logger.info("Generating spatial excitation maps for neuron id: {}".format(label_id))
    for frame_path in dataset.get_frames(video_name):
        logger.info("Generating spatial excitation map from '{}'".format(frame_path))
        image = caffe.io.load_image(frame_path)
        preprocessed_image = transformer.preprocess('data', image)
        eb.prop(preprocessed_image)
        attention_map = eb.backprop(label_id, contrastive=contrastive)
        if not np.any(attention_map):
            logger.warning("Attention map is all zeroes")
        attention_map_callback(video_name, os.path.basename(frame_path), image, attention_map)