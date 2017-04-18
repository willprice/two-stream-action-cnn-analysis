import caffe
import numpy as np


def flow_transformer(net, input_blob='data'):
    net_input_shape = net.blobs[input_blob].data.shape
    height = net_input_shape[2]
    width = net_input_shape[3]
    transformer = caffe.io.Transformer(
        {'data': (1, 1, height, width)}
    )
    transformer.set_mean('data', np.array([128]))
    transformer.set_raw_scale('data', 255.0)
    return transformer


def imagenet_transformer(net, input_blob='data'):
    transformer = caffe.io.Transformer(
        {'data': net.blobs[input_blob].data.shape}
    )
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return transformer