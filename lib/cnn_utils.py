import caffe
import numpy as np
from skimage import transform, color
import matplotlib.pyplot as plt


def imagenet_transformer(net, input_blob='data'):
    transformer = caffe.io.Transformer(
        {'data': net.blobs[input_blob].data.shape}
    )
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return transformer

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


def ucf101_spatial_transformer(net, input_blob='data'):
    """
    Untested
    :param net:
    :param input_blob:
    :return:
    """
    transformer = caffe.io.Transformer(
        {'data': net.blobs[input_blob].data.shape}
    )
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_raw_scale('data', 255.0)
    return transformer

def get_layer_output_shapes(net):
    return _get_layer_shapes(net, net.top_names)


def get_layer_input_shapes(net):
    return _get_layer_shapes(net, net.bottom_names)


def _get_layer_shapes(net, blob_names):
    layer_shapes = dict()
    for layer_name in blob_names.keys():
        input_blob_names = blob_names[layer_name]
        input_blobs = [net.blobs[name] for name in input_blob_names]
        input_shapes = [tuple(input_blob.shape) for input_blob in input_blobs]
        layer_shapes[layer_name] = input_shapes
    return layer_shapes


def print_layer_shapes(net, direction='output'):
    if direction == "output":
        layer_shapes = get_layer_output_shapes(net)
    elif direction == "input":
        layer_shapes = get_layer_input_shapes(net)
    else:
        raise RuntimeError("Expected 'input' or 'output' for `direction` arg.")
    print("Layer shapes ({})\n".format(direction))
    for layer in net.top_names.keys():
        print("{!s:<10}: {!s:<10}".format(layer, layer_shapes[layer]))


def show_filters_responses(blob):
    filter_count = blob.shape[1]
    width = int(np.ceil(np.sqrt(filter_count)))
    height = width

    filter_responses = list(blob.data[0])

    fig, ax = plt.subplots(height, width, sharex=True, sharey=True)
    for row in range(0, height):
        for col in range(0, width):
            subplot = ax[row][col]
            subplot.axis('off')
            if (row*col + col > filter_count): break
            subplot.imshow(filter_responses[row*col + col], interpolation='nearest')
    return fig

def overlay_attention_map(image, attention_map, cmap='hot', alpha=0.8):
    """
    Colors the single channel attention map which is then overlaid on the
    image
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = color.gray2rgb(image)

    attention_map -= attention_map.min()
    if attention_map.max() > 0:
        attention_map /= attention_map.max()
    attention_map = transform.resize(attention_map, image.shape[:2], order=3)

    cmap = plt.get_cmap(cmap)
    attention_heatmap = np.delete(cmap(attention_map), 3, 2)
    image_masked_with_heatmap = (1 - attention_map**alpha).reshape(attention_map.shape + (1,))*image
    scaled_heatmap = (attention_map**alpha).reshape(attention_map.shape+(1,))*attention_heatmap
    heatmap_overlaid_image = image_masked_with_heatmap + scaled_heatmap
    return heatmap_overlaid_image

class ExcitationBackprop(object):
    """
    First run ``prop`` with the image, then calculate the excitation backprop
    with ``backprop`` of an individual neuron
    """

    def __init__(self, net,
                 top_layer_name, second_top_layer_name, output_layer_name,
                 top_blob_name=None, second_top_blob_name=None,
                 output_blob_name=None):
        self.net = net

        self.top_layer_name = top_layer_name
        if top_blob_name:
            self.top_blob_name = top_blob_name
        else:
            self.top_blob_name = self._get_top_blob_name_of_layer(self.top_layer_name)

        self.second_top_layer_name = second_top_layer_name
        if second_top_blob_name:
            self.second_top_blob_name = second_top_blob_name
        else:
            self.second_top_blob_name = self._get_top_blob_name_of_layer(self.second_top_layer_name)

        self.output_layer_name = output_layer_name
        if output_blob_name:
            self.output_blob_name = output_blob_name
        else:
            self.output_blob_name = self._get_top_blob_name_of_layer(self.output_layer_name)


    def _get_top_blob_name_of_layer(self, layer):
        return self.net.top_names[layer][0]

    def _get_second_top_layer_name(self):
        layer_names = list(self.net.top_names.keys())
        for i, layer_name in enumerate(layer_names):
            if layer_name == self.top_layer_name:
                return layer_names[i - 1]
        raise RuntimeError("Could not find second top layer name")

    def _set_input(self, image):
        self.net.blobs['data'].data[...] = image

    def prop(self, image):
        self._set_input(image)
        self.net.forward(end=self.top_layer_name)

    def backprop(self, neuron_id, contrastive=True):
        # eb = excitation backprop
        caffe.set_mode_eb_gpu()

        net = self.net

        net.blobs[self.top_blob_name].diff[0][...] = 0
        net.blobs[self.top_blob_name].diff[0][neuron_id] = \
            np.exp(net.blobs[self.top_blob_name].data[0][neuron_id].copy())
        net.blobs[self.top_blob_name].diff[0][neuron_id] /= \
            net.blobs[self.top_blob_name].diff[0][neuron_id].sum()

        if contrastive:
            # invert the top layer weights
            net.params[self.top_layer_name][0].data[...] *= -1
            net.backward(start=self.top_layer_name, end=self.second_top_layer_name)
            # Grab the signal when all uninteresting neurons are set
            buff = net.blobs[self.second_top_blob_name].diff.copy()

            # restore layer: make all uninteresting neurons uninteresting again
            net.params[self.top_layer_name][0].data[...] *= -1
            # Grab the signal when only the interesting neuron is set
            net.backward(start=self.top_layer_name, end=self.second_top_layer_name)
            # Combine results of inverted and non inverted backprop to compute contrastive signal
            net.blobs[self.second_top_blob_name].diff[...] -= buff

            # compute the contrastive signal
            net.backward(start=self.second_top_layer_name, end=self.output_layer_name)
        else:
            net.backward(start=self.top_layer_name, end=self.output_layer_name)

        attention_map = np.maximum(
            net.blobs[self.output_blob_name].diff[0].sum(0),
            0
        )

        return attention_map

