import caffe
import numpy as np


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
        self.net.clear_param_diffs()
        self.net.blobs['data'].data[...] = image

    def prop(self, image):
        self._set_input(image)
        self.net.forward(end=self.top_layer_name)

    def backprop(self, neuron_id, contrastive=True):
        # eb = excitation backprop
        caffe.set_mode_eb_gpu()

        net = self.net

        top = net.blobs[self.top_blob_name].data[0][neuron_id].copy()
        if not np.any(top):
            print("WARNING: top data is empty")
        top = np.exp(1.)
        net.blobs[self.top_blob_name].diff[0][...] = 0
        net.blobs[self.top_blob_name].diff[0][neuron_id] = top
            #np.exp(net.blobs[self.top_blob_name].data[0][neuron_id].copy())
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
