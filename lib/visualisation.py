import numpy as np
from scipy.misc import toimage, imresize

from matplotlib import pyplot as plt
from skimage import color, transform

def _normalise_array(array):
    return (array - np.min(array)) / np.ptp(array)

def show_filters(net, layer):
    weights = _weights_for_layer(net, layer)
    filter_count = weights.shape[0]
    width = int(np.ceil(np.sqrt(filter_count)))
    height = width

    filters = list(weights)

    fig, ax = plt.subplots(height, width, sharex=True, sharey=True)
    for row in range(0, height):
        for col in range(0, width):
            subplot = ax[row][col]
            subplot.axis('off')
            if (row*width + col > filter_count): break
            subplot.imshow(filters[row*width + col], interpolation='nearest')
    return fig

def _scale_shape(shape, scalar):
    return tuple(map(lambda x: x * scalar, shape))

class SCALING_METHODS:
    KERNEL_SLICE = "SCALING_METHOD=KERNEL_SLICE"
    KERNEL = "SCALING_METHOD=KERNEL"
    LAYER = "SCALING_METHOD=LAYER"

def _kernel_scaler(extrema, kernel_slice):
    (min, max) = extrema
    return (kernel_slice - min) / (max - min)

def show_grayscale_filters_from_weights(weights, 
        element_size=5, 
        kernel_spacer_width=1/3.0, 
        kernel_spacer_height=1/3.0, 
        scaling_method=SCALING_METHODS.KERNEL, 
        scaling_func=_kernel_scaler):

    if len(weights.shape) == 4:
        kernel_images = []

        kernel_shape = weights[0].shape
        kernel_depth = kernel_shape[0]
        kernel_height = kernel_shape[1]
        kernel_width = kernel_shape[2]

        intra_kernel_spacer_width = int(np.ceil(kernel_width * kernel_spacer_width))
        inter_kernel_spacer_height = int(np.ceil(kernel_height * kernel_spacer_height))
        for kernel in weights:
            # Kernel shape = (20, 3, 3), that's a stack of 20 3x3 kernels forming a volume
            # which is convolved over the input stack, we want to plot this stack.
            spacer = create_spacer(intra_kernel_spacer_width,
                                   kernel_height,
                                   value=1.0)
            kernel_range = kernel.ptp()
            kernel_image = []
            for kernel_slice in kernel:
                if (scaling_method == SCALING_METHODS.KERNEL_SLICE):
                    extrema = (kernel_slice.min(), kernel_slice.max())
                elif (scaling_method == SCALING_METHODS.KERNEL):
                    extrema = (kernel.min(), kernel.max())
                elif (scaling_method == SCALING_METHODS.LAYER):
                    extrema = (weights.min(), weights.max())
                else:
                    raise RuntimeError("Unknown scaling method: {}".format(scaling_method))
                kernel_image.append(scaling_func(extrema, kernel_slice))

            kernel_with_spacers = _intersperse(kernel_image, spacer)
            kernel_row = np.concatenate(kernel_with_spacers, 1)
            kernel_images.append(kernel_row)

        spacer = create_spacer(kernel_depth * kernel_width + (kernel_depth - 1) * intra_kernel_spacer_width,
                              inter_kernel_spacer_height,
                              value=1.0)
        kernel_images_with_spacers = _intersperse(kernel_images, spacer)
        weight_vis = np.concatenate(kernel_images_with_spacers)
        image_shape = (weight_vis.shape[0] * element_size,
                       weight_vis.shape[1] * element_size)
        return imresize(weight_vis, image_shape, interp='nearest')
    if len(weights.shape) == 2:
        return toimage(weights)
    else:
        raise RuntimeException("Don't know how to visualise a layer of shape {}".format(weights.shape))



def show_grayscale_filters(net, layer, **kwargs):
    """
    element_size: For each element inside a filter, what is the width (and also height) in pixels
    kernel_spacer_width: Proportion of kernel width for spacer width (ceiled)
    kernel_spacer_height: Proportion of kernel height for spacer height (ceiled)

    Show filters for a layer that doesn't operate on RGB data (e.g. first convolutional layer of a temporal tower)
    """
    weights = _weights_for_layer(net, layer)
    return show_grayscale_filters_from_weights(weights, **kwargs)

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
            if (row*width + col > filter_count): break
            subplot.imshow(filter_responses[row*width + col], interpolation='nearest')
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


def create_spacer(width, height, value=255.0):
    """
    Generate a np.array with shape (width, height)
    """
    return np.ones((height, width), dtype=np.float64) * value


def _params_for_layer(net, layer):
    return net.params[layer]


def _weights_for_layer(net, layer):
    return _params_for_layer(net, layer)[0].data


def _biases_for_layer(net, layer):
    return _params_for_layer(net, layer)[1].data


def _intersperse(seq, spacer):
    """
    Return a list [seq[0], spacer, seq[1], spacer, ... , seq[n]]
    """
    interspersed = []
    for element in seq:
        interspersed.append(element)
        interspersed.append(spacer)
    return interspersed[:-1]
