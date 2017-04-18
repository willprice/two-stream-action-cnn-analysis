import numpy as np
from scipy.misc import toimage, imresize

from matplotlib import pyplot as plt
from skimage import color, transform


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


def show_grayscale_filters(net, layer):
    """
    Show filters for a layer that doesn't operate on RGB data (e.g. first convolutional layer of a temporal tower)
    """
    weights = _weights_for_layer(net, layer)
    kernel_images = []
    kernel_slice_image_dimensions = (15, 15)
    kernel_slice_spacer_width = 0.2
    kernel_spacer_height = 0.8

    for kernel in weights:
        # Kernel shape = (20, 3, 3), that's a stack of 20 3x3 kernels forming a volume
        # which is convolved over the input stack, we want to plot this stack.
        kernel_slice_images = [imresize(kernel_slice, kernel_slice_image_dimensions, interp='nearest') for kernel_slice in
                               kernel]
        spacer_dimensions = (int(kernel_slice_image_dimensions[0] * kernel_slice_spacer_width),
                             int(kernel_slice_image_dimensions[1]))
        spacer = _solid_image(spacer_dimensions[0], spacer_dimensions[1], brightness=1)
        kernel_slice_images_with_spacers = _intersperse(kernel_slice_images, spacer)
        kernel_images.append(np.concatenate(kernel_slice_images_with_spacers, 1))

    spacer_dimensions = (kernel_images[0].shape[1],
                         int(kernel_images[0].shape[0] * kernel_spacer_height))
    spacer = _solid_image(*spacer_dimensions, brightness=1)
    kernel_images_with_spacers = _intersperse(kernel_images, spacer)
    weight_vis = np.concatenate(kernel_images_with_spacers)
    return weight_vis


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


def _solid_image(width, height, brightness=1):
    """
    Generate a np.array with shape (width, height) 
    """
    pixel = np.array(brightness) * 255
    return np.tile(pixel, (height, width))


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
