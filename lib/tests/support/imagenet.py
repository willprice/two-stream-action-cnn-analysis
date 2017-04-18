from transformers import imagenet_transformer
import numpy as np
import caffe
import os
_image_path = os.path.join(os.path.dirname(__file__), 'image.jpg')

def example_imagenet_image(shape=(1, 3, 224, 224)):
    transformer = caffe.io.Transformer(
        {'data': shape}
    )
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    return transformer.preprocess('data', caffe.io.load_image(_image_path))
