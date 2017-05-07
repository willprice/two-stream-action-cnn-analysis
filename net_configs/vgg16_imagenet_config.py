import cnn_utils
import transformers

top_blob = 'fc8'
second_top_blob = 'fc7'
bottom_blob = 'pool3'

net_caffemodel_path = '/home/will/nets/vgg-16/VGG_ILSVRC_16_layers.caffemodel'
net_prototxt_path = '/home/will/nets/vgg-16/deploy.prototxt'

transformer = transformers.imagenet_transformer
