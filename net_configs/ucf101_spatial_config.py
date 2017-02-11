import ucf101
import caffe
import scipy.io as sio
import numpy as np
import cnn_utils

top_blob = 'fc8-1'
second_top_blob = 'fc7'
bottom_blob = 'pool3'

labeller = ucf101.labeller
# Will be run with:
# frame_pattern.format(frame_index)
frame_pattern = "frame{:06d}.jpg"


net_caffemodel_path = '/home/will/nets/vgg_16_ucf101/cuhk_action_spatial_vgg_16_split1.caffemodel'
net_prototxt_path = '/home/will/nets/vgg_16_ucf101/cuhk_action_spatial_vgg_16_split1_deploy.prototxt'

transformer = cnn_utils.imagenet_transformer
