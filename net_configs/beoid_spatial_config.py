import beoid
import cnn_utils
import transformers

top_blob = 'fc8'
second_top_blob = 'fc7'
bottom_blob = 'pool3'
underlay_frame_index = 10

labeller = beoid.labeller

net_caffemodel_path = '/home/will/nets/dual-stream/spatial/kfold1borders25.caffemodel'
net_prototxt_path = '/home/will/nets/dual-stream/spatial/deploy.prototxt'

transformer = transformers.imagenet_transformer