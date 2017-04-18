import transformers
import ucf101


top_blob = 'fc8'
second_top_blob = 'fc7'
bottom_blob = 'pool3'
underlay_frame_index = 10 # Currently ignored

labeller = ucf101.labeller
# Will be run with:
# frame_pattern.format(frame_index)
frame_pattern = "frame{:06d}.jpg"

transformer = transformers.flow_transformer

net_caffemodel_path = '/home/will/nets/vgg_16_ucf101/cuhk_action_temporal_vgg_16_split1.caffemodel'
net_prototxt_path = '/home/will/nets/vgg_16_ucf101/cuhk_action_temporal_vgg_16_split1_deploy.prototxt'
