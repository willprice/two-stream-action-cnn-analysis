import beoid
import cnn_utils


top_blob = 'fc8'
second_top_blob = 'fc7'
bottom_blob = 'pool3'
underlay_frame_index = 10

labeller = beoid.labeller

# Will be run with:
# frame_pattern.format(frame_index)
frame_pattern = "frame_{:06d}.jpg"

transformer =  cnn_utils.flow_transformer

net_caffemodel_path = '/home/will/nets/dual-stream/temporal/kfold1temporal.caffemodel'
net_prototxt_path = '/home/will/nets/dual-stream/temporal/deploy.prototxt'
