import caffe
import os

_net_root = os.path.expanduser("~/nets/coco/")

def coco_net():
    prototxt = os.path.join(_net_root, "deploy.prototxt")
    caffemodel = os.path.join(_net_root, "GoogleNetCOCO.caffemodel")

    assert os.path.exists(prototxt), "Cannot find prototxt: {}".format(prototxt)
    assert os.path.exists(caffemodel), "Cannot find caffemodel: {}".format(caffemodel)

    return caffe.Net(
        prototxt,
        caffe.TRAIN,
        weights=caffemodel
    )