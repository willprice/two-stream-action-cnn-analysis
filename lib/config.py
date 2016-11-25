# -*- coding: utf-8 -*-
import os

ROOT = os.path.dirname(os.path.realpath(__file__))


"""
Path to excitation backprop modified caffe directory (built)
"""
CAFFE_ROOT = os.path.expanduser("~/src/Caffe-ExcitationBP")


"""
Path to the root data folder which should mimic the following directory tree:
data/
├── spatial
│   ├── 04_Printer2_pull_drawer_94-276
│   ├── 04_Printer2_push_drawer_241-293
│   ├── ...
├── temporal
│   ├── 04_Printer2_pull_drawer_94-276
│   ├── 04_Printer2_push_drawer_241-293
│   ├── ...
"""
DATA_ROOT  = os.path.expanduser("~/thesis/data")

"""
Path to directory that holds your networks in caffe format.  Each network
should be in it's own subdirectory with a deploy.txt and X.caffemodel
"""
NETS_ROOT = os.path.expanduser("~/nets/")
