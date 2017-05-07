# 2SCNN Feature analysis

Working repository for feature analysis of 2SCNN

## Models

* [CUHK VGG16-2SCNN models trained on UCF101
  ](https://github.com/yjxiong/caffe/tree/action_recog/models/action_recognition)
* UoB VGG16-2SCNN trained on BEOID (not publicly available)


## Dependencies

* Python 3
  * matplotlib
  * numpy
  * scipy
  * pandas
  * seaborn
  * click
  * skimage 
* Jupyter notebook (6.0)
* IPykernel for Jupyter (5.0)
* Bash
* [Caffe (Excitation BP fork)](https://github.com/jimmie33/Caffe-ExcitationBP),
  preferably built with GPU support.

## Setup

* Edit `lib/config.py` to point to your local [excitation bp
* caffe](https://github.com/jimmie33/Caffe-ExcitationBP) installation,
  nets path (e.g. `caffe/models`, I like to keep mine in my home directory), and
  data root.

## Repository overview

* `lib`: Most of the meaty code lives in here, code for handling datasets,
  performing contrastive EBP, mapping from class ids to class names.
* `net_configs`: Python library of configuration scripts defining settings
  peculiar to models.
* `scripts`: Python CLI scripts for generating attention maps, stitching videos,
  graphing smoothness
* `generated`: A few bash scripts live in here to call the python scripts that
  do most of the work.
* `notebooks`: All the Jupyter notebooks used for experimental analysis reside
  here. They're meant to be run in numerical order. Notebooks with the same
  numerical prefix can be run in any order.
* `./run_jupyter.sh` is a helper script to setup my environment with the correct
  `LD_LIBRARY_PATH` and `PATH` env variables necessary to find Caffe, CuDNN etc.
