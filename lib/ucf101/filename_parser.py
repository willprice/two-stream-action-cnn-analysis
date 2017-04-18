#!/usr/bin/env python3


"""
Each video is of the form

    v_X_gY_cZ.avi

Where

X = Action class label (CamelCase action class)
Y = Video ID (numeric)
Z = Clip ID (numeric)

Source videos are split into action clips e.g.

* v_Drumming_g16_c01.avi
* v_Drumming_g16_c02.avi
* v_Drumming_g16_c03.avi
* v_Drumming_g16_c04.avi
* v_Drumming_g16_c05.avi
* v_Drumming_g16_c06.avi

These videos all come from the same source video unique 
identified by `v_Drumming_g16` (i.e. the 16th drumming video).
The video is split into short segments containing the action,
e.g. c_01, c_02, ..., c_06, these indices only make sense in 
the context of the source video `v_Drumming_g16`.
"""

from pypeg2 import *
import re

number = re.compile(r'\d+')


class Action(Symbol):
    """
    An action performed by a human, e.g. 'HeadMassage'
    """
    regex = re.compile(r'[a-zA-Z]+([a-zA-Z]+)*')


class Clip(Symbol):
    """
    The ID of a clip present in a video
    """
    regex = number


class Video(Symbol):
    """
    The ID of a video  
    """
    regex = number


class UCF101ActionClip(str):
    grammar = "v_", attr("action", Action), \
              "_g", attr("video", Video), \
              "_c", attr("clip", Clip), optional(omit(".avi"))


    def __str__(self):
        return "UCF101ActionVideo([action: {}, video: {}, clip: {}])".format(
            self.action, self.video, self.clip
        )


def ucf101_filename_parser(filename):
    p = Parser()
    return p.parse(filename, UCF101ActionClip)[1]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("USAGE: {} filename".format(sys.argv[0]))
        sys.exit(1)

    print(parse(sys.argv[1], UCF101ActionClip))
