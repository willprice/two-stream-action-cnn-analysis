#!/usr/bin/env python3


"""
Each video is of the form

    v_X_gY_cZ.avi

Where

X = Action class label (CamelCase action class)
Y = Group (numeric)
Z = Clip number (numeric)
"""

from pypeg2 import *
import re

number = re.compile(r'\d+')


class Action(Symbol):
    regex = re.compile(r'[a-zA-Z]+([a-zA-Z]+)*')

class Clip(Symbol):
    regex = number

class Group(Symbol):
    regex = number

class UCF101ActionVideo:
    grammar = omit("v_"), attr("action", Action), \
              omit("_g"), attr("group", Group), \
              omit("_c"), attr("clip", Clip), optional(omit(".avi"))


    def __str__(self):
        return "UCF101ActionVideo([action: {}, group: {}, clip: {}])".format(
            self.action, self.group, self.clip
        )


def ucf101_filename_parser(filename):
    p = Parser()
    return p.parse(filename, UCF101ActionVideo)[1]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("USAGE: {} filelist".format(sys.argv[0]))
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        p = Parser()
        vids = f.readlines()[:-1]
        videos = [p.parse(vid, UCF101ActionVideo)[1] for vid in vids]
        for video in videos:
            print(video)
