#!/usr/bin/env python3

from pypeg2 import *
import re

number = re.compile(r'\d+')


class NumericId(Symbol):
    regex = number

class Location(str):
    """
    location: Location of the video (Desk, Door, Printer, Sink, Row, Treadmill)
    run: The run id (1-3)

    N.B: Subjects performed the actions repeatedly in the same location thus producing multiple `run`s which are encoded
    in the `run` field.
    """
    grammar = attr("location", re.compile(r'[a-zA-Z]+')), \
              attr("run", re.compile(r'\d*'))


class Action(Symbol):
    regex = re.compile(r'[a-zA-Z]+(-[a-zA-Z]+)*')


class Object(Symbol):
    """
    """
    regex = re.compile(r'[a-zA-Z]+(-[a-zA-Z]+)*')


class Objects(List):
    grammar = Object, maybe_some("+", Object)


class Point(Symbol):
    regex = re.compile(r'\d+')


class Range(str):
    grammar = attr("start", Point), "-", attr("end", Point)


class Video(str):
    """
    subject: The ID of the person performing the action
    location: The location in which the action was performed
    """
    grammar = attr("subject", NumericId), "_", \
              attr("location", Location)


class Clip(str):
    """
    video: The source video, describes the subject performing the actions, and the location
    action: Mode of interaction (MOI)
    objects: Task Relevant Objects (TROs) taking part in the action
    range: The start and end frame offsets from the initial frame index of the video

    Descriptions of the attributes were determined from the file
    "BEOID - Narrations and Object Interactions Ground Truth.pdf" distributed in the BEOID download
    """
    grammar = attr("video", Video), "_", \
              attr("action", Action), "_", \
              attr("objects", Objects), "_", \
              attr("range", Range)


if __name__ == '__main__':
    import sys
    clip = sys.argv[1]
    print("Parsing '{}'".format(clip))
    parsed = parse(clip, Clip)
    print("Composed: '{}'".format(compose(parsed)))
