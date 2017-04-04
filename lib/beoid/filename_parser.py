#!/usr/bin/env python3

from pypeg2 import *
import re

number = re.compile(r'\d+')


class NumericId(Symbol):
    regex = number


class Location(Symbol):
    regex = re.compile(r'[a-zA-Z]+\d*')


class Action(Symbol):
    regex = re.compile(r'[a-zA-Z]+(-[a-zA-Z]+)*')


class Object(Symbol):
    regex = re.compile(r'[a-zA-Z]+(-[a-zA-Z]+)*')


class Objects(List):
    grammar = Object, maybe_some("+", Object)


class Point(Symbol):
    regex = re.compile(r'\d+')


class Range(str):
    grammar = attr("start", Point), "-", attr("end", Point)


class ActionVideo(str):
    grammar = attr("id", NumericId), "_", \
              attr("location", Location), "_", \
              attr("action", Action), "_", \
              attr("objects", Objects), "_", \
              attr("range", Range)


if __name__ == '__main__':
    import sys
    vid = sys.argv[1]
    print("Parsing '{}'".format(vid))
    parsed = parse(vid, ActionVideo)
    print("Composed: '{}'".format(compose(parsed)))
