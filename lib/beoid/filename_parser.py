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
    grammar = Object, maybe_some(omit("+"), Object)


class Point(Symbol):
    regex = re.compile(r'\d+')


class Range:
    grammar = attr("start", Point), omit("-"), \
              attr("end", Point)

    def __str__(self):
        return "Range([{}, {}])".format(self.start, self.end)


class ActionVideo:
    grammar = attr("id", NumericId), omit("_"), \
              attr("location", Location), omit("_"), \
              attr("action", Action), omit("_"), \
              attr("objects", Objects), omit("_"), \
              attr("range", Range), endl

    def __str__(self):
        return "ActionVideo([id: {}, location: {}, action: {}, objects: {}, range: {}])".format(
            self.id, self.location, self.action, self.objects, self.range
        )


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        p = Parser()
        videos = [p.parse(vid, ActionVideo)[1] for vid in f.readlines()]
        for video in videos:
            print(video)
