# -*- coding: utf-8 -*-

import os
import re


class ActionRecognitionDataSet():
    """
    Represents an action recognition data set on disk
    structured by the following file hierarchy:

    .
    ├── videos
    │   ├── 00_Desk2_pick-up_plug_334-366.avi
    │   ├── 00_Desk2_pick-up_tape_1070-1099.avi
    │   ...
    ├── frames
    │   ├── 00_Desk2_pick-up_plug_334-366/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │   ├── 00_Desk2_pick-up_tape_1070-1099/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │   ...
    ├── flow
    │   ├── u
    │       ├── 00_Desk2_pick-up_plug_334-366/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │       ├── 00_Desk2_pick-up_tape_1070-1099/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │       ...
    │   ├── v
    │       ├── 00_Desk2_pick-up_plug_334-366/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │       ├── 00_Desk2_pick-up_tape_1070-1099/
    │           ├── frame000001.jpg
    │           ├── frame000002.jpg
    │           ...
    │       ...

    (The naming of the video files doesn't matter)
    """

    def __init__(self, data_root_path):
        self.data_root = data_root_path
        self.flow_root = os.path.join(self.data_root, 'flow')
        self.frames_root = os.path.join(self.data_root, 'frames')
        self.flow_u_root = os.path.join(self.flow_root, 'u')
        self.flow_v_root = os.path.join(self.flow_root, 'v')

    def get_frames(self, video_name):
        frames = [self.get_frame(video_name, frame_number) for frame_number in range(1, self.video_length(video_name))]
        return frames

    def get_frame(self, video_name, frame_number):
        return os.path.join(self.frames_root, video_name, "frame{:06d}.jpg".format(frame_number))

    def video_length(self, video_name):
        """
        Get the length of ``video_name`` in frames
        """
        return len(os.listdir(os.path.join(self.frames_root, video_name)))

    def get_flow_batches(self, video_name, batch_size=10):
        """
        We generate batches of flow u,v pairs by sliding a window of size ``batch_size``
        """
        frame_count = self._get_flow_count(video_name)
        batches = []
        for batch_index in range(0, frame_count - batch_size + 1):
            batch = self.get_flow_batch(video_name, start_frame=(batch_index + 1), batch_size=batch_size)
            batches.append(batch)
        return batches

    def get_flow_batch(self, video_name, start_frame=1, batch_size=10):
        """
        Construct a list of u,v flow frame pair paths to be read in and used for a temporal CNN

        e.g.
        ["../u/blah/frame000001.jpg",
         "../v/blah/frame000001.jpg",
         "../u/blah/frame000002.jpg",
         "../v/blah/frame000002.jpg",
         ...
        ]
        """

        assert start_frame >= 1, "start_frame should be greater than 0, but was {}".format(start_frame)
        assert batch_size >= 1, "batch_size should be greater than 0, but was {}".format(batch_size)

        def frame_number(frame_filename):
            """
            Parses frame of format frame%06d.jpg to get the number from the
            filename
            """
            pattern = re.compile(r'.*?(\d+)')
            matches = pattern.match(frame_filename)
            return int(matches.groups()[0])

        optical_flow_u_root = os.path.join(self.flow_u_root, video_name)
        optical_flow_v_root = os.path.join(self.flow_v_root, video_name)

        u_frames = sorted(list(os.listdir(optical_flow_u_root)), key=frame_number)
        v_frames = sorted(list(os.listdir(optical_flow_v_root)), key=frame_number)

        u_frames = [os.path.join(optical_flow_u_root, file) for file in  u_frames]
        v_frames = [os.path.join(optical_flow_v_root, file) for file in  v_frames]

        start_frame_index = start_frame - 1
        last_frame = int(start_frame_index + batch_size)
        selected_u_frames = u_frames[start_frame_index:last_frame]
        selected_v_frames = v_frames[start_frame_index:last_frame]

        frames = []
        for i in range(len(selected_u_frames)):
            frames.append(selected_u_frames[i])
            frames.append(selected_v_frames[i])
        return frames

    def _get_flow_count(self, video_name):
        flow_frames_dir = os.path.join(self.flow_u_root, video_name)
        flow_frame_count = len(os.listdir(flow_frames_dir))
        assert self.video_length(video_name) == flow_frame_count + 1, \
            "Expected exactly 1 less flow frame than spatial frames for video {}".format(video_name)
        return flow_frame_count

    def get_videos(self):
        return os.listdir(self.frames_root)
