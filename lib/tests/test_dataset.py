# -*- coding: utf-8 -*-
from pprint import pformat
from nose.tools import eq_
import os
import unittest
from .support.fake_data_set import FakeDataSet
from ..dataset import ActionRecognitionDataSet


def eq_list(expected, actual):
    eq_(expected, actual,
        msg="expected: \n{}\nactual: \n{}\n".format(pformat(expected), pformat(actual)))

class ActionRecognitionDataSetTests(unittest.TestCase):
    fake_dataset = FakeDataSet()

    @classmethod
    def setUpClass(cls):
        cls.fake_dataset.create()

    @classmethod
    def tearDownClass(cls):
        cls.fake_dataset.remove()

    def test_get_flow_batch_from_first_frame(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        video = self.fake_dataset.videos[0]
        expected_flow_batch = list(map(lambda dir: os.path.join(self.fake_dataset.root, dir), [
            os.path.join("flow", "u", video, "frame000001.jpg"),
            os.path.join("flow", "v", video, "frame000001.jpg"),
            os.path.join("flow", "u", video, "frame000002.jpg"),
            os.path.join("flow", "v", video, "frame000002.jpg"),
            os.path.join("flow", "u", video, "frame000003.jpg"),
            os.path.join("flow", "v", video, "frame000003.jpg"),
            os.path.join("flow", "u", video, "frame000004.jpg"),
            os.path.join("flow", "v", video, "frame000004.jpg"),
            os.path.join("flow", "u", video, "frame000005.jpg"),
            os.path.join("flow", "v", video, "frame000005.jpg"),
            os.path.join("flow", "u", video, "frame000006.jpg"),
            os.path.join("flow", "v", video, "frame000006.jpg"),
            os.path.join("flow", "u", video, "frame000007.jpg"),
            os.path.join("flow", "v", video, "frame000007.jpg"),
            os.path.join("flow", "u", video, "frame000008.jpg"),
            os.path.join("flow", "v", video, "frame000008.jpg"),
            os.path.join("flow", "u", video, "frame000009.jpg"),
            os.path.join("flow", "v", video, "frame000009.jpg"),
            os.path.join("flow", "u", video, "frame000010.jpg"),
            os.path.join("flow", "v", video, "frame000010.jpg"),
        ]))

        actual_flow_batch = dataset.get_flow_batch(video, start_frame=1, batch_size=10)

        eq_list(expected_flow_batch, actual_flow_batch)

    def test_get_flow_batch_up_to_last_frame(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        video = self.fake_dataset.videos[0]
        expected_flow_batch = list(map(lambda dir: os.path.join(self.fake_dataset.root, dir), [
            os.path.join("flow", "u", video, "frame000010.jpg"),
            os.path.join("flow", "v", video, "frame000010.jpg"),
            os.path.join("flow", "u", video, "frame000011.jpg"),
            os.path.join("flow", "v", video, "frame000011.jpg"),
            os.path.join("flow", "u", video, "frame000012.jpg"),
            os.path.join("flow", "v", video, "frame000012.jpg"),
            os.path.join("flow", "u", video, "frame000013.jpg"),
            os.path.join("flow", "v", video, "frame000013.jpg"),
            os.path.join("flow", "u", video, "frame000014.jpg"),
            os.path.join("flow", "v", video, "frame000014.jpg"),
            os.path.join("flow", "u", video, "frame000015.jpg"),
            os.path.join("flow", "v", video, "frame000015.jpg"),
            os.path.join("flow", "u", video, "frame000016.jpg"),
            os.path.join("flow", "v", video, "frame000016.jpg"),
            os.path.join("flow", "u", video, "frame000017.jpg"),
            os.path.join("flow", "v", video, "frame000017.jpg"),
            os.path.join("flow", "u", video, "frame000018.jpg"),
            os.path.join("flow", "v", video, "frame000018.jpg"),
            os.path.join("flow", "u", video, "frame000019.jpg"),
            os.path.join("flow", "v", video, "frame000019.jpg"),
        ]))

        actual_flow_batch = dataset.get_flow_batch(video, start_frame=10, batch_size=10)

        eq_list(expected_flow_batch, actual_flow_batch)

    def test_get_flow_batches(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        video = self.fake_dataset.videos[0]
        flow_batch_size = 10

        actual_flow_batches = dataset.get_flow_batches(video, flow_batch_size)

        # video length is 20 frames
        # so there will be 19 flow frame pairs
        # We're sliding with batch size 10
        # batch 1 will be frames 1 - 10
        # batch 2 will be frames 2 - 11
        # batch 9 will be frames 9 - 18
        # batch 10 will be frames 10 - 19

        total_flow_frame_count = self.fake_dataset.frame_counts[video] - 1
        expected_batch_count = total_flow_frame_count - flow_batch_size + 1

        eq_(expected_batch_count, len(actual_flow_batches))

        for batch in actual_flow_batches:
            eq_(flow_batch_size*2, len(batch))

        for batch_index in range(expected_batch_count):
            eq_(os.path.join(self.fake_dataset.root, "flow", "u", video, "frame{:06d}.jpg".format(batch_index + 1)),
                actual_flow_batches[batch_index][0])

    def test_get_frame(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        video = self.fake_dataset.videos[0]
        frame_number = 1
        expected_frame_path = os.path.join(self.fake_dataset.root, 'frames', video, "frame{:06d}.jpg".format(frame_number))

        frame_path = dataset.get_frame(video, frame_number)
        eq_(expected_frame_path, frame_path)

    def test_get_frames(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        video = self.fake_dataset.videos[0]
        expected_frame_paths = [os.path.join(self.fake_dataset.root, 'frames', video, 'frame{:06d}.jpg'.format(i)) for i in range(1, self.fake_dataset.frame_counts[video])]

        actual_frame_paths = dataset.get_frames(video)
        eq_list(expected_frame_paths, actual_frame_paths)

    def test_get_videos(self):
        dataset = ActionRecognitionDataSet(self.fake_dataset.root)
        eq_list(self.fake_dataset.videos, dataset.get_videos())

if __name__ == '__main__':
    import nose
    nose.main()
