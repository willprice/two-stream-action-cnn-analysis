import os
import shutil
from tempfile import mkdtemp


def touch(filename):
    with open(filename, 'a'):
        os.utime(filename, None)


class FakeDataSet():
    def __init__(self):
        self.root = mkdtemp()
        self.videos = [
            "04_Door2_open_door_284-333",
            "05_Treadmill2_press_button_2966-2985"
        ]

        self.frame_counts = {
            self.videos[0]: 20,
            self.videos[1]: 21,
        }

    def create(self):
        frames_root = os.path.join(self.root, 'frames')
        flow_root = os.path.join(self.root, 'flow')
        flow_u_root = os.path.join(flow_root, 'u')
        flow_v_root = os.path.join(flow_root, 'v')

        for dir in [frames_root, flow_root, flow_u_root, flow_v_root]:
            os.makedirs(dir)

        for video_name in self.videos:
            frame_count = self.frame_counts[video_name]

            for dir in [flow_u_root, flow_v_root, frames_root]:
                os.makedirs(os.path.join(dir, video_name))

            for dir in [flow_u_root, flow_v_root]:
                for i in range(frame_count - 1):
                    touch(os.path.join(dir, video_name, "frame{:06d}.jpg".format(i + 1)))

            for dir in [frames_root]:
                for i in range(frame_count):
                    touch(os.path.join(dir, video_name, "frame{:06d}.jpg".format(i + 1)))

    def remove(self):
        if self.root is not None and os.path.exists(self.root):
            shutil.rmtree(self.root)

