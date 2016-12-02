#!/usr/bin/env python3

import skimage
import click
import os
import subprocess

@click.command()
@click.argument('video-path')
@click.argument('output-path')
@click.option('--frame-pattern',
              default='frame_%06d.jpg')
def decompose_video(video_path, output_path, frame_pattern):
    """
    Decomposes video into a series of frames written to `output_path`
    """
    if not os.path.exists(video_path):
        raise RuntimeError("Could not find video: {}".format(video_path))
    os.makedirs(output_path, exist_ok=True)

    ffmpeg_cmd = ['ffmpeg',
                  '-i', video_path,
                  '-f', 'image2',
                  output_path + '/' + frame_pattern]
    process = subprocess.Popen(ffmpeg_cmd)
    process.wait()


if __name__ == '__main__':
    decompose_video()
