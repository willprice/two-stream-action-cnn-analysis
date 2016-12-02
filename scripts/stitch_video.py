#!/usr/bin/env python3

from subprocess import Popen
import subprocess
import os
import click


@click.command()
@click.argument('source-dir', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--frame-pattern', default="frame_%06d.jpg", help="Path to directory containing frames")
@click.option('--vcodec', default='libx264', help='Video codec')
@click.option('--fps', default=24, help='Video frames per second')
def stitch_videos(source_dir, output_path, frame_pattern, vcodec, fps):
    if os.path.exists(output_path):
        overwrite = click.prompt(
            "The output path '{}' already exists, are you sure you want to regenerate and overwrite it?".format(output_path),
            type=bool
        )
        if not overwrite:
            return

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # -y = overwrite without asking
    # -r <fps> = set fps
    # -i <filename|pattern> = input filename or pattern
    # -vcodec <vcodec> = set video codec
    ffmpeg_process =Popen(["ffmpeg",
                           "-y",
                           "-start_number", "1",
                           "-i", frame_pattern,
                           "-r", str(fps),
                           "-vcodec", vcodec,
                           output_path],
                          cwd=source_dir,
                          stdin=subprocess.DEVNULL)
    ffmpeg_process.wait()


if __name__ == '__main__':
    stitch_videos()
