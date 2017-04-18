#!/usr/bin/env python

# NOTE: The files from the download were first sanitized as they have a mix of delimiters
# I used the following bash script to sanitize the tabs/spaces into single space delimiters
#
# #!/usr/bin/env bash
# for f in *.txt; do
#   tmp="$(mktemp)"
#   expand "$f" | awk '$1=$1' > "$tmp"
#   mv "$tmp" "$f"
# done

import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt


def gaussian_2d(size, center=None, variance=0):
    """
    Calculate a discerete gaussian matrix

    Arguments:
        size: (width : int, height : int) or int, specifies the resulting size of the gaussian matrix
        center: center of gaussian (defaults to (width/2, height/2)
        variance: $\sigma$ in the gaussian formula

    Returns:
        np.array(): shape (height, width), gaussian matrix
    """

    if isinstance(size, tuple) or isinstance(size, list):
        width = size[0]
        height = size[1]
    else:
        width = size
        height = size

    if isinstance(variance, tuple) or isinstance(variance, tuple):
        var_x = variance[0]
        var_y = variance[1]
    elif variance == 0:
        var_x = width**2
        var_y = height**2
    else:
        var_x = variance
        var_y = variance

    if center is None:
        center = ((width - 1) / 2, (height - 1) / 2)

    x, y = np.mgrid[0:width+1, 0:height+1]
    pos = np.dstack((x, y))
    gaussian = scipy.stats.multivariate_normal(center, [[var_x, 0], [0, var_y]])

    return gaussian.pdf(pos)


def read_gaze(gaze_ssv_filepath):
    """
    Read BEOID gaze data from space separated values (SSV) file

    Arguments:
        gaze_ssv_filepath: Path to gaze data file (after sanitisation using the
                           bash script at the top of this file)

    Returns:
        gaze: pd.DataFrame

    """
    with open(gaze_ssv_filepath, 'r') as gaze_csv_file:
        csv_header = [
            "time",
            "frame",
            "gaze_2d_x",
            "gaze_2d_y",
            "camera_3d_x",
            "camera_3d_y",
            "camera_3d_z",
            "gaze_3d_x",
            "gaze_3d_y",
            "gaze_3d_z",
            "fixation_3",
            "fixation_5",
            "fixation_9"
        ]
        # -1:    Indicates that the camera could not be tracked, occurs in all
        #        3d and fixation columns
        # -2000: Indicates absence of gaze data due to failure in pupil tracking, occurs
        #        only in gaze_2d_x, and gaze_2d_y columns
        # We swap them out for numpy.na since they cannot be confused as each other.
        na_values = {
            'gaze_2d_x': '-2000',
            'gaze_2d_y': '-2000',
            'camera_3d_x': '-1',
            'camera_3d_y': '-1',
            'camera_3d_z': '-1',
            'gaze_3d_x': '-1',
            'gaze_3d_y': '-1',
            'gaze_3d_z': '-1',
            'fixation_3': '-1',
            'fixation_5': '-1',
            'fixation_9': '-1',
        }
        gaze = pd.read_csv(gaze_csv_file,
                           delimiter=' ',
                           na_values=na_values,
                           names=csv_header)
        return gaze


if __name__ == '__main__':
    import sys
    gaze = read_gaze(sys.argv[1])
    g = gaussian_2d(28, variance=3**2)
    g /= g.max()
    plt.imshow(g)
    plt.set_cmap('viridis')
    plt.colorbar()
    plt.show()
