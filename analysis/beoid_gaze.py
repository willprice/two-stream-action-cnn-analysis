import os
import math
from scipy.signal import argrelmax
import matplotlib.pyplot as plt

from beoid.gaze import *
from beoid.filename_parser import *
import data

def first(pair):
    return pair[0]

def second(pair):
    return pair[1]

def np_to_pairs(xs, ys):
    """
    >>> np_to_pairs(np.array([1, 3]), np.array([2, 4]))
    [(1, 2), (3, 4)]
    """
    return list(zip(xs, ys))


def pairs_to_np(pairs):
    """
    >>> pairs_to_np([(1, 2), (3, 4)])
    (array([1, 3]), array([2, 4]))
    """
    xs = np.array([first(pair) for pair in pairs])
    ys = np.array([second(pair) for pair in pairs])
    return (xs, ys)


def peak_locations(attention_map, top_n=5):
    """
    >>> attention_map = np.array([[1, 2, 3], [2, 4, 2]])
    >>> peak_locations(attention_map, top_n=2)
    (array([1, 0]), array([1, 2]))
    """
    order = 1
    maxima_x_idx = argrelmax(attention_map, axis=0, order=order, mode='wrap')
    maxima_y_idx = argrelmax(attention_map, axis=1, order=order, mode='wrap')
    maxima_idx_pairs = set(zip(first(maxima_x_idx), second(maxima_x_idx))) & \
                       set(zip(first(maxima_y_idx), second(maxima_y_idx)))
    maxima_idx = pairs_to_np(maxima_idx_pairs)
    maxima_values = attention_map[maxima_idx]
    top_n_maxima_idx = maxima_values.argsort()[::-1][:top_n]
    return (maxima_idx[0][top_n_maxima_idx], maxima_idx[1][top_n_maxima_idx])


def scale_locations(src_shape, dest_shape, locations):
    """
    Scale indices from src_shape to dest_shape
    
    >>> scale_locations((28, 28), (640, 480), ([0, 28], [0, 28]))
    ([0, 640], [0, 480])
    >>> scale_locations((5, 5), (6, 6), ([4], [4]))
    ([5], [5])
    
    """
    assert(len(src_shape) == 2)
    assert(len(dest_shape) == 2)
    f1 = lambda x: round(x * dest_shape[0] / src_shape[0])
    f2 = lambda x: round(x * dest_shape[1] / src_shape[1])
    return list(map(f1, locations[0])), list(map(f2, locations[1]))


def gaze_data_for_clip(video_info_df, clip_name):
    parsed_name = parse(clip_name, Clip)
    key = parsed_name.subject + "_" + compose(parsed_name.location, autoblank=False)
    start_offset = int(parsed_name.range.start)
    stop_offset = int(parsed_name.range.end)
    first_video_frame = int(video_info_df.loc[key].initial_frame)
    first_clip_frame = first_video_frame + start_offset
    last_clip_frame = first_video_frame + stop_offset
    assert(first_clip_frame <= last_clip_frame)
    gaze_for_full_video = gaze[key]
    gaze_for_clip = gaze_for_full_video.query('frame >= {} and frame <= {}'.format(first_clip_frame, last_clip_frame))
    gaze_for_clip['frame'] = gaze_for_clip['frame'] - first_clip_frame + 1
    # We don't necessarily have gaze data for all the frames in the clip
    assert(len(gaze_for_clip) <= (last_clip_frame - first_clip_frame + 1))
    return gaze_for_clip


def select_valid_gaze(gaze_df):
    valid_gaze_idx = pd.notnull(gaze_df.fixation_9)
    return gaze_df[valid_gaze_idx]


def select_temporal_attention_map_from_gaze_df(temporal_attention_map_df, gaze_df):
    valid_frames = gaze_df['frame']
    attention_maps_with_gaze_data_idx = temporal_attention_map_df.starting_frame.isin(valid_frames)
    return temporal_attention_map_df[attention_maps_with_gaze_data_idx]


def select_attention_maps_for_clip(attention_maps_df, clip_name):
    return attention_maps_df[attention_maps_df.video == clip_name]

def combine_gaze_and_attention_map(gaze_df, attention_maps_df):
    merged_df = pd.merge(gaze_df, attention_maps_df, left_on='frame', right_on='starting_frame')
    expected_output_length = min(len(gaze_df), len(attention_maps_df))
    assert(len(merged_df) == expected_output_length)
    return merged_df


def compare_attention_maps_to_gaze(attention_maps_with_gaze_df, frame_size=(640, 480)):
    def compare_attention_map_to_gaze(attention_map_with_gaze_row):
        attention_map = attention_map_with_gaze_row.attention_map
        attention_map_maxima_idx = peak_locations(attention_map)
        attention_map_gaze_locations = scale_locations(attention_map.shape, frame_size, attention_map_maxima_idx)
        def distance_to_gaze(point):
            return (gaze[0] - point[0], gaze[1] - point[1])
        distance_components = list(map(distance_to_gaze, np_to_pairs(*attention_map_gaze_locations)))
        def euclidean_distance(distance_pair):
            x_dist = distance_pair[0]
            y_dist = distance_pair[1]
            return math.sqrt(x_dist**2 + y_dist**2)

        distances_euclidean = list(map(euclidean_distance, distance_components))
        closest_peak_to_gaze = np_to_pairs(*attention_map_gaze_locations)[np.argmin(distances_euclidean)]
    attention_maps_with_gaze_df.apply(compare_attention_map_to_gaze, axis=1)


def compute_peaks(attention_maps_df, top_n=10):
    peaks = attention_maps_df.apply(lambda row: peak_locations(row.attention_map, top_n), axis=1)
    try:
        frame_column = attention_maps_df.frame
    except AttributeError:
        frame_column = attention_maps_df.starting_frame
    return pd.DataFrame({ 'frame': frame_column,
                          'peaks': peaks })



def convert_peak_idx_to_gaze_locations(peak_idx, attention_map_shape, video_shape):
    cnn_gaze_locations = peak_idx.apply(lambda peak: scale_locations(attention_map_shape, video_shape, peak))
    gaze_2d_xs = cnn_gaze_locations.apply(first)
    gaze_2d_ys = cnn_gaze_locations.apply(second)
    return pd.DataFrame(data={'gaze_2d_xs': gaze_2d_xs,
                              'gaze_2d_ys': gaze_2d_ys})


def compare_gaze(potential_gaze_df, gaze_ground_truth_df):
    """
    
    :param potential_gaze_df: DataFrame with 'gaze_2d_xs' and 'gaze_2d_ys'
    :param gaze_ground_truth_df: DataFrame with 'gaze_2d_x' and 'gaze_2d_y'
    :return: pd.Series where each element is a list containing the distance of each element in potential_gaze_df
             from the ground truth taken from gaze_ground_truth_df.
             
    """
    print(potential_gaze_df)
    print(gaze_ground_truth_df)
    comparison_df = pd.merge(potential_gaze_df, gaze_ground_truth_df, left_index=True, right_index=True)
    def distance(comparison_row):
        x_difference = np.array(map(lambda x: x - comparison_row.gaze_2d_x, comparison_row.gaze_2d_xs))
        y_difference = np.array(map(lambda y: y - comparison_row.gaze_2d_y, comparison_row.gaze_2d_ys))
        difference = np.sqrt(np.power(x_difference, 2) + np.power(y_difference, 2))
        return difference
    comparison_df['distance'] = comparison_df.apply(distance, axis=1)


if __name__ == '__main__':
    spatial_contrastive_beoid = pd.read_pickle(data.path_attention_map_spatial_contrastive_beoid)
    spatial_non_contrastive_beoid = pd.read_pickle(data.path_attention_map_spatial_non_contrastive_beoid)
    temporal_contrastive_beoid = pd.read_pickle(data.path_attention_map_temporal_contrastive_beoid)
    temporal_non_contrastive_beoid = pd.read_pickle(data.path_attention_map_temporal_non_contrastive_beoid)

    temporal_contrastive = temporal_contrastive_beoid

    gaze = dict()
    for filepath in data.beoid_gaze_filenames:
        filename = os.path.basename(filepath)
        gaze_key = filename[:-len(data.gaze_file_suffix)]
        gaze[gaze_key] = read_gaze(filepath)

    video_info = []
    for video in gaze.keys():
        video_gaze_rows = select_valid_gaze(gaze[video])
        info = {
            'video': video,
            'initial_frame': video_gaze_rows.frame.min(),
            'gaze_2d_x_min': video_gaze_rows.gaze_2d_x.min(),
            'gaze_2d_x_max': video_gaze_rows.gaze_2d_x.max(),
            'gaze_2d_y_min': video_gaze_rows.gaze_2d_y.min(),
            'gaze_2d_y_max': video_gaze_rows.gaze_2d_y.max(),
        }
        video_info.append(info)
    video_info = pd.DataFrame(video_info).sort_values('video').set_index('video')

    # I think argrelmax is just looking across one dimension, in strips, so we end up with
    # lots of strips with non 2D maxima
    # We need to get it to look in both dimensions somehow, perhaps doing it twice, once in one direction,
    # then again in the other
    #scale_locations(attention_map.shape, (640, 480), amap_argmax)

    #if False:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    x_bounds = range(0, 28)
    #    y_bounds = x_bounds
    #    ax.plot_wireframe(*np.meshgrid(x_bounds, y_bounds), first_amap, cmap='viridis', rstride=1, cstride=1)
    #    ax.scatter(slice_max_idx[1], slice_max_idx[0], attention_map[slice_max_idx], c='yellow', depthshade=False, s=100)
    #    plt.show()

    clip = '04_Sink2_press_button_676-718'
    temporal_attention_maps = temporal_contrastive
    gaze = select_valid_gaze(gaze_data_for_clip(video_info, clip))

    attention_maps = select_attention_maps_for_clip(temporal_attention_maps, clip)
    analyzable_attention_maps = select_temporal_attention_map_from_gaze_df(attention_maps, gaze)

    peak_idx = compute_peaks(analyzable_attention_maps, top_n=1)
    print(peak_idx)

    video_size = (640, 480)
    cnn_gaze_idx = convert_peak_idx_to_gaze_locations(peak_idx, (28, 28), video_size)

    comparison = compare_gaze(cnn_gaze_idx, gaze)
    print(comparison)

    #compare_gaze(cnn_gaze_idx, gaze)


    #slice_max_idx = peak_locations(attention_map)
