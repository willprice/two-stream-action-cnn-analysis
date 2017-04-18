import numpy as np
import pandas as pd


def successive_pairs(l):
    """ Return a list of pairs by taking successive elements from the list l

    >>> list(successive_pairs([1,2]))
    [(1, 2)]
    >>> list(successive_pairs([1,2,3,4]))
    [(1, 2), (2, 3), (3, 4)]
    >>> list(successive_pairs([1]))
    []

    """
    return zip(l[:-1], l[1:])


def normalise(ndarray):
    """ Normalises an numpy array to between 0 and 1 by the max/min

    >>> normalise(np.array([1, 2]))
    array([ 0.,  1.])
    >>> normalise(np.array([[1, 2], [2, 5]]))
    array([[ 0.  ,  0.25],
           [ 0.25,  1.  ]])

    """
    range = ndarray.max() - ndarray.min()
    if range == 0:
        return ndarray
    return (ndarray - ndarray.min()) / range


def jitter_l2(att_map_frame1, att_map_frame2):
    """ Compute the jitter between two arrays by computing the L2 difference and summing

    >>> jitter_l2(np.array([1, 2]), np.array([1, 2]))
    0.0
    >>> jitter_l2(np.array([1, 2]), np.array([0, 2]))
    1.0
    >>> jitter_l2(np.array([1, 2]), np.array([3, 2]))
    2.0
    """
    assert att_map_frame1.shape == att_map_frame2.shape
    delta = att_map_frame1 - att_map_frame2
    return np.sum(np.sqrt(np.power(delta, 2)))


def jitter(attention_maps_df, jitter_measure=jitter_l2):
    """
    Take a DataFrame of attention maps for a single network/EBP type
    over multiple clips and compute the jitter for each pair of
    contiguous attention maps per clip
    """
    def clip_jitter(clip_df):
        clip_jitter = []

        previous_row = clip_df.iloc[0]
        for row_index, row in clip_df[1:].iterrows():
            jitter = jitter_measure(normalise(previous_row['Attention Map']),
                                    normalise(row['Attention Map']))
            jitter_entry = previous_row.to_dict()
            del jitter_entry['Attention Map']
            jitter_entry['Jitter'] = jitter
            clip_jitter.append(jitter_entry)
            previous_row = row
        return pd.DataFrame(clip_jitter)
    return attention_maps_df.groupby('Clip').apply(clip_jitter)
