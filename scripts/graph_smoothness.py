#!/usr/bin/env python3

from prelude import *
import os
import click
import numpy as np
import pandas as pd
import seaborn as sns

@click.command()
@click.argument('spatial-contrastive-excitation-maps-blob',
                type=click.Path(exists=True))
@click.argument('spatial-non-contrastive-excitation-maps-blob',
                type=click.Path(exists=True))
@click.argument('temporal-contrastive-excitation-maps-blob',
                type=click.Path(exists=True))
@click.argument('temporal-non-contrastive-excitation-maps-blob',
                type=click.Path(exists=True))
@click.argument('measure',
                type=click.STRING)
@click.argument('graph-file',
                type=click.Path(exists=False))
def graph_smoothness_program(spatial_contrastive_excitation_maps_blob,
                             spatial_non_contrastive_excitation_maps_blob,
                             temporal_contrastive_excitation_maps_blob,
                             temporal_non_contrastive_excitation_maps_blob,
                             measure, graph_file):
    spatial_contrastive_excitation_maps = pd.read_pickle(spatial_contrastive_excitation_maps_blob)
    spatial_non_contrastive_excitation_maps = pd.read_pickle(spatial_non_contrastive_excitation_maps_blob)
    temporal_contrastive_excitation_maps = pd.read_pickle(temporal_contrastive_excitation_maps_blob)
    temporal_non_contrastive_excitation_maps = pd.read_pickle(temporal_non_contrastive_excitation_maps_blob)


def graph_smoothness(video_excitation_maps, smoothness_measure):
    smoothness = {}
    for video, excitation_maps in video_excitation_maps.items():
        smoothness[video] = ExcitationMapSmoothness(smoothness_measure, excitation_maps)

    means = map(lambda name, smoothness: smoothness.mean, smoothness.items())


