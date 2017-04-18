import click
import caffe
import visualisation
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@click.command()
@click.argument('net-prototxt-path',
                type=click.Path(exists=True))
@click.argument('net-caffemodel-path',
                type=click.Path(exists=True))
@click.argument('layer',
                default='conv1')
@click.argument('output-image-path',
                type=click.Path(exists=False))
def filter_visualisation(net_prototxt_path, net_caffemodel_path, layer, output_image_path):
    net = caffe.Net(net_prototxt_path,
                    net_caffemodel_path,
                    caffe.TEST)
    filter_responses = visualisation.show_filters(net, layer)
    filter_responses.savefig(output_image_path)


if __name__ == '__main__':
    filter_visualisation()
