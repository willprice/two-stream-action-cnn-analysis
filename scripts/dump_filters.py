import logging
import click
import caffe
import importlib
from lib import visualisation
import matplotlib.pyplot as plt


logger = logging.getLogger()


@click.command()
@click.argument('net-config-module', type=click.STRING)
@click.argument('layer', type=click.STRING, default='conv1_1')
@click.argument('filter-image-path', type=click.Path(exists=False))
def dump_filters(net_config_module, layer, filter_image_path):
    net_config = importlib.import_module(net_config_module)
    net = caffe.Net(net_config.net_prototxt_path,
                    net_config.net_caffemodel_path,
                    caffe.TEST)
    fig = visualisation.show_filters(net, layer)
    plt.show()
    fig.savefig(filter_image_path)

if __name__ == '__main__':
    caffe.set_mode_gpu()
    logging.basicConfig(level=logging.INFO)
    dump_filters()
