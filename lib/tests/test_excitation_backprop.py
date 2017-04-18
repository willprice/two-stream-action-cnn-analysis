import unittest
from .support.nets import coco_net
from .support.imagenet import example_imagenet_image
from nose.tools import eq_
import numpy as np
from excitation_backprop import ExcitationBackprop


class ExcitationBackpropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = coco_net()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_prop_is_idempotent(self):
        eb = ExcitationBackprop(self.net, 'loss3/classifier' , 'pool5/7x7_s1', 'pool3/3x3_s2')
        image = 255 * np.random.rand(3, 224, 224)
        first = eb.prop(image)
        second = eb.prop(image)
        eq_(first, second)

    def test_backprop_is_idempotent(self):
        eb = ExcitationBackprop(self.net, 'loss3/classifier' , 'pool5/7x7_s1', 'pool3/3x3_s2')
        image = example_imagenet_image()
        eb.prop(image)

        neuron = np.argmax(self.net.blobs[eb.top_blob_name].data[0].flatten())

        first = eb.backprop(1)
        assert np.any(first), "All elements were 0, unlikely this is correct behaviour, more likely net wasn't backpropped"

        second = eb.backprop(1)
        comparison =  first == second
        assert np.all(comparison)
