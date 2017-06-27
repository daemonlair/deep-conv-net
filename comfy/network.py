#!/usr/bin/env python



from __future__ import print_function
from conv_layer import ConvLayer
from dropout_layer import DropoutLayer
from fc_layer import FullyConnectedLayer
from input_layer import InputLayer
from pooling_layer import MaxPoolLayer
from relu_layer import ReLULayer
from softmax_layer import SoftmaxLayer


class Network(object):
    """
    docstring for Network
    """
    def __init__(self, layers, l2_decay=0.001, debug=False, learning_rate=0.001):
        super(Network, self).__init__()
        mapping = {
            "input": lambda x: InputLayer(x),
            "fc": lambda x: FullyConnectedLayer(x),
            "convolution": lambda x: ConvLayer(x),
            "pool": lambda x: MaxPoolLayer(x),
            "squaredloss": lambda x: SquaredLossLayer(x),
            "softmax": lambda x: SoftmaxLayer(x),
            "relu": lambda x: ReLULayer(x),
            "dropout": lambda x: DropoutLayer(x)
        }
        self.layers = []
        self.l2_decay = l2_decay
        self.debug = debug
        self.learning_rate = learning_rate

    def train(self):
        pass
