#!/usr/bin/env python



from __future__ import print_function
from layer import Layer

class InputLayer(Layer):
    """
    docstring for InputLayer
    """
    def __init__(self, config):
        super(InputLayer, self).__init__()
        self.shape = input_shape
