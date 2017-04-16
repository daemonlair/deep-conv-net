#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class SoftmaxLayer(Layer):
    """
    docstring for SoftmaxLayer
    """
    def __init__(self, arg):
        super(SoftmaxLayer, self).__init__()
        self.arg = arg
    
