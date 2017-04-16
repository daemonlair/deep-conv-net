#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class ReLULayer(Layer):
    """
    docstring for ReLULayer
    """
    def __init__(self, arg):
        super(ReLULayer, self).__init__()
        self.arg = arg

    def fprop(self, X):
        out = np.maximum(X, 0)
        cache = X
        return out, cache

    def bprop(self, dout, cache):
        dX = dout.copy()
        dX[cache <= 0] = 0
        return dX
