#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class DropoutLayer(Layer):
    """
    docstring for DropoutLayer
    """
    def __init__(self, arg):
        super(DropoutLayer, self).__init__()
        self.arg = arg

    def fprop(self, X, p_dropout):
        u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
        out = X * u
        cache = u
        return out, cache

    def bprop(self, dout, cache):
        dX = dout * cache
        return dX        
