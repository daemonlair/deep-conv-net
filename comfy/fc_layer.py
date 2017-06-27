#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class FullyConnectedLayer(Layer):
    """
    docstring for FullyConnectedLayer
    """
    def __init__(self, arg):
        super(FullyConnectedLayer, self).__init__()
        self.arg = arg

    def fprop(self, X, W, b):
        out = X @ W + b
        cache = (W, X)
        return out, cache

    def bprop(self, dout, cache):
        W, h = cache

        dW = h.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ W.T

        return dX, dW, db
        
