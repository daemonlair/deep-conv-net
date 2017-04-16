#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class MaxPoolLayer(Layer):
    """
    docstring for MaxPoolLayer
    """
    def __init__(self, arg):
        super(PoolingLayer, self).__init__()
        self.arg = arg

    def fprop(self, X, size=2, stride=2):
        def maxpool(X_col):
            max_idx = np.argmax(X_col, axis=0)
            out = X_col[max_idx, range(max_idx.size)]
            return out, max_idx

        return _pool_forward(X, maxpool, size, stride)

    def bprop(dout, cache):
        def dmaxpool(dX_col, dout_col, pool_cache):
            dX_col[pool_cache, range(dout_col.size)] = dout_col
            return dX_col

        return _pool_backward(dout, dmaxpool, cache)

    def _pool_forward(self, X, pool_fun, size=2, stride=2):
        n, d, h, w = X.shape
        h_out = (h - size) / stride + 1
        w_out = (w - size) / stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        X_reshaped = X.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

        out, pool_cache = pool_fun(X_col)

        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)

        cache = (X, size, stride, X_col, pool_cache)

        return out, cache


    def _pool_backward(self, dout, dpool_fun, cache):
        X, size, stride, X_col, pool_cache = cache
        n, d, w, h = X.shape

        dX_col = np.zeros_like(X_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dX = dpool_fun(dX_col, dout_col, pool_cache)

        dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
        dX = dX.reshape(X.shape)

        return dX
