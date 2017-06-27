#!/usr/bin/env python



from __future__ import print_function
import numpy as np
from layer import Layer

class ConvLayer(Layer):
    """
    docstring for ConvLayer
    """
    def __init__(self, arg):
        super(ConvLayer, self).__init__()
        self.arg = arg

    def fprop(self, X, W, b, stride=1, padding=1):
        cache = W, b, stride, padding
        n_filters, d_filter, h_filter, w_filter = W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)

        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        cache = (X, W, b, stride, padding, X_col)

        return out, cache

    def bprop(self, dout, cache):
        X, W, b, stride, padding, X_col = cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

        return dX, dW, db
