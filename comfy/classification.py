#!/usr/bin/env python


from __future__ import print_function
import numpy as np
import network

n_classes = 40
net = Network([
        {"type": "input", "input_shape": (112, 92, 3)},
        {"type": "convolution", "filters": 5, "size": 3},
        {"type": "dropout"},
        {"type": "relu"},
        {"type": "pool", "size": 2},
        {"type": "fc", "neurons": 100},
        {"type": "dropout"},
        {"type": "relu"},
        {"type": "fc", "neurons": n_classes},
        {"type": "relu"},
        {"type": "softmax", "categories": n_classes}]
    )
