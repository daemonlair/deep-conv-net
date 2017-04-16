#!/usr/bin/env python



from __future__ import print_function

class Layer(object):
    """
    docstring for Layer
    """
    def __init__(self, name, type):
        super(Layer, self).__init__()
        self.name = name
        self.type = type

    def fprop(self):
        assert False, "No implementation for fprop"

    def bprop(self):
        assert False, "No implementation for bprop"

    def getOutputShape(self):
        assert False, "No implementation for getOutputShape"

    def changeBatchSize(self, batch_size):
        assert False, "No implementation for changeBatchSize"

    def update(self):
        assert False, "No implementation for update"


    def predict(self):
        assert False, "No implementation for predict"


    def loss(self):
        assert False, "No implementation for loss"
