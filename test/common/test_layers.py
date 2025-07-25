import common.layers as L

import numpy as np
import logging


def test_sigmoid():
    sigmoid = L.Sigmoid()
    x = np.array([-1.0, 1.0, 2.0])
    assert np.allclose(sigmoid(x), np.array([0.26894142, 0.73105858, 0.88079708]))



