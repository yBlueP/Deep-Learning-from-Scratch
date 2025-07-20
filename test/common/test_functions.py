import pytest
import numpy as np

import common.functions as f


def test_step_function():
    assert np.all(
        f.step_function(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == np.array([0, 0, 0, 1, 1])
    )

def test_sigmoid():
    assert np.allclose(
        f.sigmoid(np.array([-1.0, 1.0, 2.0])),
        np.array([0.26894142, 0.73105858, 0.88079708]),
    )

def test_ReLU():
    assert np.all(
        f.ReLU(np.array([-1.0, 1.0, 2.0])) == np.array([0.0, 1.0, 2.0]),
    )

def test_softmax():
    assert np.allclose(
        f.softmax(np.array([0.3, 2.9, 4.0])),
        np.array([0.01821127, 0.24519181, 0.73659692])
    )