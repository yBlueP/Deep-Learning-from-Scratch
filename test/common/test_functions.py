import pytest
import numpy as np
import logging

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
        f.relu(np.array([-1.0, 1.0, 2.0])) == np.array([0.0, 1.0, 2.0]),
    )

def test_softmax():
    assert np.allclose(
        f.softmax(np.array([0.3, 2.9, 4.0])),
        np.array([0.01821127, 0.24519181, 0.73659692])
    )

def test_mean_squared_error():
    # assert np.allclose(
    #     f.mean_squared_error(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0, 0.1, 0, 0])),
    #     0.0975
    # )

    y = np.random.rand(5, 5)
    t = np.random.rand(5, 5)
    assert np.allclose(
        f.mean_squared_error_1(y, t), 
        f.mean_squared_error_2(y, t)
    )

    # logging.info(f"y = {y}")
    # logging.info(f"t = {t}")
    # logging.info(f"mean_squared_error_1 = {f.mean_squared_error_1(y, t)}")
    # logging.info(f"mean_squared_error_2 = {f.mean_squared_error_2(y, t)}")

def test_cross_entropy_error():
    y = np.random.rand(5, 5)
    t = np.random.rand(5, 5)
    assert np.allclose(
        f.cross_entropy_error_1(y, t), 
        f.cross_entropy_error_2(y, t)
    )

    # logging.info(f"y = {y}")
    # logging.info(f"t = {t}")
    # logging.info(f"cross_entropy_error_1 = {f.cross_entropy_error_1(y, t)}")
    # logging.info(f"cross_entropy_error_2 = {f.cross_entropy_error_2(y, t)}")