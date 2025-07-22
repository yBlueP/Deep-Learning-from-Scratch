import numpy as np

import common.gradient as g

def test_numerical_gradient():
    def f(x):
        return x[0]**2 + x[1]**2
    
    assert np.allclose(
        g.numerical_gradient(f, np.array([3.0, 4.0])),
        np.array([6.0, 8.0])
    )

    assert np.allclose(
        g.numerical_gradient(f, np.array([0.0, 2.0])),
        np.array([0.0, 4.0])
    )
