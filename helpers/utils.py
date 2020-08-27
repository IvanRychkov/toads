import numpy as np


def float_equal(a, b, threshold=1e-6):
    return np.abs(a - b) < threshold


__all__ = ['float_equal']
