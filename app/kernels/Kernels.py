import math
import numpy as np
import sys


def lineal(**kwargs):
    """
    K(x1,x2) =  x1.T x2
    """
    def f(x1, x2):
        return np.inner(x1, x2)
    return f


def polynomial(power, coef, **kwargs):
    """
    K(x1,x2) = (coef +k(x1,x2))**power
    """
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f


def rbf_kernel(gamma, **kwargs):
    """
    It is a Universal approximator K(x1,x2) = e**(-gama * d(x1, x2))
    """
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f