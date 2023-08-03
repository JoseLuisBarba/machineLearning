import numpy as np
import math

    
class L2():
    """ Regularization for Ridge Regression """
    def __init__(self, Lambda: float):
        self.Lambda = Lambda
    
    def __call__(self, w: np.array) :
        """
            L2 = lambda * w.T w or lambda * sum( all bj**2 in B)
        """
        return self.Lambda *  w.T.dot(w)

    def grad(self, w):
        return self.Lambda * w
    
