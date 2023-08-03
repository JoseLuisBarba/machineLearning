import numpy as np
import math

class L1():
    """ Regularization for Lasso Regression """
    def __init__(self, Lambda: float):
        self.Lambda = Lambda
    
    def __call__(self, w):
        """
            L1 = lambda * (all i,j Sum |Wij|^2 ) ^ 1/2
        """
        return self.Lambda * np.linalg.norm(w)

    def grad(self, w):
        return self.Lambda * np.sign(w)
    
