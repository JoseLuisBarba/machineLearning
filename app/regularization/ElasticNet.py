import numpy as np
import math

class L1L2():
    """ Regularization for Elastic Net Regression """
    def __init__(self, Lambda: float, Alpha: float) -> None:
        self.Lambda = Lambda
        self.Alpha = Alpha
    
    def __call__(self, w):
        """
            L1 = lambda * (all i,j Sum |Wij|^2 ) ^ 1/2
        """
        l1 =  self.Alpha * np.linalg.norm(w)
        l2 =  0.5 * (1 - self.Alpha) *  w.T.dot(w)
        return self.Lambda * (l1 + l2)

    def grad(self, w):
        gradL1 = self.Alpha * np.sign(w)
        gradL2 = (1 - self.Alpha) * w
        return self.Lambda * (gradL1 + gradL2)
