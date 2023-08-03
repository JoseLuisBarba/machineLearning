import math
import numpy as np
import sys
from mStd import Std

class MatrixCov():
    def __init__(self, X, y=None) -> np.array:
        """ 
        Calculate the covariance matrix for matrix X 
        """
        self.X = X
        self.y = y

    def __call__(self,) -> np.array:

        if self.y is None:
            self.y = self.X

        n = np.shape(self.X)[0]

        Cov = (1 / (n-1)) * (self.X - self.X.mean(axis=0)).T.dot(self.y - self.y.mean(axis=0))

        return np.array(Cov, dtype=float)