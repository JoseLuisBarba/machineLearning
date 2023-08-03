import math
import numpy as np
import sys
from mStd import Std

class MatrixCorr():
    def __init__(self, X, y=None) -> np.array:
        """ 
        Calculate the correlation matrix for matrix X 
        """
        self.X = X
        self.y = y

    def __call__(self,) -> np.array:
        if self.y is None:
            self.y = self.X #inner
        n = np.shape(self.X)[0]
        Cov = (1 / n) * (self.X - self.X.mean(0)).T.dot(self.Y - self.Y.mean(0))
        std_dev_X = np.expand_dims(Std(self.X)(), 1)
        std_dev_y = np.expand_dims(Std(self.y)(), 1)
        Corr = np.divide(Cov, std_dev_X.dot(std_dev_y.T))
        return np.array(Corr, dtype=float)



    


