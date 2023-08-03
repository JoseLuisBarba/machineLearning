import math
import numpy as np
import sys


class MSE():
    def __init__(self, yTrue, yPred) -> None:
        """
        Calculate the metric MSE between yTrue and yPred 
        """
        self.yTrue = yTrue
        self.yPred = yPred
    def __call__(self,) -> float:
        return np.mean(np.power(self.yPred - self.yTrue,2))
    


if __name__ == '__main__':
   #test

   x = MSE(np.array([1,2,3,4,5]),np.array([1,2,5,4,5]))()
   print(x)

