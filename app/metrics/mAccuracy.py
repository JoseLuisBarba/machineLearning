import math
import numpy as np
import sys

class Accurasy():
    def __init__(self, yTrue, yPred) -> float:
        """
        Calculate the accuaracy between yTrue and yPred 
        """
        self.yTrue = yTrue
        self.yPred = yPred
    def __call__(self,) -> float:
        return np.sum( self.yPred == self.yTrue, axis=0) / len(self.yPred)
    


if __name__ == '__main__':
   #test

   x = Accurasy(np.array([2,2,3,4,5]),np.array([1,2,5,4,5]))()
   print(x)