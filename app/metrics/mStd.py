import math
import numpy as np
import sys
from mVariance import Var



class Std():
    def __init__(self, X) -> None:
        """ Return the std of the matrix X """

        self.X = X

    def __call__(self,) -> np.array:
        
        return np.sqrt(Var(self.X)())
    
    

if __name__ == '__main__':
   

   X = np.array([[1,2,3,4,5],[1,2,7,8,5],[1,5,6,4,5]])
   print(np.std(X,axis=0))
   print(Std(X)())
