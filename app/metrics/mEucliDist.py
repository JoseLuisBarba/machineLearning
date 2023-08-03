import math
import numpy as np
import sys




class EuclideanDistance():
    def __init__(self, x1: np.array, x2: np.array) -> None:
        """ 
        Calculates the eucledean distance between two vectors.
        sqrt(all i sum((x1i - x2i)**2))
        """
        self.x1 = x1
        self.x2 = x2
    
    def __call__(self,) -> float:
        return math.sqrt(sum( pow(i - j, 2) for i , j in zip(self.x1, self.x2)))
    
    

if __name__ == '__main__':
    x1 = np.array([1,2,3,7])
    x2 = np.array([1,2,3,4])
    print(EuclideanDistance(x1,x2)())


