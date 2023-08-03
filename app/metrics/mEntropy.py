import math
import numpy as np
import sys




class Entropy():
    def __init__(self, y) -> float:
        """ 
        Return the entropy of vector y
        H(x) = - sum(pi * log2(pi))
        """
        self.y = y

    def __call__(self,) -> np.array:
        
        Labels = np.unique(self.y) #get unique lables of vector y
        #calculates the probability pi of label yi
        Hx= 0
        for label in Labels:
            count = len(self.y[self.y == label]) #frequency
            pi = count / len(self.y) #probability
            Hx += (-pi) * math.log2(pi)
        return Hx
    
    

if __name__ == '__main__':
   

    y = np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9])
    print(Entropy(y)())
   