import math
import numpy as np



class Var():
    def __init__(self, X) -> None:
        """ Return the variance of the matrix X """

        self.X = X

    def __call__(self,):

        mean = np.ones(np.shape(self.X )) * self.X.mean(0)

        m = self.X.shape[0]

        return (1 / m) * np.diag((self.X - mean).T.dot((self.X - mean))) 
    



    


if __name__ == '__main__':
   

   X = np.array([[1,2,3,4,5],[1,2,7,8,5],[1,5,6,4,5]])
 
   print(np.var(X,axis=0))
   print( Var(X)())
