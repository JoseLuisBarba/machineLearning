import numpy as np


class Loss():
    def __init__(self, yTrue, yPred) -> None:
        self.yTrue = yTrue
        self.yPred = yPred
    def __call__(self, ) -> any:
        raise NotImplementedError()
    
    def grad(self,):
        raise NotImplementedError()
    

class MSE(Loss):
    def __init__(self, yTrue, yPred) -> None:
        super().__init__(yTrue= yTrue, yPred= yPred)

    def __call__(self) -> any:
        return np.sum((self.yPred - self.yTrue)**2) / (2 * len(self.yTrue)) 
    
    def grad():
        pass