import numpy as np

class Sigmoid():
    def __call__(self, z):
        """
        O~(z) = 1 / (1  + e^(-z))
        Link: https://paperswithcode.com/method/sigmoid-activation
        """
        return 1 / (1 + np.exp( -z))
    
    def grad(self, z):
        """
        O~'(z) = aO~(z)/az = O~(z) (1  - O~(z))
        link: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        """
        return self.__call__(z) * ( 1 - self.__call__(z))