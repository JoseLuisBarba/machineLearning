import numpy as np
import math
import random
from functools import reduce
from itertools import combinations




class TabuSearch():
    def __init__(self, initialConfiguration: function, 
                 searchOperator: function, 

                 ) -> None:
        pass

    def fitness(self, x):
        pass

    def searchOperator(self, x):
        pass

    def calculateTenureTime(self,):
        return random.random()

    def chooseInitialConfiguration(self,):
        pass

    def generateNeighbors(self):
        pass
    def SelectTheBestNeighbor(self):
        pass

    def updateTheBest(self,):
        pass
    def updateTheCurrent(self,):
        pass

    def endCondition(self,):
        pass
    def updateTabuList():
        pass
    def outputBestSolution():
        pass


    def __call__(self, ):
        self.chooseInitialConfiguration()
        while True:
            self.generateNeighbors()
            self.SelectTheBestNeighbor()
            self.updateTheBest()
            if(self.endCondition()):
                return self.outputBestSolution()
            self.updateTabuList()
            self.updateTheCurrent()

    def __str__(self) -> str:
        pass

if __name__ == '__main__':
    F = np.array(
        [
            [0, 5, 2, 4, 1],
            [5, 0, 3, 0, 2],
            [2, 3, 0, 0, 0],
            [4, 0, 0, 0, 5],
            [1, 2, 0, 5, 0],
        ], dtype=np.int8
    )

    D = np.array(
        [
            [0, 1, 1, 2, 3],
            [1, 0, 2, 1, 2],
            [1, 2, 0, 1, 2],
            [2, 1, 1, 0, 1],
            [3, 2, 2, 1, 0],
        ], dtype=np.int8
    )

    T = np.zeros(shape=(5, 5), dtype = np.int8)

    x = np.array([1, 2, 3, 4, 5])

    

    print(random.shuffle(list(x)))



    for swamp in combinations(np.array([1,2,3,4]), 2):

        print(swamp[0], swamp[1])

        





