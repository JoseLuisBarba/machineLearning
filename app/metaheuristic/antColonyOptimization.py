import numpy as np
from functools import reduce


class Graph(object): #check
    def __init__(self, costMatrix: np.ndarray) -> None:
        self.costMatrix = costMatrix
        self.nNodes = len(costMatrix)

    def pathCost(self, path: np.ndarray) -> float:
        return sum( self.costMatrix[edge] for edge in path)

    

    

class Ant():
    def __init__(self, startNode: int, graph: Graph, pheromoneMatrix: Graph) -> None:
        self.startNode = startNode
        self.graph = graph
        self.pheromoneMatrix  = pheromoneMatrix 
        try:
            if startNode < 0 or startNode > (self.graph.nNodes - 1):
                raise ValueError(f'<<startNode>> must be in [0 , {self.graph.nNodes - 1}]')
        except ValueError as e:
            print(e) 
        


    def chooseEdge(self,edgesPheromone: np.ndarray, edgesCost: np.ndarray, visited: set):
        """
        Select an edge that leads  to non-viseted node
        """
        pheromone = np.copy(edgesPheromone)
        pheromone[list(visited)] = 0 #zeroes the propability of revisiting a node 
        frecuency = (pheromone ** self.alpa) * ((1.0 / edgesCost) ** self.beta)  
        probEdges = frecuency / frecuency.sum() #probability of selecting each edge that leads to a non-viseted node
        move = np.random.choice(a = range(self.graph.nNodes), size= 1, p=probEdges)[0]
        return move

    def generatePath(self, ): #check
        path = []
        visited = set() 
        visited.add(self.startNode)
        current = self.startNode
        for i in range(self.graph.nNodes - 1):
            next = self.chooseEdge(edgesPheromone= self.pheromoneMatrix[current], 
                                   edgesCost= self.graph.costMatrix[current], 
                                   visited=visited)
            path.append((current, next))
            current = next
            visited.add(next)
        path.append((current, self.startNode))
        return path


    def __call__(self, alpha: float, beta: float) -> list:
        self.alpa = alpha
        self.beta = beta
        return self.generatePath()



class ElitistAS(object): #check
    def __init__(   
                self,
                graph: Graph,
                startNode: int,
                nAnts: int,
                nbest: int,
                iterations: int,
                rho: float,
                alpha: float,
                beta: float,
                w: float ) -> None:
        self.graph = graph 
        self.cost =  graph.costMatrix
        self.pheromone = np.ones(shape= self.cost.shape)
        self.startNode =  startNode
        self.nAnts = nAnts
        self.nbest = nbest
        self.iterations = iterations
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.w = w
        

    def canidateSolutions(self): #check
        allPaths = []
        #for each ant
        for i in range(self.nAnts):
            path = Ant(
                    startNode= self.startNode, 
                    graph= self.graph,
                    pheromoneMatrix= self.pheromone 
            )(alpha= self.alpha, beta = self.beta )
            allPaths.append((path, self.graph.pathCost(path)))
        return allPaths
    
    def updatePheromone(self,candidateSolutions, nSolutions, w): #check
        sortedPaths = sorted(candidateSolutions, key= lambda path: path[1]) # O(nSolution * log(nSolution))
        for path, dist in sortedPaths[: nSolutions]:
            for edge in path:
                #update the pheromone quantity 
                self.pheromone[edge] +=  (w / self.cost[edge])

    def __call__(self,): #check
        bestHistorical = [[], np.inf]
        try:
            if self.nAnts <= 0 :
                raise ValueError(f'<<Ants>> must be in [0 , n]')
        except ValueError as e:
            print(e) 
        except Exception as e:
            print(e)
        else:
            for i in range(self.iterations):
                candidateSolutions = self.canidateSolutions()
                #decay pheromone level
                self.pheromone = (1 - self.rho) *  self.pheromone
                self.updatePheromone(
                    candidateSolutions= candidateSolutions,
                    nSolutions= self.nAnts,
                    w = 1.0
                )
                self.updatePheromone(
                    candidateSolutions= candidateSolutions,
                    nSolutions= self.nbest,
                    w = self.w
                )
                bestPath = min(candidateSolutions, key= lambda path: path[1] )
                if bestPath[1] < bestHistorical[1]:
                    bestHistorical = bestPath
        return bestHistorical

    




if __name__ == '__main__':


    CostMatrix = np.array(
        [
            [np.inf,      9,      11,       7],
            [     9, np.inf,      15,       5],
            [    11,     15,  np.inf,       4],
            [     7,      5,       4,  np.inf]
        ]
    )
    graph = Graph(costMatrix= CostMatrix)
    node0 = 0
    nAnts = 5
    nBest = 2
    w = 0.01
    nGen = 50
    rho = 0.5
    alpha = 0.5
    beta = 0.5
    shortestPath = ElitistAS(
        graph= graph,
        startNode=  node0,
        nAnts= nAnts,
        nbest= nBest,
        iterations= nGen,
        rho= rho,
        alpha= alpha,
        beta= beta,
        w= w
    )()
    print(f'The shortest path is {shortestPath }')


 

