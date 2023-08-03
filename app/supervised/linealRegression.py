import numpy as np
import math
import random
from app.regularization.Ridge import L2
from app.regularization.Lasso import L1
from app.regularization.ElasticNet import L1L2



"""
Permiso de entorno virtual 
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
 .\env\Scripts\activate
"""


    

class Regression():
    def __init__(self, epochs: int, learningRate: float) -> None: #correct
        self.epochs = epochs
        self.learningRate = learningRate
        
    def initializeWeights(self, instanceDimension): #correct
        """
        Xavier/Glorot Initialization:
            sigma = 1 / sqrt(instanceDimension)
            Wi ~ U( -sigma , sigma)
        """
        sigma = 1 / math.sqrt(instanceDimension)
        self.w = np.random.uniform(low=-sigma,high=sigma,size=(instanceDimension,))
        #self.w = np.ones(instanceDimension)
        
    def cost(self, X , y):  #correct
      yPred = X.dot(self.w) #[m, 1]
      return np.sum((yPred - y)**2) / (2 * len(y)) 

    def gradient(self, X, y): #correct
      yPred = X.dot(self.w) #[m, 1]
      return   X.T.dot(yPred - y) / len(y)  # vector dw [n, 1]

    def fit(self,X, y): #correct
        """
            w := w - alfa aJ/aw
        """
        X = np.insert(X,0,1, axis=1) #[[1,x1,x2,x3],[1,x1,x2,x3],[1,x1,x2,x3]]
        self.trainError = []
        self.initializeWeights(instanceDimension= X.shape[1])
        # Gradient descent:
        for i in range(self.epochs):
            J = self.cost(X,y)
            self.trainError.append(J)
            # Update the weights
            self.w = self.w - self.learningRate * self.gradient(X, y)
            
    def predict(self, X): #correct
        X = np.insert(X,0,1, axis=1) #[[1,x1,x2,x3],[1,x1,x2,x3],[1,x1,x2,x3]]
        yPred = X.dot(self.w) 
        return yPred



class LassoRegression(Regression):
    def __init__(self, epochs: int, learningRate: float, regFact: float) -> None:
        super().__init__(epochs=epochs,learningRate=learningRate)
        self.regFact = regFact

    def cost(self, X, y):
        return super().cost(X, y) + L1(Lambda= self.regFact)(w= self.w) 
    
    def gradient(self, X, y):
        return super().gradient(X, y) + L1(Lambda= self.regFact).grad(w= self.w)
    
    def fit(self, X, y):
        return super().fit(X,y)


class RidgeRegression(Regression):
    def __init__(self, epochs: int, learningRate: float, regFact: float) -> None:
        super().__init__(epochs=epochs,learningRate=learningRate)
        self.regFact = regFact

    def cost(self, X, y):
        return super().cost(X, y) + L2(Lambda= self.regFact)(w= self.w) 
    
    def gradient(self, X, y):
        return super().gradient(X, y) + L2(Lambda= self.regFact).grad(w= self.w)
    
    def fit(self, X, y):
        return super().fit(X,y)
    

class ElasticNet(Regression):
    def __init__(self, epochs: int, learningRate: float, Lambda: float, Alpha: float) -> None:
        super().__init__(epochs=epochs,learningRate=learningRate)
        self.Lambda = Lambda
        self.Alpha = Alpha

    def cost(self, X, y):
        return super().cost(X, y) + L1L2(Lambda=self.Lambda, Alpha= self.Alpha)(w= self.w)
    
    def gradient(self, X, y):
        return super().gradient(X, y) + L1L2(Lambda=self.Lambda, Alpha= self.Alpha).grad(w= self.w)
    
    def fit(self, X, y):
        return super().fit(X,y)





if __name__ == '__main__':
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('C:\\Users\\Admin\\Documents\\AdvancedAI\\app\\supervised\\gabcargo.csv')
    df.head(-1)
    X = np.array(df.drop(['index','Consignatario',	'Origen','Flete'], 1))
    y = np.array(df['Flete'])
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

    lm = Regression(1000,0.01)
    lm.fit(X_train,y_train)
    yPred = lm.predict(X_test)



    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    print('MSE test: %.3f' % (
            mean_squared_error(y_test, yPred)))
    print('R^2 test: %.3f' % (
            r2_score(y_test, yPred)))
    
    print('······························ Lasso ···········································')
    ls = LassoRegression(epochs=1000,learningRate=0.01, regFact=100)
    ls.fit(X_train,y_train)
    yPred = ls.predict(X_test)



    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    print('MSE test: %.3f' % (
            mean_squared_error(y_test, yPred)))
    print('R^2 test: %.3f' % (
            r2_score(y_test, yPred)))
    
    print('······························ Ridge ···········································')
    rd = RidgeRegression(epochs=1000,learningRate=0.01, regFact=10)
    rd.fit(X_train,y_train)
    yPred = rd.predict(X_test)



    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    print('MSE test: %.3f' % (
            mean_squared_error(y_test, yPred)))
    print('R^2 test: %.3f' % (
            r2_score(y_test, yPred)))
    


    print('······························ Elastic Net ···········································')
    en = ElasticNet(epochs=1000,learningRate=0.01,Lambda= 0.5, Alpha= 0.9)
    en.fit(X_train,y_train)
    yPred = en.predict(X_test)



    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    print('MSE test: %.3f' % (
            mean_squared_error(y_test, yPred)))
    print('R^2 test: %.3f' % (
            r2_score(y_test, yPred)))
    






