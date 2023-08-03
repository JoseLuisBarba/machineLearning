import numpy as np
import math


class GaussianNaiveBayes():
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = np.unique(self.y) # gets unique classes  of y
        self.w = [] #we create a document for each class
        #for each row that has label Yi
        for index , label in enumerate(self.labels):
            XEqualsL =  X[np.where(y == label)]
            self.w.append([])
            #add mean and variance for each feature 
            for feature in XEqualsL.T:
                self.w[index].append({'mean': feature.mean(), 'var': feature.var()})

    def likelihood(self, mean: float, var: float, x: float) -> float:
        """
            Gaussian likelihood 
            Link: https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html

        """
        divZero = 1e-3 
        alpha = 1 / math.sqrt((2*math.pi)*var + divZero)
        exponent = math.exp(-((x - mean)** 2)/(2 * var + divZero))
        return alpha * exponent
    
    def prior(self,label: int):
        """
            Calculates the prio of label class 
        """
        return np.mean(self.y == label) #frecuency
    
    def classify(self, x):
        posteriors = []
        for index , label in enumerate(self.labels):
            PY = self.prior(label=label)
        
            PX_Y = math.prod(  self.likelihood(mean=document['mean'],var=document['var'],x=feature) for feature, document in zip(x, self.w[index]))

            posteriors.append(PX_Y* PY)

        return self.labels[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self.classify(x) for x in X])



    








if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    data = load_iris()

    X = data.data
    y = data.target
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))



    lr = GaussianNaiveBayes()
    

    lr.fit(X_train,y_train)
    yPred = lr.predict(X_test)



    from sklearn.metrics import accuracy_score


    print('Accuracy test: %.3f' % (
            accuracy_score(y_test,yPred)))

    