import numpy as np
import pandas as pd
from app.activation.sigmoid import Sigmoid
from app.supervised.linealRegression import Regression



class LogisticRegression(Regression):
    def __init__(self, epochs: int, learningRate: float) -> None: #correct test
        super().__init__(epochs= epochs, learningRate= learningRate)

    def predict(self, X): #correct
        X = np.insert(X,0,1, axis=1) #[[1,x1,x2,x3],[1,x1,x2,x3],[1,x1,x2,x3]]
        z = X.dot(self.w) # lineal 
        return np.round(Sigmoid()(z)).astype(np.int32) #threshold 

    
    def cost(self, X , y):  #correct
        yPred = X.dot(self.w) #[m, 1]
        z = Sigmoid()(z= yPred)
        print(z)
        return np.sum(-y * np.log(z) - (1 - y) * np.log(1 - z))

    
    def gradient(self, X, y): #correct
        yPred = X.dot(self.w) #[m, 1]
        z = Sigmoid()(z= yPred)
        return   X.T.dot(z - y) # vector dw [n, 1]



if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('C:\\Users\\Admin\\Documents\\AdvancedAI\\app\\supervised\\passexam.csv',sep=';')
    df.head(-1)
    print(df.columns)
    X = np.array(df.drop(['pass'],1))
    y = np.array(df['pass'])
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))



    lr = LogisticRegression(epochs=4000,learningRate=0.1)
    

    lr.fit(X_train,y_train)
    yPred = lr.predict(X_test)



    from sklearn.metrics import accuracy_score


    print('Accuracy test: %.3f' % (
            accuracy_score(y_test,yPred)))

    
  