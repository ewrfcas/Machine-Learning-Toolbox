import numpy as np
from sklearn.base import BaseEstimator

#traditional fisher with different b for binary
#the default b is (n1m1+n2m2)/n1+n2
class Fisher(BaseEstimator):
    def __init__(self, intercept=(1,2)):
        self.intercept_=intercept

    def fit(self, X, y):
        labels=np.unique(y)
        if len(labels)>2:
            print('just for binary')
            return
        X0=X[np.where(y==labels[0]),:][0]
        X1=X[np.where(y==labels[1]),:][0]
        self.m0=np.mean(X0,axis=0)
        self.m1=np.mean(X1,axis=0)
        self.Sw=np.dot(np.transpose(X0-self.m0),X0-self.m0)+np.dot(np.transpose(X1-self.m1),X1-self.m1)
        self.W=np.dot(np.linalg.pinv(self.Sw),np.transpose(self.m0-self.m1))

        #calculate for b
        self.mx0=np.dot(self.m0,self.W)
        self.mx1=np.dot(self.m1,self.W)
        self.b=-1*(min(self.mx0,self.mx1)+abs(self.mx0-self.mx1)*(self.intercept_[0]/self.intercept_[1]))

        return self

    def get_params(self, deep=True):
        return [self.W,self.b]

    def decision_function(self,X):
        return np.dot(X,self.W)+self.b

    def predict_proba(self,X):
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([prob, 1 - prob])

    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)

