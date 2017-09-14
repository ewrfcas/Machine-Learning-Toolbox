import numpy as np
from sklearn.base import BaseEstimator

#traditional fisher with different b for binary
#the default b is (n1m1+n2m2)/n1+n2
class Fisher(BaseEstimator):
    def __init__(self, intercept_num=1):
        self.intercept_num=intercept_num

    def fit(self, X, y):
        labels=np.unique(y)
        if len(labels)>2:
            print('just for binary')
            return
        X0=X[np.where(y==labels[0]),:]
        X1=X[np.where(y==labels[1]),:]
        self.m0=np.mean(X0,axis=0)
        self.m1=np.mean(X1,axis=0)
        self.Sw=np.dot(np.ravel(X0-self.m0),X0-self.m0)+np.dot(np.ravel(X1-self.m1),X1-self.m1)
        self.W=np.dot(np.linalg.pinv(self.Sw),np.ravel(self.m0-self.m1))

        #calculate for b
        self.b=[]
        self.mx0=np.dot(X0,self.W)
        self.mx1=np.dot(X1,self.W)
        for i in range(self.intercept_num):
            self.b.append(min(self.mx0,self.mx1)+abs(self.mx0-self.mx1)*((i+1)/(self.intercept_num+1)))

        return self

    def get_params(self, deep=True):
        return [self.W,self.b]

    def decision_function(self,X,b_index):
        if b_index==None:
            dis=np.dot(X,self.W)+np.mean(self.b)
        else:
            dis=np.dot(X,self.W)+self.b[b_index]
        return dis

    def predict_proba(self,X,b_index=None):
        prob = self.decision_function(X,b_index)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict(self,X,b_index=None):
        return np.argmax(self.predict_proba(X,b_index),axis=1)

