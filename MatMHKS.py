from sklearn.base import BaseEstimator
import numpy as np
import math
from numpy import linalg as LA

class MatMHKS(BaseEstimator):
    def __init__(self, penalty='l2', C=1.0, matrix_type=None,class_weight=None, max_iter=100,
                 u0=0.5,b0=10**(-6),eta=0.99,min_step=0.0001,multi_class='ovr', verbose=0):
        self.penalty = penalty
        self.C = C
        self.matrix_type=matrix_type
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.u0=u0
        self.b0=b0
        self.eta=eta
        self.min_step=min_step
        self.multi_class = multi_class
        self.verbose = verbose
        self.n_class=2

    def reshape(self,X):
        X_matrix=[]
        for i in range(X.shape[0]):
            x = np.reshape(X[i, :], (self.matrix_type[0], self.matrix_type[1]),'F')
            x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
            x = np.asmatrix(np.concatenate((x, np.zeros((1, x.shape[1]))), axis=0))
            x[-1, -1] = 1
            X_matrix.append(x)
        return X_matrix

    def get_mat_v(self,X_matrix,y,S2,I,u,b):
        Y = np.asmatrix(np.zeros((len(X_matrix), S2.shape[0])))
        for i in range(len(X_matrix)):
            x = X_matrix[i]
            Y[i, :] = y[i] * (u.T * x)
        return (self.C * S2 + Y.T * Y).I * Y.T * (I + b)

    def get_mat_e(self,X_matrix,y,I,u,v,b):
        E = np.asmatrix(np.zeros((len(X_matrix), 1)))
        for i in range(len(X_matrix)):
            x=X_matrix[i]
            E[i]=y[i]*u.T*x*v
        return E-I-b

    def get_mat_u(self,X_matrix,y,S1,I,v,b):
        Z = np.asmatrix(np.zeros((len(X_matrix), S1.shape[0])))
        for i in range(len(X_matrix)):
            x = X_matrix[i]
            Z[i, :] = y[i] * (x*v).T
        u=(self.C * S1 + Z.T * Z).I * Z.T * (I + b)
        u[-1]=1
        return u

    def fun(self,X,y):
        u=np.asmatrix(np.ones((self.matrix_type[0]+1,1))*self.u0)
        u[-1]=1
        v=np.asmatrix(np.zeros((self.matrix_type[1]+1,1)))
        b=np.asmatrix(np.ones((X.shape[0],1))*self.b0)
        I=np.asmatrix(np.ones((X.shape[0],1)))
        S1=np.asmatrix(self.matrix_type[0]*np.eye(self.matrix_type[0]+1))
        S1[-1,-1]=1
        S2=np.asmatrix(self.matrix_type[1]*np.eye(self.matrix_type[1]+1))
        S2[-1,-1]=1
        X_matrix=self.reshape(X)
        iter=1
        while iter<self.max_iter:
            v=self.get_mat_v(X_matrix,y,S2,I,u,b)
            e=self.get_mat_e(X_matrix,y,I,u,v,b)
            b_next=b+self.eta*(e+abs(e))
            if LA.norm(b_next-b,2)<self.min_step:
                break
            else:
                b=b_next
                u=self.get_mat_u(X_matrix,y,S1,I,v,b)
            iter+=1
        return u,v

    def get_params(self, deep=False):
        """Get parameter.s"""
        params = super(BaseEstimator, self).get_params(deep=deep)
        if isinstance(self.kwargs, dict):  # if kwargs is a dict, update params accordingly
            params.update(self.kwargs)
        if params['missing'] is np.nan:
            params['missing'] = None  # sklearn doesn't handle nan. see #4725
        if not params.get('eval_metric', True):
            del params['eval_metric']  # don't give as None param to Booster
        return params

    def fit(self, X, y):
        if self.matrix_type==None:
            self.matrix_type=(1,X.shape[1])
        labels=np.unique(y)
        self.real_class=labels #save real classes
        self.n_class = len(labels)
        y=y-min(labels)
        labels = labels - min(labels)
        self.u={}
        self.v={}
        if self.n_class==2:
            y_temp = np.zeros(y.shape[0])
            y_temp[np.where(y==0)] = 1
            y_temp[np.where(y!=0)] = -1
            self.u[0], self.v[0] = self.fun(X, y_temp)
        else:
            #ovo or ovr
            if self.multi_class=='ovr':
                for positive in labels:
                    y_temp=np.zeros(y.shape[0])
                    y_temp[np.where(y==positive)]=1
                    y_temp[np.where(y!=positive)]=-1
                    self.u[positive],self.v[positive]=self.fun(X,y_temp)

            # elif self.multi_class=='ovo':
        return self

    def softmax(self,X):
        prob=np.zeros((len(X),self.n_class))
        for i in range(len(X)):
            for i_class in range(self.n_class):
                prob[i,i_class]=math.exp(self.u[i_class].T*X[i]*self.v[i_class])
            prob[i,:]=prob[i,:]/sum(prob[i,:])
        return prob

    def sigmoid(self,X):
        prob=np.zeros((len(X),2))
        for i in range(len(X)):
            prob[i,0]=1/(1+math.exp(-1*(self.u[0].T*X[i]*self.v[0])))
        prob[:,1]=1-prob[:,0]
        return prob

    def predict_proba(self,X):
        X_matrix = self.reshape(X)
        if self.n_class==2:
            prob=self.sigmoid(X_matrix)
        else:
            prob=self.softmax(X_matrix)

        return prob

    def predict(self,X):
        prob=self.predict_proba(X)
        labels=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            labels[i]=np.argmax(prob[i,:])

        return labels+min(self.real_class)