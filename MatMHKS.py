from sklearn.base import BaseEstimator
import numpy as np
import math
from numpy import linalg as LA

class MatMHKS(BaseEstimator):
    def __init__(self, penalty='l2', C=1.0, matrix_type=None,class_weight=None, random_seed=None,
                 feature_shuffle=False, max_iter=100,v0=1,b0=10**(-6),eta=0.99,min_step=0.0001,
                 multi_class='ovr',verbose=0):
        self.penalty = penalty
        self.C = C
        self.matrix_type=matrix_type
        self.random_seed=random_seed
        self.feature_shuffle=feature_shuffle
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.v0=v0
        self.b0=b0
        self.eta=eta
        self.min_step=min_step
        self.multi_class = multi_class
        self.verbose = verbose

    def reshape(self,X):
        X_matrix=[]
        for i in range(X.shape[0]):
            x = np.reshape(X[i, :], (self.matrix_type[0], self.matrix_type[1]),'C')
            x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
            x = np.asmatrix(np.concatenate((x, np.zeros((1, x.shape[1]))), axis=0))
            x[-1, -1] = 1
            X_matrix.append(x)
        return X_matrix

    def get_mat_e(self,X_matrix,y,u,v,b):
        E = np.asmatrix(np.zeros((len(X_matrix), 1)))
        for i in range(len(X_matrix)):
            x=X_matrix[i]
            E[i]=y[i]*u.T*x*v
        return E-self.I-b

    def get_mat_u(self,X_matrix,y,S,v,b):
        self.Z = np.asmatrix(np.zeros((len(X_matrix), S.shape[0])))
        for i in range(len(X_matrix)):
            x = X_matrix[i]
            self.Z[i, :] = y[i] * (x*v).T
        S=np.sum(S,axis=1)
        u=np.linalg.pinv(self.C * S * S.T + self.Z.T * self.Z) * self.Z.T * (self.I + b)
        return u

    def get_mat_v(self,X_matrix,y,S,u,b):
        if type(self.pinv) != np.matrix:
            self.Y = np.asmatrix(np.zeros((len(X_matrix), S.shape[1])))
            for i in range(len(X_matrix)):
                x = X_matrix[i]
                self.Y[i, :] = y[i] * (u.T * x)
            S = np.sum(S, axis=0)
            self.pinv=np.linalg.pinv(self.C * S.T * S + self.Y.T * self.Y)
        return self.pinv * self.Y.T * (self.I + b)

    def fun(self,X,y):
        u=np.asmatrix(np.zeros((self.matrix_type[0]+1,1)))
        v=np.asmatrix(np.ones((self.matrix_type[1]+1,1))*self.v0)
        b=np.asmatrix(np.ones((X.shape[0],1))*self.b0)
        self.I=np.asmatrix(np.ones((X.shape[0],1)))
        S=np.asmatrix(np.ones((self.matrix_type[0]+1,self.matrix_type[1]+1)))
        S[:,-1]=0;S[-1,:]=0
        X_matrix=self.reshape(X)
        iter=1
        self.pinv=None
        while iter < self.max_iter:
            if iter == 1:
                u = self.get_mat_u(X_matrix, y, S, v, b)
            v = self.get_mat_v(X_matrix, y, S, u, b)
            e = self.get_mat_e(X_matrix, y, u, v, b)
            b_next = b + self.eta * (e + abs(e))
            if LA.norm(b_next - b, 2) < self.min_step:
                break
            b = b_next
            iter += 1
        # print(iter)
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

    def shuffle(self,X):
        if self.random_seed==None:
            self.random_seed=2
        np.random.seed(self.random_seed)
        colum_shuffle=np.arange(X.shape[1])
        np.random.shuffle(colum_shuffle)
        return X[:,colum_shuffle]

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
        if self.feature_shuffle==True:
            X=self.shuffle(X)
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
                    y_temp[np.where(y==positive)[0]]=1
                    y_temp[np.where(y!=positive)[0]]=-1
                    self.u[positive],self.v[positive]=self.fun(X,y_temp)

            elif self.multi_class=='ovo':
                for c1 in range(len(labels)-1):
                    for c2 in range(c1+1,len(labels)):
                        X_c1=X[np.where(y==c1)[0],:]
                        X_c2=X[np.where(y==c2)[0],:]
                        X_temp=np.concatenate((X_c1,X_c2),axis=0)
                        y_temp=np.ones(X_temp.shape[0])
                        y_temp[X_c1.shape[0]:]=-1
                        self.u[(c1,c2)],self.v[(c1,c2)]=self.fun(X_temp,y_temp)
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
            if self.multi_class=='ovr':
                prob[i, 0] = 1 / (1 + math.exp(-1 * (self.u[0].T * X[i] * self.v[0])))
            else:
                prob[i, 0] = 1 / (1 + math.exp(-1 * (self.u[(0,1)].T * X[i] * self.v[(0,1)])))
        prob[:,1]=1-prob[:,0]
        return prob

    def sigmoid_ovo(self,X):
        prob=np.zeros((len(X),self.n_class))
        for i in range(len(X)):
            for k in self.u.keys():
                prob_temp=(1 / (1 + math.exp(-1 * (self.u[k].T * X[i] * self.v[k]))))
                prob[i,k[0]]+=prob_temp
                prob[i,k[1]]+=(1-prob_temp)
            prob[i, :] = prob[i, :] / sum(prob[i, :])
        return prob

    def predict_proba(self,X):
        if self.feature_shuffle==True:
            X=self.shuffle(X)
        X_matrix = self.reshape(X)
        if self.n_class==2:
            prob=self.sigmoid(X_matrix)
        else:
            if self.multi_class=='ovr':
                prob=self.softmax(X_matrix)
            else:
                prob=self.sigmoid_ovo(X_matrix)
        return prob

    def predict(self,X):
        prob=self.predict_proba(X)
        labels=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            labels[i]=np.argmax(prob[i,:])

        return labels+min(self.real_class)