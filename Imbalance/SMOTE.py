import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_interpolate(X, y, k=5, rate=1.0):
    '''
    :param k: neighbor
    :param X:
    :param y:
    :param rate: N(X_minority)/N(X_majority)
    :return: X_smote, y_smote
    '''
    # get minority/majority labels
    unique_labels = np.unique(y)
    assert len(unique_labels)==2
    num0 = len(np.where(y == unique_labels[0])[0])
    num1 = len(np.where(y == unique_labels[1])[0])
    if num0 > num1:
        y_minority = unique_labels[1]
        n_minority = num1
        n_majority = num0
    else:
        y_minority = unique_labels[0]
        n_minority = num0
        n_majority = num1

    # get R smote_samples for each minority
    R = round((rate * n_majority - n_minority) / n_minority)
    if R == 0:
        R += 1

    KNN = NearestNeighbors(n_neighbors=k).fit(X)
    minority_ind = np.where(y == y_minority)[0]
    k_neighbors = KNN.kneighbors(X[minority_ind,::], n_neighbors=k, return_distance=False)
    X_add = []
    y_add = []
    for i in range(k_neighbors.shape[0]):
        target_neighbor = np.random.choice(k_neighbors[i,:], size=R, replace=True)
        for t in target_neighbor:
            X_add.append(X[minority_ind[i],::] + (np.random.rand()*(X[t,::] - X[minority_ind[i],::])))
            y_add.append(y_minority)
    X_add = np.array(X_add)
    y_add = np.array(y_add)
    X_smote = np.concatenate((X, X_add), axis=0)
    y_smote = np.concatenate((y, y_add))

    return X_smote, y_smote