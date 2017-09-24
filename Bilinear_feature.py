import numpy as np

#feature1:n1*n2,feature2:n1*n3,return bilinear_feature:n1*(n2*n3)
def bilinear_fun(feature1,feature2,method='out_product'):
    bilinear_feature=np.zeros((feature1.shape[0],feature1.shape[1]*feature2.shape[1]))
    if method=='out_product':
        for i in range(feature1.shape[0]):
            f1=np.reshape(feature1[i,:],(-1,1))
            f2=np.reshape(feature2[i,:],(1,-1))
            bilinear_feature[i,:]=np.reshape(np.dot(f1,f2),(1,-1))[0]
    else:
        print('Can\'t do '+method)
    return bilinear_feature