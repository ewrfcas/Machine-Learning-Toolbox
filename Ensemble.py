import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def stacking_proba(clf,X_train,y,X_test,nfolds=5,random_seed=2017,return_score=False,
                   shuffle=True,metric='acc',clf_name='UnKnown'):
    folds = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=random_seed).get_n_splits(X_train,y)
    #return stacking_proba for train set
    train_stacking_proba=np.zeros((X_train.shape[0],np.unique(y).shape[0]))
    score=0
    for i,(train_index, validate_index) in enumerate(folds.split(X_train, y)):
        print(str(clf_name)+" folds:"+str(i+1)+"/"+str(nfolds))
        X_train_fold=X_train[train_index,:]
        y_train_fold=y[train_index]
        X_validate_fold=X_train[validate_index,:]
        y_validate_fold=y[validate_index]
        clf.fit(X_train_fold,y_train_fold)
        fold_preds=clf.predict_proba(X_validate_fold)
        train_stacking_proba[train_index,:]=fold_preds
        #validation
        fold_preds_a = np.argmax(fold_preds, axis=1)
        fold_score=len(np.nonzero(y_validate_fold - fold_preds_a == 0)[0]) / len(y_validate_fold)
        print('validate '+metric+":"+str(fold_score))
        score+=fold_score
    score/=folds
    #return stacking_proba for test set
    clf.fit(X_train,y)
    test_stacking_proba=clf.predict_proba(X_test)

    if np.unique(y).shape[0] == 2: # when binary classification only return positive class proba
        train_stacking_proba=train_stacking_proba[:,1]
        test_stacking_proba=test_stacking_proba[:,1]
    if return_score:
        return train_stacking_proba,test_stacking_proba,score
    else:
        return train_stacking_proba,test_stacking_proba

#clfs=list[clf1,clf2,clf3,...]
#clfs_name=list[name1,name2,...]
def stacking(clfs,X_train,y,X_test,nfolds=5,stage=1,random_seed=2017,shuffle=True,clfs_name=None,final_clf=None):
    for s in range(stage):
        print('stage:'+str(s+1)+'/'+str(stage))
        X_train_stack=[];X_test_stack=[]
        for i in len(clfs):
            [train_stacking_proba,test_stacking_proba]=stacking_proba(clfs[i],X_train,y,X_test,nfolds,
                                                                      random_seed,shuffle=shuffle,clf_name=clfs_name[i])
            X_train_stack.append(train_stacking_proba)
            X_test_stack.append(test_stacking_proba)
        X_train=np.concatenate(tuple(X_train_stack),axis=0)
        X_test=np.concatenate(tuple(X_test_stack),axis=0)

    #final_clf default is LogisticRegression
    if final_clf == None:
        final_clf=LogisticRegression()
        final_clf.fit(X_train,y)
    final_proba=final_clf.predict_proba(X_test)
    return final_proba

