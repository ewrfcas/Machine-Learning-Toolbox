# Machine-Learning-Toolbox

## 1.DataToolBox
### (1) Metric:
acc=get_acc(real_label,predict_label)

auc=get_auc(real_label, scores)

[tpr,tnr,macc]=get_macc(real_label, predict_label)

### (2) Re-sampling:
[data_much,data_less,much_label,less_label]=divide_data(data, label)

Random Under-Sampling:
[train_data_temp,train_label_temp]=RUS(data_much, data_less, much_label, less_label)

Random Over-Sampling:
[train_data_temp,train_label_temp]=ROS(data_much, data_less, much_label, less_label)

## 2.Ensemble
### (1) Stacking
stacking(clfs,X_train,y,X_test,nfolds=5,stage=1,random_seed=2017,shuffle=True,clfs_name=None,final_clf=None)

## 3. MatMHKS
A matrix based linear classifier

clf=MatMHKS(penalty='l2', C=1.0, matrix_type=None,class_weight=None, max_iter=100,u0=0.5,b0=10**(-6),eta=0.99,min_step=0.0001,multi_class='ovr', verbose=0)

clf.fit(X,y)

clf.predict(X)

clf.predict_proba(X)
