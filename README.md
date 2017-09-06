# Machine-Learning-Toolbox

##1.DataToolBox
###(1) Metric:
acc=get_acc(real_label,predict_label)

auc=get_auc(real_label, scores)

[tpr,tnr,macc]=get_macc(real_label, predict_label)

###(2) Re-sampling:
[data_much,data_less,much_label,less_label]=divide_data(data, label)

Random Under-Sampling:
[train_data_temp,train_label_temp]=RUS(data_much, data_less, much_label, less_label)

Random Over-Sampling:
[train_data_temp,train_label_temp]=ROS(data_much, data_less, much_label, less_label)

##2.Ensemble
###(1) Stacking
