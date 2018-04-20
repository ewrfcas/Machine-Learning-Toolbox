# aud features
import pandas as pd
import numpy as np
df_train=pd.read_csv('aud_feature_train_256d.csv')
df_val=pd.read_csv('aud_feature_val_256d.csv')
df_train.sort_values(by=['filename_1'],inplace=True)
df_val.sort_values(by=['filename_1'],inplace=True)
df_val.head()
id_train=df_train.pop('filename_1').values
id_val=df_val.pop('filename_1').values
id_train=np.array(list(map(lambda x:x.split('_')[0]+'_'+x.split('_')[-1],id_train)))
id_val=np.array(list(map(lambda x:x.split('_')[0]+'_'+x.split('_')[-1],id_val)))
X_train_aud=df_train.values
X_val_aud=df_val.values
# read label
from tqdm import tqdm
label_train=pd.read_csv('train_label.csv')
id_label_train=label_train['id'].values
y_label_train=label_train[['y1','y2']].values

label_val=pd.read_csv('val_label.csv')
id_label_val=label_val['id'].values
y_label_val=label_val[['y1','y2']].values

# id_label_train=np.concatenate((id_label_train,id_label_val))
id_label_train=np.array(list(map(lambda x:x.split('_')[0]+'_'+x.split('_')[-1],id_label_train)))
dicti={}
for i,iid in enumerate(id_label_train):
    dicti[id_label_train[i]]=y_label_train[i,:]
y_train_aud=[]
for i in tqdm(id_train):
    y_train_aud.append(dicti[i])
y_train_aud=np.array(y_train_aud)

id_label_val=np.array(list(map(lambda x:x.split('_')[0]+'_'+x.split('_')[-1],id_label_val)))
dicti={}
for i,iid in enumerate(id_label_val):
    dicti[id_label_val[i]]=y_label_val[i,:]
y_val_aud=[]
for i in tqdm(id_val):
    y_val_aud.append(dicti[i])
y_val_aud=np.array(y_val_aud)
print(X_train_aud.shape)
print(X_val_aud.shape)
print(y_train_aud.shape)
print(y_val_aud.shape)
print(id_train)
X_train=X_train_aud
X_val=X_val_aud
y_train=y_train_aud
y_val=y_val_aud
y_val[:,0]=(y_val[:,0]+1)/2

# # 按照人来进行分割
# X_person=np.unique(list(map(lambda x:x.split('_')[0],id_all)))
# print(X_person)
# np.random.seed(2018)
# np.random.shuffle(X_person)
# X_person_val=X_person[0:len(X_person)//5]
# X_person_train=X_person[len(X_person)//5:]
# train_index=[]
# val_index=[]
# from tqdm import tqdm
# for i in tqdm(range(len(X_all))):
#     if id_all[i].split('_')[0] in X_person_val:
#         val_index.append(i)
#     else:
#         train_index.append(i)
# train_index=np.array(train_index)
# val_index=np.array(val_index)
# print(train_index)
# print(val_index)

# lightgbm
import lightgbm as lgb
from metrics import calccc
from sklearn.model_selection import ParameterGrid
# 自定义ccc评价
def metric_ccc(preds,lgbdata):
    labels=lgbdata.get_label()
    ccc,_=calccc.ccc(labels,preds)
    return 'ccc value:',ccc,True

def correct(train_y, pred_val_y):
    train_std = np.std(train_y)
    val_std = np.std(pred_val_y)
    mean = np.mean(pred_val_y)
    pred_val_y = np.array(pred_val_y)
    pred_val_y = mean + (pred_val_y - mean) * train_std / val_std
    return pred_val_y

params = {
    'metric': ['metric_ccc'],
    'application': ['regression'],
    'learning_rate':[0.015],
    'feature_fraction': [0.9],
    'max_depth': [15],
    'num_leaves':[100],
    'bagging_fraction': [0.8],
    'bagging_freq':[5],
    'min_data_in_leaf':[10],
    'min_gain_to_split':[0],
    'num_iterations':[500],
    'lambda_l1':[1],
    'lambda_l2':[0.1],
    'verbose':[1]
#     'is_unbalance':[True]
}
params=list(ParameterGrid(params))
lgbtrain=lgb.Dataset(X_train,label=y_train[:,1])
lgbeval=lgb.Dataset(X_val,label=y_val[:,1],reference=lgbtrain)
best_ccc=0
for param in params:
    print(param)
    clf = lgb.cv(param, lgbtrain, nfold=5, num_boost_round=param['num_iterations'], \
                    early_stopping_rounds=50, feval=metric_ccc, verbose_eval=True)
    print(clf)
#     if clf.best_score['valid_0']['ccc value:']>best_ccc:
#         best_ccc=clf.best_score['valid_0']['ccc value:']
#         best_param=param
#         best_it=clf.best_iteration
#     print('noval best interation: '+str(clf.best_iteration))
# y_pred=clf.predict(X_val)
# y_pred2 = np.clip(correct(y_train[:,1], y_pred),-1,1)
# ccc2,_=calccc.ccc(y_val[:,1],correct(y_train[:,1], y_pred2))
# print('best ccc:',best_ccc,'(',ccc2,')')
# print('best param:',best_param)
# print('best iteration:',best_it)