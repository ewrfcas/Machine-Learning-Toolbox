import bayesiantests as bt
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm import tqdm
#使用csv文件路径作为输入，要求csv文件格式为第一行算法名称，二至末行为多个数据集某一评价指标的结果，输出的是各个算法性能从优到劣的排序
#使用论文Time for a Change: a Tutorial for Comparing Multiple Classiers Through Bayesian Analysis中的贝叶斯signrank性能评价方法及原始函数，github地址
# https://github.com/BayesianTestsML/tutorial/

def wrap_function(path,rope=0.01,verbose=False,names=('left_classifier', 'right_classifier')):
    x=pd.read_csv(path)
    x.dropna(axis=0, how='all', inplace=True)
    x.dropna(axis=1, how='all', inplace=True)
    try:
        names=x.columns.values
        n_classifier=len(names)
    except:
        print('Wrong input type,the algorithm takes DataFrame as standard input')
    if n_classifier==2:
        left_bigger,equal,right_bigger=bt.signrank(np.array(x),rope=rope,verbose=verbose,names=names)
    else:
        result_line=np.zeros((n_classifier,))
        result_matrix=np.zeros((n_classifier,n_classifier))
        for i in tqdm(range(0,n_classifier)):
            for j in range(0,n_classifier):
                if i==j:
                    continue
                else:
                    x_t=x.iloc[:,[i,j]]
                    names=x_t.columns.values
                    left_bigger, equal, right_bigger = bt.signrank(np.array(x_t), rope=rope, verbose=verbose, names=names)
                    if max([left_bigger,equal,right_bigger])==left_bigger:
                        result_line[i]+=1
                    result_matrix[i][j]=left_bigger
    result=pd.DataFrame(result_line,index=x.columns.values)
    result = result.sort_values(by=[0], ascending=False)
    result = result.index.values
    result_matrix=pd.DataFrame(result_matrix,index=x.columns.values,columns=x.columns.values)
    for al in result:
        print(al + ' ', end=' ')
    return result_matrix

path='./data/total_report_macc.csv'
matrix=wrap_function(path,verbose=True,rope=0.01)
matrix.to_csv('./result_matrix.csv')

sb.heatmap(matrix,annot=True,linewidths=0.5, cmap='YlGnBu')
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.savefig("bayes_heatmap_macc.png")
plt.show()