#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 共享单车回归
import sys
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,explained_variance_score
from housing import plot_feature_importances
from IPython import embed

# In[5]:


# CSV文件读取器
def load_dataset(filename):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    X,y = [],[]
    for row in file_reader:
        X.append(row[2:13])# 只提取2-13列的特征
        y.append(row[-1])# 提取最后一列作为label
    
    # 提取特征名称
    feature_names = np.array(X[0])# 在这里打一个断点
    
    # 将第一行特征名称移除，仅保留数据部分
    return np.array(X[1:]).astype(np.float32),np.array(y[1:]).astype(np.float32),feature_names


# In[6]:


# 读取数据并做训练
X,y,feature_names = load_dataset(sys.argv[1])# 使用python3 filename.py data_filename.csv命令读取csv文件
X,y = shuffle(X,y,random_state=7)

num_training =  int(0.9*len(X))
X_train,y_train = X[:num_training],y[:num_training]
X_test,y_test = X[num_training:],y[num_training:]
# embed()
# 参数n_estimators是指评估器（estimator）的数量，表示随机森林需要使用的决策树数量
# 参数max_depth是指每个决策树的最大深度
# 数min_samples_split是指决策树分裂一个节点需要用到的最小数据样本量
rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=2)
rf_regressor.fit(X_train,y_train)

# 评价训练效果
y_test_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test,y_test_pred)
evs = explained_variance_score(y_test,y_test_pred)
print("\n#### Random Forest regressor performance ####") 
print("Mean squared error =", round(mse, 2)) 
print("Explained variance score =", round(evs, 2))

# 画特征重要性条形图
plot_feature_importances(rf_regressor.feature_importances_,'Random Forest regressor',feature_names)


# In[ ]:




