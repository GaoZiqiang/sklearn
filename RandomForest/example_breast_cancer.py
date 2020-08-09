
# coding: utf-8

# In[2]:


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#导入数据集
data = load_breast_cancer()
data.data.shape
data.target.shape
# In[12]:


rfc = RandomForestClassifier(n_estimators=100,random_state=90)#rfc：经实例化后的模型
score_pre = cross_val_score(rfc,data.data,data.target,cv=10).mean()

#这里可以看到，随机森林在乳腺癌数据上的表现本就还不错，在现实数据集上，基本上不可能什么都不调就看到95%以
#上的准确率

score_pre


# In[14]:


#调整n_estimators
#1-1 使用学习曲线，对n_estimators进行调参，n_estimators是最重要的一个参数
#第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何
#引起模型整体准确率的变化
score1 = []
for i in range(0,200,10):#步长为10
    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    score1.append(score)
print(max(score1),(score1.index(max(score1))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),score1)
plt.show()


# In[21]:


#1-2 在定义好的（35-45）范围内，进一步细化学习曲线
score1 = []
for i in range(35,45):
    rfc = RandomForestClassifier(n_estimators=i,n_jobs=-1,random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    score1.append(score)
print(max(score1),([*range(35,45)][score1.index(max(score1))]))
plt.figure(figsize=[20,5])
plt.plot(range(35,45),score1)
plt.show()
#最优n_estimators = 39


# In[23]:


#2 继续优化，进行网格搜索
#调整max_depth
param_grid = {'max_depth':np.arange(1, 20, 1)}
# 一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
# 但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够）
# 更应该画出学习曲线，来观察深度对模型的影响

rfc = RandomForestClassifier(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)#rfc：实例化的模型
GS.fit(data.data,data.target)

GS.best_params_
GS.best_score_

#最优max_depth = 11
#当模型位于图像左边时，我们需要的是增加模型复杂度（增加方差，减少偏差）的选项，因此max_depth应该尽量
#大，min_samples_leaf和min_samples_split都应该尽量小。这几乎是在说明，除了max_features，我们没有任何
#参数可以调整了，因为max_depth，min_samples_leaf和min_samples_split是剪枝参数，是减小复杂度的参数。


# In[27]:


#3 使用网格搜索调整max_features
param_grid = {'max_features':np.arange(5,30,1)}
rfc = RandomForestClassifier(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)#实例化GS
GS.fit(data.data,data.target)
              
GS.best_params_
GS.best_score_

#max_features = 5


# In[30]:


#4 调整min_samples_leaf
param_grid = {'min_samples_leaf':np.arange(1,1+10,1)}

#对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20
#面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围
#如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度

rfc = RandomForestClassifier(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)

GS.best_params_
GS.best_score_
最优min-sample_leaf = 1
#

#可以看见，网格搜索返回了min_samples_leaf的最小值，并且模型整体的准确率还降低了，这和max_depth的情
#况一致，参数把模型向左推，但是模型的泛化误差上升了。在这种情况下，我们显然是不要把这个参数设置起来
#的，就让它默认就好了。


# In[33]:


#5 继续尝试min_samples_split
param_grid={'min_samples_split':np.arange(2, 2+20, 1)}
rfc = RandomForestClassifier(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)

GS.best_params_
GS.best_score_

#最优min_sample_leaf = 2
#和min_samples_leaf一样的结果，返回最小值并且模型整体的准确率降低了。


# In[35]:


#6 最后尝试一下criterion
param_grid = {'criterion':['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=39,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)

GS.best_params_
GS.best_score_


# In[37]:


#7 调整完毕，总结出模型的最佳参数
rfc = RandomForestClassifier(n_estimators=39,random_state=90)
score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
score

#经调餐参之后的score，提高了0.5%
score - score_pre

