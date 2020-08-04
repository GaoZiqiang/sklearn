
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# In[4]:


wine = load_wine()


# In[11]:


wine.data#数据


# In[8]:


wine.target#标签


# In[10]:


import pandas as pd
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)#表格展示


# In[13]:


wine.feature_names#特征的名字


# In[14]:


wine.target_names#标签分类


# In[15]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)#数据是随机划分的


# In[16]:


Xtrain.shape


# In[17]:


Xtest.shape


# In[20]:


Ytrain.shape


# In[19]:


Ytest


# In[38]:


#建立模型
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random")#DecisionTreeClassifier具有随机性，clf = classfier，random_state，splitter控制随机性
clf = clf.fit(Xtrain,Ytrain)#clf实例化
score = clf.score(Xtest,Ytest)#返回预测的准确度accuracy


# In[39]:


score


# In[41]:


#import graphviz


# In[41]:


#import graphviz


# In[27]:


clf.feature_importances_#特征贡献度

