
# coding: utf-8

# In[1]:


#DecisionTreeRegressor 决策树-决策回归树
#决策回归树的几乎所有参数，属性以及接口都和分类树一模一样，不同的是，回归是连续性变量


# In[2]:


#重要参数，接口以及属性


# In[3]:


#交叉验证
#交叉验证是用来观察模型的稳定性的一种方法


# In[9]:


from sklearn.datasets import load_boston#波士顿房价数据集
from sklearn.model_selection import cross_val_score#交叉验证
from sklearn.tree import DecisionTreeRegressor


# In[10]:


boston = load_boston()


# In[29]:


regressor = DecisionTreeRegressor(random_state = 0)#实例化
#参数解释
#regressor：实例化好的模型 boston.data：不需要划分训练集和测试集的数据矩阵 boston.target：数据标签，不需要划分训练集和测试集
#cv：cross_val，进行10次交叉验证，取1份为测试集，剩下9份为训练集。cv默认为5
#scoring = "neg_mean_squared_error"：使用neg_mean_squared_error负的均方误差来评估我的模型。该参数默认返回R^2，R^2越接近1越好
cross_val_score(regressor,boston.data,boston.target,cv = 10,scoring = "neg_mean_squared_error")#交叉验证


# In[13]:


boston.data


# In[14]:


boston.target

