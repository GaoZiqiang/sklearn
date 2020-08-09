
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# In[2]:


boston = load_boston()


# In[12]:


boston.data.shape


# In[22]:


regressor = RandomForestRegressor(n_estimators=100,random_state=0)
cross_val_score(regressor,boston.data,boston.target,cv=10,scoring = "neg_mean_squared_error")#默认返回均方误差R^2


# In[18]:


#sklearn中模型评估指标列表（所有的模型评估指标）
import sklearn
sorted(sklearn.metrics.SCORERS.keys())

