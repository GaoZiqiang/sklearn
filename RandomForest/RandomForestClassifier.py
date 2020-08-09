
# coding: utf-8

# In[4]:


get_ipython().magic('matplotlib inline')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine


# In[9]:


#导入需要的数据集
wine = load_wine()
wine.data.shape
wine.target.shape


# In[53]:


from sklearn.model_selection import train_test_split
#训练集与测试集必须分开导入
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

#实例化
#决策树
clf = DecisionTreeClassifier(random_state=0)
#随机森林
rfc = RandomForestClassifier(n_estimators=25)

#将训练集导入模型进行训练
clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)

#将测试集导入，进行测试
score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)

#输出测试得分
print("Single Tree:{}".format(score_c)
      ,"Random Forest:{}".format(score_r))


# In[57]:


#带大家复习一下交叉验证
#交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#做10次交叉验证
rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10)#cv=10 进行10次交叉验证
clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf,wine.data,wine.target,cv=10)

plt.plot(range(1,11),rfc_s,label = "RandomForest")
plt.plot(range(1,11),clf_s,label = "Decision Tree")
plt.legend()#显示图例
plt.show()


# In[56]:





# In[60]:


#做100次交叉验证
rfc_1 = []
clf_1 = []

for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
    clf_l.append(clf_s)
    
plt.plot(range(1,11),rfc_l,label = "Random Forest")
plt.plot(range(1,11),clf_l,label = "Decision Tree")
plt.legend()
plt.show()


# In[ ]:


# n_estimators的学习曲线
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

superpa = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.plot(range(1,201),superpa)
plt.show()

