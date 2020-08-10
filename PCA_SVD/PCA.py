
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


# In[34]:


#提取数据
iris = load_iris()
X = iris.data
y = iris.target
iris.target_names#各类标签的名字


# In[24]:


X.shape
y == 0


# In[14]:


import pandas as pd
pd.DataFrame(iris.data)
pd.DataFrame(iris.target)
y


# In[31]:


#建模
#调用PCA
pca = PCA(n_components=2)#实例化 n_components=2：降为2维
pca = pca.fit(X)#拟合模型
X_dr = pca.transform(X)#获取新矩阵

X_dr
pd.DataFrame(X_dr)

#也可以fit_transform一步到位
X_dr = PCA(2).fit_transform(X)

X_dr
pd.DataFrame(X_dr)
X_dr.shape
X_dr[:,0]#第0列数据 
X_dr[y ==0,0]#这里是布尔索引
#取出label为0的所有样本中的第0列特征（第一个特征）
pd.DataFrame(X_dr[y == 0,0])
pd.DataFrame(X_dr[y == 0,1])


# In[40]:


#可视化
#要展示三中分类的分布，需要对三种鸢尾花分别绘图
plt.figure()
plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1], c="red",label=iris.target_names[0])
plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()


# In[38]:


#可视化方法2，使用循环
colors = ['red', 'black', 'orange']

iris.target_names
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0]
               ,X_dr[y == i, 1]
               alpha("=.7#改变点的透明度")
               c("=colors[i]")
               label("=iris.target_names[i]")
               )
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()


# In[45]:


#探索降维后的数据
#属性explained_variance，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
pca.explained_variance_#因为降维后还有两个特征，所以有两个可解释性方差

#属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
#又叫做可解释方差贡献率
pca.explained_variance_ratio_#大部分信息都被有效地集中在了第一个特征上

#两个新特征量的贡献率总和
pca.explained_variance_ratio_.sum()#特征降维/压缩之后，特征损失不足3%


# In[48]:


#选择最好的n_components：累积可解释方差贡献率曲线
#但我们可以使用这种输入方式来画出累计可解释方差贡献率曲线，以此选择最好的n_components的整数取值。
#累积可解释方差贡献率曲线是一条以降维后保留的特征个数为横坐标，降维后新特征矩阵捕捉到的可解释方差贡献
#率为纵坐标的曲线，能够帮助我们决定n_components最好的取值

pca_line = PCA().fit(X)#在本例子中，n_components默认为4
pca_line.explained_variance_ratio_#降维后得到4个新的特征量，4个新特征量的贡献率

import numpy as np
np.cumsum(pca_line.explained_variance_ratio_)#对4个特征量的贡献率进行累计加和

#进行曲线显示
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance")
plt.show()
#由图可得，n_components为2或3为理想取值
#选取技巧：选择曲线的转折点


# In[58]:


#最大似然估计（maximum likelihood estimation）自选超参数
#输入“mle”作为n_components的参数输入

pca_mle = PCA(n_components="mle")#数据量很大时，使用mle默认参数可能会导致运算量很大
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)

X_mle.shape#可以发现，mle为我们自动选择了n_components=3为最佳取值

pca_mle.explained_variance_ratio_.sum()


# In[67]:


#按信息量占比选超参数
#输入[0,1]之间的浮点数，并且让参数svd_solver =='full'，表示希望降维后的总解释性方差占比大于n_components
#指定的百分比，即是说，希望保留百分之多少的信息量。比如说，如果我们希望保留97%的信息量，就可以输入
#n_components = 0.97，PCA会自动选出能够让保留的信息量超过97%的特征数量。

pca_f = PCA(n_components=0.95,svd_solver="full")
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)

pca_f.explained_variance_ratio_
pca_f.explained_variance_ratio_.sum()
X_f

