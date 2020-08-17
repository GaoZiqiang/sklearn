#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import *


# In[16]:


eye(4).shape#维度
eye(4).ndim#维度/轴的个数，即**维数据
arr = eye(4)
arr
arr.size#数组元素的个数
arr.dtype#一个对象，用于描述数组中的元素的数据类型
arr.itemsize#数组中每个元素的大小
arr.data
eye(4)


# In[18]:


import numpy as np


# In[62]:


a = np.arange(15)#生成15个元素的一维数组
a.ndim
a.shape
a.reshape(3,5)
b = a.reshape(3,5)
b.shape
b.dtype
type(b)
a
c = np.array([6,7,8])
b
a
c = np.array(12)#生成单个数据 array(12)
d = np.array([12])#生成一个数组 array([12])
d
a
a.ndim


# In[108]:


#创建数组
import numpy as np

#一个常见的错误，就是调用array的时候传入多个 数字参数，而不是提供单个数字的 列表类型 作为参数。
a = np.array([12,11,13])
a
#生成多维数组
b = np.array([(1,2,3),(4,5,6)])
b
b.ndim
b.shape
c = np.array([[1,2,3],[4,5,6]])
c
#可以在创建时显式指定数组的类型
c = np.array( [ [1,2], [3,4] ], dtype = complex )
c.shape
#具有初始占位符内容的数组

#函数zeros创建一个由0组成的数组
np.zeros((3,4))
#函数empty创建一个数组，其内容是随机的，取决于内存的状态
np.empty((2,3))
#为了创建数字组成的数组，NumPy提供了一个类似于range的函数，该函数返回数组而不是列表。
a = np.arange(15)#生成15个元素的一维数组，该15个元素为0-14
a
b = np.arange(10,20,5)#设置开始结束点，设置步长，包含前面，不包含后面
b
c = np.arange(0,2,0.5)
c
a
c.sum()
c.max()


# In[80]:


#打印数组

print(np.arange(10000))


# In[96]:


#基本操作
a = array((1,2,3))
b = np.arange(0,15,5)
a*b
b - a
np.sin(a)

#矩阵相乘、相加
A = np.array([[1,2],[3,4]])
A
B = np.array([[2,0],[3,4]])
A*B
A + B


# In[109]:


#随机数
a = np.random.random((2,3))
a
#因为sum和max、min等不是基本属性，所以需要用函数来实现
a.sum()
a.max()
a.min()


# In[118]:


#axis
a = np.arange(12).reshape(3,4)
a
a.sum(axis=0)#每列元素求和
a.sum(axis=1)#每行元素求和
a.max(axis=1)
a.max(axis=0)
a.cumsum(axis=1)


# In[137]:


#索引、切片和迭代
a = np.arange(10)**3
a.shape
a[2:5]
a[:6:2] = -500#equivalent to a[0:6:2] = -500; from start to position 6, exclusive, set every 2nd element to -1000
a
a[0:6:2] = -100
a[::-1]#倒置a
for i in a:
    print(i**(1/3.))
a


# In[145]:


#索引
b = array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b
b[1,1]
b[:,1]#所有行的第一个元素
b[1,:]#所有列的第一个元素
b[1:3,:]#所有列的第1和2行数据1:3取1和2
b[-1,:]#-1是最后一行


# In[149]:


#迭代
#对多维数组进行 迭代（Iterating） 是相对于第一个轴完成的：
b = array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b
for row in b:
    print(row)

#如果想要对数组中的每个元素执行操作，可以使用flat属性，该属性是数组的所有元素的迭代器
for element in b.flat:
    print(element)


# In[169]:


#改变数组形状
b = array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b.ravel()#平铺
b.shape
c = b.reshape(4,5)
c.T#转置

