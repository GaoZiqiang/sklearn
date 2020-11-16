import sys
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from sklearn import linear_model

filename = sys.argv[1]
# embed()
X = []
y = []
with open(filename,'r') as f:
    for line in f.readlines():
            xt,yt = [float(i) for i in line.split(',')]
            X.append(xt)
            y.append(yt)

# 数据集准备
num_training = int(0.8*len(X))
num_test = len(X) - num_training
# 用于训练的数据
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])
# 测试数据
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# 训练模型

# 创建线性回归对象
linear_regressor = linear_model.LinearRegression()
# 训练模型
linear_regressor.fit(X_train,y_train)

# 图示
y_train_pred = linear_regressor.predict(X_train)

plt.figure()
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,y_train_pred,color='red',linewidth=4)
plt.title('Training data')
plt.show()

# 测试
y_test_pred = linear_regressor.predict(X_test)
# 图示
plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,y_test_pred,color='red',linewidth=4)
plt.title('Testing data')
plt.show()

import sklearn.metrics as sm
sm.mean_squared_error(y_test,y_test_pred)
print(sm.mean_squared_error(y_test,y_test_pred))
print(round(sm.mean_squared_error(y_test, y_test_pred), 4))# 保留4位小数

# 保存模型
import pickle as pickle

output_model_file = 'saved_model.pkl'

with open(output_model_file, 'wb') as f:# Python3使用write()写入，用二进制写入'wb'
    pickle.dump(linear_regressor, f)
# 加载并使用模型
with  open(output_model_file,'rb') as f:# 同样，读出用二进制读出'rb'
    model_linregr = pickle.load(f)# 读出model并命名为model_linregr
# 使用模型
y_test_pred_new = model_linregr.predict(X_test)
print('new pred result:',sm.mean_squared_error(y_test,y_test_pred_new))

# 多项式回归器
from sklearn.preprocessing import PolynomialFeatures

# 生成多项式，将曲线的多项式的次数的初始值设置为 3
polynomial = PolynomialFeatures(degree=3)

X_train_transformed = polynomial.fit_transform(X_train)# 用作训练数据

# 下面做一组测试
# datapoint = np.arange(6).reshape(6, 1)
datapoint = [0.39,2.78,7.11]
datapoint = np.reshape(datapoint,(3,1))
# from IPython import embed
# embed()
poly_datapoint = polynomial.fit_transform(datapoint)# 对数据进行多项式处理

# 实例化poly_linear_model
poly_linear_model = linear_model.LinearRegression()
# 训练模型
poly_linear_model.fit(X_train_transformed,y_train)

# 预测
print('普通线性回归结果：',linear_regressor.predict(datapoint)[0])# 普通线性回归器
print('多项式回归结果：',poly_linear_model.predict(poly_datapoint)[0])# 多项式回归器
