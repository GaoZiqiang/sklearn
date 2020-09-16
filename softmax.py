# -*-coding: utf-8 -*-
 
import tensorflow as tf
import numpy as np
 
def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
    print("reshape之前的row_max的shape: ",row_max.shape)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    print("reshape之后的row_max的shape: ",row_max.shape)
    x = x - row_max
 
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s
 
 
A = [[1, 1, 5, 3],
     [0.2, 0.2, 0.5, 0.1]]
print("A的值: ",format(A))
A= np.array(A)
axis = 1  # 默认计算最后一维
 
# [1]使用自定义softmax
s1 = softmax(A, axis=axis)
print("s1:{}".format(s1))
 
 
#[2]使用TF的softmax
with tf.Session() as sess:
    tf_s2=tf.nn.softmax(A, axis=axis)
    s2=sess.run(tf_s2)
    print("s2:{}".format(s2))
