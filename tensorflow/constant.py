# tensorflow v2 api 之 tf.constant

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow

import numpy

# 通过 python list创建一个常量 Tensor 对象

constant_tensor1 = tensorflow.constant([0,1,2,3,4,5,6]) 

constant_tensor1.shape # 返回 TensorShape 类型

constant_tensor1.dtype  # 若不给定 dtype 会自动从给定值中推导

constant_tensor1.device # 设备信息

# 通过 numpy array 创建一个常量 Tensor 对象

constant_tensor2 = tensorflow.constant(numpy.array([[1, 2, 3], [4, 5, 6]]))

constant_tensor2

# 通过 scalar 创建一个常量 Tensor 对象

constant_tensor3 = tensorflow.constant(3, shape=(2, 3)) # 创建一个形状为(2, 3) 所有元素值为3的常量Tensor

constant_tensor3  # 一个所有元素值都为3的常量Tensor

# 给定 常量 Tensor 对象的数据类型

constant_tensor4 = tensorflow.constant(4, shape=(4, 1), dtype=tensorflow.float64)

constant_tensor4 # 打印出的元素值为浮点类型

