# tensorflow v2 api 之 tensorflow.eye 

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # 只是关掉一些没必要的显示

import tensorflow

# 生成一个形状为2 * 2的单位矩阵

a = tensorflow.eye(num_rows=2)

a.shape  # Tensor 的形状为 2 * 2

a.dtype  # Tensor 的类型默认为 float32

a  # 除主对角线上元素为1外，其他元素都为0

# 生成一个形状为2 * 3的二维 Tensor，所有(i, i)位置上的元素都为1

a = tensorflow.eye(num_rows=2, num_columns=3)

a.shape  # Tensor 的形状为 2 * 3

a.dtype  # Tensor 使用默认类型float32

a  # 除(i, i), i 属于 [0, num_rows) 外，其他元素都为0

# 生成一个形状为4 * 3的二维 Tensor，所有(i, i)位置上的元素都为1

a = tensorflow.eye(num_rows=4, num_columns=3)

a.shape  # Tensor 的形状为 4 * 3

a.dtype  # Tensor 使用默认类型float32

a  # 除(i, i), i 属于 [0, num_columns) 外，其他元素都为0

# 生成一个带 batch 维度的 Tensor，其中在每个batch上，都为单位矩阵

a = tensorflow.eye(num_rows=2, batch_shape=[2, 3], dtype=tensorflow.int32)

a.shape # Tensor 的形状为 2 * 3 * 2 * 2

a.dtype # 数据类型使用指定的int32类型

a  # 每个小batch上都是一个形状为2 * 2的单位矩阵

