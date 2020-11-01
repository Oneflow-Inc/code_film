# tensorflow v2 api 之 tensorflow.bitcast 

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # 只是关掉一些没必要的显示

import tensorflow

# bitcast 主要定义了两种转换规则：

# 1. 当传入数据类型 type 所占位数 小于 输入张量的数据类型 T 时，产生结果的形状完成从[...] 到 [..., sizeof(T)/sizeof(type)]的扩展

# 创建 Tensor，如果用位来表示，1 = 0x00000001，256 = 0x0000100，65537 = 0x00010001

a = tensorflow.constant([1, 256, 65537], dtype=tensorflow.int32)

# 由于 int8 < int32，在内存空间不变的前提下，对输入 Tensor 按位信息进行切分

bitcast_a = tensorflow.bitcast(a, tensorflow.int8)

bitcast_a.shape  # 产生的结果 Tensor 的形状为，(3) => (3, 32 / 8)

bitcast_a  # 遵守位信息保持不变的原则，以第二行元素256为例，它被切分为[0x00, 0x01, 0x00, 0x00]，存储顺序有点像"小端法"

# 2. 当传入数据类型 type 所占位数 大于 输入张量的数据类型 T 时，产生结果的形状需要完成从 [..., sizeof(type)/sizeof(T)] 到 [...] 的转换 

# 创建 Tensor，如果用位来表示，[0x00, 0x01, 0x00, 0x00]，对输入 Tensor 按位信息进行合并

a = tensorflow.constant([0, 1, 0, 0], dtype=tensorflow.int8)

bitcast_a = tensorflow.bitcast(a, tensorflow.int32)

bitcast_a.shape  # 产生结果的形状为，(4 / (32 / 8), 32 / 8) => (1) scalar

bitcast_a  # 把 4 个 8 位元素合成一个32位的数，即 0x00000100 = 256，依然遵守"小端法"

# 如果输入 Tensor 的形状不能被 sizeof(type) / sizeof(T) 整除怎么办？

a = tensorflow.constant([0, 1, 0], dtype=tensorflow.int8)

bitcast_a = tensorflow.bitcast(a, tensorflow.int32) # 由于输入 Tensor 的形状为 (3) 不能被 sizeof(int32) / sizeof(int8) 整除，故抛出异常

