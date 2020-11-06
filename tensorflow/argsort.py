# tensorflow v2 api 之 tf.argsort

# 返回在指定维度上排序后的下标，形状和输入保持一致

import tensorflow

# 创建一个二维的常量 Tensor

a = tensorflow.constant([[1, 7, 3, 9], [19, 4, 7, 2], [7, 3, 5, 2]])

# 按升序在最后一个维度（二维张量中为每一行）上按升序排序，并返回排序后的下标
# 出于性能考虑，如果没有特殊需求，默认采用非稳定排序，若需要稳定排序将stable置为True

b = tensorflow.argsort(a, axis=-1, direction='ASCENDING', stable=False, name=None)

print(b)  # 返回类型一定为int32，形状和输入 Tensor 保持一致

# 按降序在第一个维度（二维张量中为每一列）上按降序排序，并返回排序后的下标

b = tensorflow.argsort(a, axis=0, direction='DESCENDING', stable=False, name=None)

print(b)

# 创建一个一维的常量 Tensor

a = tensorflow.constant([1, 10, 26.9, 2.8, 166.32, 62.3])

# 对一维 Tensor 来说，gather + argsort = sort

b = tensorflow.gather(a, tensorflow.argsort(a))

print(b)

