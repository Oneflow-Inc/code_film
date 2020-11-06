# tensorflow v2 api 之 tensorflow.boolean_mask

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # 只是关掉一些没必要的显示

import tensorflow

# 根据生成提供的布尔值的 mask 对 Tensor 做过滤 

# 输入被过滤的 Tensor

tensor = tensorflow.constant([0, 1, 2, 3])

# 输入只包含布尔值的掩码

mask = [True, False, True, False]

# 对输入 Tensor 进行过滤，只保留 mask 对应为 True 位置的元素

result = tensorflow.boolean_mask(tensor, mask) 

result.shape  # 由于下标0，2位置处的 mask 为 True，故结果只有两个元素

result # 只保留输入 Tensor 中0，2位置的元素，即[0, 2]

# 输入一个二维的 3 * 3 的 Tensor

tensor = tensorflow.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 输入对应的 mask

mask = [True, False, True]

# 默认按行进行过滤

result = tensorflow.boolean_mask(tensor, mask)

result.shape  # 只保留输入 Tensor 中第一行和第三行的元素，修改后的形状为 2 * 3

result

# 对输入张量按列进行过滤

result = tensorflow.boolean_mask(tensor, mask, axis=1)

result.shape # 只保留输入 Tensor 中第一列和第三列的元素，修改后的形状为 3 * 2

result

# 对输入 mask 的一些要求：
# 1. 0 < dim(mask) <= dim(tensor)
# 2. mask 的形状必须和 tensor 对应的前 dim(mask) 维保持一致

# 输入一个 3 * 2 的二维 Tensor

tensor = tensorflow.constant([[1, 2], [3, 4], [5, 6]])

# 因为输入 Tensor 是二维的，理论上 mask 的维度不能大于2，如果申请了一个三维的 mask

mask = [[[True, False], [True, False]]]  # 此时形状为[1, 3, 2]

result = tensorflow.boolean_mask(tensor, mask)  # dim(mask) > dim(tensor)，故报错

# 如果把 mask 的形状修改为[2]

mask = [True, False]

result = tensorflow.boolean_mask(tensor, mask)  # mask.shape[0] != tensor.shape[0]，故报错
