# tensorflow v2 api 之 tf.ones_like

import tensorflow

# 创建一个常量 Tensor 作为 ones_like 的输入

tensor = tensorflow.constant([[1, 2, 3], [4, 5, 6]])

# 根据参数，创建对应的 ones_like 张量

output = tensorflow.ones_like(tensor)

# 输出张量的形状和输入张量保持一致

output.shape

# 输出张量内元素值都为1

output

