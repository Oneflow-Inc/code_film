# 讲解 Split 知识

import paddle 

# 创建一个大小为 (4, 16) 的全1张量

one = paddle.fluid.layers.ones(shape=(4, 16), dtype="float32")

# Split 算子是在特定的维度上将输入Tensor分割，得到一系列子Tensor

# 属性 num_or_sections 表示划分数量，支持输入int或者一个List

# 属性 dim 指定分割的维度

# num_or_sections 是整数的情况

x_1, x_2 = paddle.fluid.layers.split(one, num_or_sections=2, dim=1)

# 我们在维度1，拆分成2个子Tensor，因此每个Tensor的shape应为 4, 8

x_1.shape, x_2.shape

# num_or_sections 为列表的情况。列表中每个元素代表每个子Tensor分割维度的大小

x_3, x_4, x_5 = paddle.fluid.layers.split(one, num_or_sections=[2, 4, 10], dim=1)

# 我们将one张量，切分成形为(4, 2), (4, 4), (4, 10)

x_3.shape, x_4.shape, x_5.shape

# 列表中允许有一个元素值为-1， 框架会自动推导出对应的维度大小

x_6, x_7 = paddle.fluid.layers.split(one, num_or_sections=[10, -1], dim=1)

# 我们将one张量，切分成形为(4, 10), (4, 6)

x_6.shape, x_7.shape

