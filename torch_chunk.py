# Pytorch 算子介绍：torch.chunk
# https://pytorch.org/docs/stable/generated/torch.chunk.html

# 导入必要的库
import torch
import numpy as np

# chunk 可以将一个矩阵分为多个小矩阵
# 我们首先随机取一个 shape = [5, 3]的矩阵

x = torch.rand(5, 3)
x

# chunk 格式为：torch.chunk(输入, 被切分后矩阵数量, 切分维度(dim = 维度))

res_list = torch.chunk(x, 5, dim=0)
[ chunk.shape for chunk in res_list] # 显示切分后的各个小张量的形状
res_list # 显示切分后的内容

# 其中，当dim = 0，chunk就会把矩阵“按行横着切”；当dim = 1时，chunk就会把矩阵“按列竖着切”
# 针对以上5行3列的矩阵。

# 以下就是按行切：
torch.chunk(x, 5, dim=0)

# 以下就是按列切：
torch.chunk(x, 5, dim=1)

# 当然，其实 dim 可以是任意维度
x = torch.rand(5, 3, 4, 8)
res_list = torch.chunk(x, 4, dim=3)
[ chunk.shape for chunk in res_list] # 显示切分后的各个小张量的形状


# 值得注意的是，chunk 方法的第二个参数(指定切分的数量)不一定需要被维度大小所整除
# 当不能整除时，最后一个张量的特定维度大小为余数
x = torch.rand(7, 3)
res_list = torch.chunk(x, 3, dim=0)
[ chunk.shape for chunk in res_list] # 显示切分后的各个小张量的形状
