# Pytorch 教学：torch.chunk

#导入必要的库
import torch
import numpy as np

# chunk 可以将一个矩阵分为多个小矩阵
# 我们首先随机取一个 shape = [5, 3]的矩阵

x = torch.rand(5, 3)
x

# chunk 格式为：torch.chunk(输入, 被切分后矩阵数量, 切分维度(dim = 维度))

torch.chunk(x, 5, dim=0)

# 其中，当dim = 0，chunk就会把矩阵“横着切”；当dim = 1时，chunk就会把矩阵“竖着切”
# 最开始我们已经随机取了一个5行3列的矩阵。若我们想要将矩阵以行分开的话,上面这行代码(dim=0)即可做到

# 当然，若想以列分开：

torch.chunk(x, 3, dim=1)

# 值得注意的是，chunk函数中的第二个输入(被切分后矩阵数量)不一定需要被矩阵形状所整除

torch.chunk(x, 3, dim=0)

# 我们可以看到，chunk会尽可能均分输入矩阵，但最后一个矩阵的大小为 矩阵形状除以被切分矩阵数量 的余数
