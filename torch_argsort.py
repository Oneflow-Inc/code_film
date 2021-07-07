# Pytorch 算子介绍：torch.argsort
# https://pytorch.org/docs/stable/generated/torch.argsort.html?highlight=argsort#torch.argsort

# 导入Pytorch
import torch

# argsort 的作用是将输入张量按从小到大的顺序重新排列。其输入格式为 torch.argsort(输入张量，排列维度，是否改变排列顺序)
# 首先建立一个随机 4 x 4 的张量
input = torch.randn(4, 4)
input

# 当 dim=1 时，argsort会将输入张量按行横着排序。输出矩阵中每一个元素对应的是输入矩阵元素在每行的位置
output = torch.argsort(input, dim=1)
output

# 当 dim=0 时，argsort会将输入张量按列竖着排序。输出矩阵中每一个元素对应的是输入矩阵元素在每列的位置
output = torch.argsort(input, dim=0)
output

# 若想将输入张量按从大到小的顺序排列，我们可以通过指定argwhere的第三个输入来达成此目的
output = torch.argsort(input, dim=0, descending=True)
output

