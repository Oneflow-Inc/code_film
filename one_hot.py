# Pytorch 算子介绍：torch.nn.functional.one_hot
# https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html?highlight=one_hot#torch.nn.functional.one_hot

# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F

# one_hot 可以将一个 1D 张量分类成一个 2D 独热矩阵
# 不明白独热矩阵没有关系，让我们先看看 one_hot 输出张什么样子

# 首先建立一个 1D 张量
x = torch.tensor([3,1,0,2,4], dtype = torch.int64) # one_hot只接受 64bit 的张量
x

# 进行独热编译
output = F.one_hot(x)
output

# 其中，这个矩阵的每一行依次按顺序代表[0, 1, 2, 3, 4]，而每一列代表的是这五个数的出场顺序
# 比如，3 是张量中的第一个数，所以 3 所对应的列（第四列）的第一行是 “1”
# 再比如，0 是张量中第三个数，所以 0 所对应的列（第一列）的第三行是 “1”

# 稍微复杂点的例子
x = torch.arange(0, 8) # 建立一个规律的 1D 张量
x

# 在括号内进行数学运算
output = F.one_hot(x % 3)

# 余数计算后，括号内张量 = [0, 1, 2, 0, 1, 2, 0, 1]
# 所以结果为：
output

# one_hot的功能不限于此，其完整的输入为 F.one_hot(输入张量，输出形状（num_classes)
# 例如：
output = F.one_hot(x % 3, num_classes = 5)
output
# 可以看到，输出的形状从 (8, 3) 变成了 (8, 5)
# 所以，num_classes 的作用就是调整输出张量的列数，或者说 “需要被分类的数量”

# 当然，我们也可以通过切割输入张量来达到切割输出张量的效果
output = F.one_hot(x.view(4, 2) % 3)
output

