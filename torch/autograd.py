# pytorch 之 autograd

import torch

# 首先定义一个带后向的变量x

x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)
x

# 再定义一个常量c

c = torch.tensor([3.0])
c

# 变量乘常量c

y = x * c
y

# 留意y与前两者的区别

y.grad_fn
type(x.grad_fn)
type(c.grad_fn)

# 如果把表达式 y = x * c 看成是树，则x和c是叶节点，而y是内节点。pytorch对内节点和叶节点的处理是不一样的，内节点会有grad_fn属性，且后向默认不保存grad。

z = y * 2
ones = torch.ones((4,), dtype=torch.float)
z.backward(ones)

# 查看y的grad张量

y.grad # 第一次访问grad的是时候，pytorch会往stderr里输出一些提示信息说明默认不保存grad
type(y.grad)

# 如果想让后向强制保留grad信息，这需要用retain_grad
# 为了没有其他干扰，我们从头开始

x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)
y = x * c
y.retain_grad() # 强制后向保留grad信息
z = y * 2
z.backward(ones)

# 再次观察这次的y是否有grad信息

y.grad

# 符合预期!

