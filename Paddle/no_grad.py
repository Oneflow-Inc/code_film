# 关于paddle对于no_grad的小知识

import paddle.fluid.dygraph


# 拆解with语句，方便交互式展示
with_dygraph_guard = paddle.fluid.dygraph.guard()

# # 进入动态图上下文
with_dygraph_guard.__enter__()

# 创建一个 shape为(1)值为1的张量
x = paddle.fluid.layers.ones(shape=[1], dtype='float32')

# 允许该张量反传梯度
x.stop_gradient = False

# 进入no_grad上下文内， 在作用域内的计算结果, 对应变量的stop_gradient = True
paddle.fluid.dygraph.no_grad().__enter__()

# 计算 y = x*x 
y = x * x 

# 打印查看y变量的stop_gradient属性
y.stop_gradient

# 退出no_grad上下文
paddle.fluid.dygraph.no_grad().__exit__(StopIteration, None, None)

# 张量y进行反向传播
y.backward()

# 由于y是在no_grad上下文内,因此是没有梯度的
y.gradient() is None

# y没有梯度,自然不能反向传播到x,因此x也是没有梯度的
x.gradient() is None

# 退出动态图上下文
with_dygraph_guard.__exit__(None, None, None)

