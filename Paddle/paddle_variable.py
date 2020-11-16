# 关于Paddle Variable 知识
import paddle

# 虽然Paddle自身有Variable构造函数，但是Paddle官方不建议直接用，会造成错误，它针对静态图和动态图有两种构造方式

# 我们先看下静态图

# 编译器定义一个 program

cur_program = paddle.fluid.Program()

# 在当前 program 定义一个 block， 一般block包含： 1. 本地变量定义 2. 一些op

cur_block = cur_program.current_block()

# 创建一个shape=(1, 10) 的 variable

X_variable = cur_block.create_var(name="X", shape=[1, 10], dtype='float32')

# 打印shape

X_variable.shape

# 当同一个block下，variable同名，意味着它们共享参数，因此shape， dtype需要保持一致

Y_variable = cur_block.create_var(name="X", shape=[1, 10], dtype='float32')

# 此时Y_variable 和 X_variable 共享参数

# 下面我们来看下动态图

import numpy as np 

# 通过numpy先简单创建一个array

x = np.arange(10).astype(np.float32) 

# 拆解with语句，方便交互式展示

with_dygraph_guard = paddle.fluid.dygraph.guard()

# 进入动态图上下文（原本代码是`with paddle.fluid.dygraph.guard():`）

with_dygraph_guard.__enter__()

# 调用to_variable来创建变量

x_variable = paddle.fluid.dygraph.to_variable(x)

# 打印 Variable 信息

print(x_variable)

# 如果只是想知道实际数值，我们可以调用numpy()方法，只适用于动态图

x_variable.numpy()

# 退出动态图上下文
with_dygraph_guard.__exit__(None, None, None)

