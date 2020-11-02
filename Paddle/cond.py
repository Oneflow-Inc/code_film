# 讲解 paddle.fluid.layers.cond 知识

import paddle 

import numpy as np 

# 简单创建两个array

# 创建一个值为1的array

a = np.array([1]).astype(np.float32)

# 创建一个值为2的array

b = np.array([2]).astype(np.float32)

# 拆解with语句，方便交互式展示

with_dygraph_guard = paddle.fluid.dygraph.guard()

# 进入动态图上下文（原本代码是`with paddle.fluid.dygraph.guard():`）

with_dygraph_guard.__enter__()

# 将array转化成variable

a_var = paddle.fluid.dygraph.to_variable(a)

b_var = paddle.fluid.dygraph.to_variable(b)

# cond 第一个参数是一个bool表达式，若为True，则执行true_fn， 反之则执行false_fn()

c_1 = paddle.fluid.layers.cond(a_var < b_var, true_fn=lambda: a_var+b_var, false_fn=lambda:a_var-b_var)

# 此时a < b 为True，因此执行a+b

c_1.numpy()

# 我们让第一个参数bool表达式为False，让其执行false_fn()

c_2 = paddle.fluid.layers.cond(a_var > b_var, true_fn=lambda: a_var+b_var, false_fn=lambda:a_var-b_var)

c_2.numpy()

# 同时该算子也支持true/false_fn设置为None，若执行，返回值将为None

c_3 = paddle.fluid.layers.cond(a_var > b_var, true_fn=lambda: a_var+b_var, false_fn=None)

c_3

# 退出动态图上下文

with_dygraph_guard.__exit__(None, None, None)

