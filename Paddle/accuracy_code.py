# 关于 Paddle accuracy 知识
import paddle

# 首先通过numpy 构造数据

import numpy as np

# 构造label， shape为(sample_number, 1)

label = np.array([[0], [1], [0]]).astype(np.int64)

# 构造logit， shape为(sample_number, 分类数目)

logit = np.array([[0.1, 0.2, 0.7], 
                  [0.1, 0.7, 0.2], 
                  [0.2, 0.7, 0.1],]).astype(np.float32)

# 拆解with语句，方便交互式展示

with_dygraph_guard = paddle.fluid.dygraph.guard()

# 进入动态图上下文（原本代码是`with paddle.fluid.dygraph.guard():`）

with_dygraph_guard.__enter__()

# 将label和logit转换成variable

label_variable = paddle.fluid.dygraph.to_variable(label)

logit_variable = paddle.fluid.dygraph.to_variable(logit)

# Accuracy算子的 input和label两个参数分别接受 预测值 和 标签

# 参数k表示当标签在预测值里的topK内，就算做是正确

acc_1 = paddle.fluid.layers.accuracy(input=logit_variable, label=label_variable, k=1)

# 当我们取 k = 1, 它只会在预测值的top1中与标签计算正确值，这里我们3个数据只预测对了1个，因此输出会是0.33

acc_1.numpy()

# 当我们取 k = 2, 它只会在预测值的top2中与标签计算正确值, 这里我们第3个数据也算预测对了，因此输出会是0.67

acc_2 = paddle.fluid.layers.accuracy(input=logit_variable, label=label_variable, k=2)

acc_2.numpy()

# 退出动态图上下文

with_dygraph_guard.__exit__(None, None, None)

