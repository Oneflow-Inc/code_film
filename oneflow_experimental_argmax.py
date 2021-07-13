# OneFlow 算子介绍：oneflow.experimental.argmax
# https://oneflow.readthedocs.io/en/master/experimental.html?highlight=argmax#oneflow.experimental.argmax

# 导入必要的库
import numpy as np
import oneflow.experimental as flow
flow.enable_eager_execution()

# 此算子输出一个维度上最大值的位置
# 首先，先建立一个 2D numpy 张量

x = np.array([[1, 3, 8, 7, 2],
              [1, 9, 4, 3, 2]], dtype=np.float32)

# argmax 的输入为: flow.argmax(输入张量，维度(dim)，是否保留输出矩阵在输入矩阵中的位置(keepdim)）
# 其中，后两项输出是 optional 的

out = flow.argmax(flow.Tensor(x))
out

# 可以看到，当我们没有指定维度时，oneflow 默认张量为 1D list(dim = -1)

out = flow.argmax(flow.Tensor(x), dim=1)
out

# dim = 1 意味着算子要横着按行找最大值
# 此处 8， 9 分别对应第一行和第二行的最大值
# 所以输出为8和9的位置[2, 1]
# 当 dim = 0 时，算子会竖着按列找最大值

out = flow.argmax(flow.Tensor(x), dim=0)
out

# 注意，当一列或一行中由若干个最大值时，argmax 只会输出第一个最大值的位置
# 第三个输入 keepdim 在没有指定之前默认为 false。若我们加上 keepdim=True 后：

out = flow.argmax(flow.Tensor(x), dim=1, keepdim=True)
out

# 可以看到，keepdim 的作用是保留每一维度最大值在输入张量中的位置

