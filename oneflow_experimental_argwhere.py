# OneFlow 算子介绍：oneflow.experimental.argwhere
# https://oneflow.readthedocs.io/en/master/experimental.html?highlight=argwhere#oneflow.experimental.argwhere

# 导入必要的库
import numpy as np
import oneflow.experimental as flow
flow.enable_eager_execution()

# argwhere 会输出张量中非零元素的位置，其格式为 flow.argwhere(输入张量，dtype)。其中dtype为optional

x = np.array([[0, 1, 0],
              [2, 0, 2]]).astype(np.float32)

# 在上张量中，非零元素为1，2，2。其对应的位置也显而易见（先行后列）：

output = flow.argwhere(flow.Tensor(x))
output