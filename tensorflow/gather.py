# tensorflow v2 api 之 tensorflow.gather

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # 只是关掉一些没必要的显示

import tensorflow

# tensorflow.gather 是从输入 Tensor 的 axis 维根据 indices 的参数值获取切片

# 1. 如果输入的 indices 参数为标量

# 输入一个二维的 Tensor

tensor = tensorflow.constant([[0, 1], [2, 3], [4, 5]])

# axis 默认在第一个非 batch 维，此时 axis 为 0，按行取切片

result = tensorflow.gather(tensor, indices=1)  

result.shape  # 只保留第二行元素，故切片后的形状为[2]

result  # 切片后保留第二行元素为，[2, 3]

# 如果手动指定 axis 参数，则按列取切片

result = tensorflow.gather(tensor, indices=1, axis=1)

result.shape # 只保留第二列元素，故切片后的形状为[3]

result # 切片后保留第二列元素为，[1, 3, 5]

# 2. 如果输入的 indices 参数为一维 Tensor

# 输入一个二维的 Tensor

tensor = tensorflow.constant([[0, 1], [2, 3], [4, 5]])

# 输入一个一维的切片下标

indices = [0, 1]

# axis 默认为第一个非 batch 维，此时 axis 为 0，按行取切片

result = tensorflow.gather(tensor, indices)

result.shape  # 只保留第一行和第二行的元素，故切片后的形状为[2, 2]

# 3. 如果输入的 indices 参数为二维 Tensor 

# 输入一个二维的 Tensor

tensor = tensorflow.constant([[0, 1], [2, 3], [4, 5]])

# 输入一个高维的切片下标

indices = tensorflow.constant([[0, 1], [0, 1]])

# 设置 axis 为 0，batch 维的数量为 0

axis = 0

batch_dims = 0

result = tensorflow.gather(tensor, indices, axis=axis, batch_dims=batch_dims)

# 输出形状参考公式 tensor.shape[:axis] + indices.shape[batch_dims:] + tensor.shape[axis + 1:]

tensor.shape[:axis] # 为空 

indices.shape[batch_dims:] # [2, 2] 

tensor.shape[axis+1:] # 2

result.shape # 输出形状为，[2, 2, 2]

result  # 按行做切片后的内容为，[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]

# 如果自定义 batch 维的数量大于零，需要保证 axis >= batch_dims 

batch_dims = 1  # 如果有一个 batch 维

axis = 1

# 首先需要保证，tensor.shape[:batch_dims] = indices.shape[:batch_dims]

tensor.shape[:batch_dims]  # [3]

indices.shape[:batch_dims] # [2] 

result = tensorflow.gather(tensor, indices, batch_dims=1)  # 报错

# 调整 indices，为

indices = tensorflow.constant([[0, 1], [1, 0], [0, 1]])

tensor # 复习输入 tensor 的值为

result = tensorflow.gather(tensor, indices, axis=axis, batch_dims=batch_dims) # 重新获取，按列做切片

tensor.shape[:axis] # [3] 

indices.shape[batch_dims:] # [2] 

tensor.shape[axis+1:] # 为空 

result.shape  # 输出形状为，[3, 2]

result # 按列做切片后的内容为，[[0, 1], [3, 2], [4, 5]]

