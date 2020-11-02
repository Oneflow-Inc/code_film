# tensorflow v2 api 之 tensorflow.convert_to_tensor 

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # 只是关掉一些没必要的显示

import tensorflow

# 通过tensorflow.convert_to_tensor函数来创建Tensor 

# convert_to_tensor接受的参数类型有: 
# tensorflow.Tensor对象, numpy数组, 普通list和scalar 

tensor = tensorflow.convert_to_tensor([1, 2, 3, 4]) # 由list创建一个tensorflow.Tensor对象 

type(tensor) # tensorflow v2 默认是eager模式 

tensor.shape # 形状 

# 注意这个返回结果类型是tensorflow.TensorShape 

tensor.dtype # 未指定数据类型，会根据传入数据自己推导类型，故为int32 

tensor.device # 通过convert_to_tensor创建的tensor设备类型只能是cpu 

# 由API调用者显式指定创建 Tensor 的数据类型

tensor = tensorflow.convert_to_tensor([1, 2, 3, 4], dtype=tensorflow.float32)

tensor.dtype  # 数据类型使用指定的float32类型

# API调用者也不是很明确自己想创建的 Tensor 是什么类型，以提示的形式给出一个类似的偏好设置
# 注意：当同时设置dtype和dtype_hint，会忽略 dtype_hint 参数

tensor = tensorflow.convert_to_tensor([1, 2, 3, 4], dtype_hint=tensorflow.float32)  

tensor.dtype  # 偏好设置的数据类型可行，故使用float32

# 当调用者传入的偏好数据类型不可行时，会直接忽略此设置

tensor = tensorflow.convert_to_tensor([5.1, 4.1, 3.7, 3.9], dtype_hint=tensorflow.int8)

tensor.dtype  # 浮点类型不能转换为int8，故类型自动推断为float32

# 什么时候会将偏好数据类型判定为不可行，实验感觉从低精度到高精度可行，高精度到低精度不可行

tensor = tensorflow.convert_to_tensor([5.1, 4.1, 3.5, 3.4], dtype_hint=tensorflow.int32)

tensor.dtype # 不能完成从浮点数到int32的转换，判定为不可行，数据类型仍为float32

# 如果使用dtype，强制执行从高精度向低精度转换，会直接报错

tensor = tensorflow.convert_to_tensor([5.1, 4.1, 3.5, 3.4], dtype=tensorflow.int8)

