# tensorflow v2 api 之 tensorflow.Tensor 
import tensorflow

# 通过tensorflow.convert_to_tensor函数来创建Tensor 

# convert_to_tensor接受的参数类型有: 
# tensorflow.Tensor对象, numpy数组, 普通list和scalar 

tensor = tensorflow.convert_to_tensor([1, 2, 3, 4]) # 由list创建一个tensorflow.Tensor对象 
type(tensor) # tensorflow v2 默认是eager模式 

tensor.shape # 形状 

# 注意这个返回结果类型是tensorflow.TensorShape 

tensor.dtype # 数据类型 

tensor.device # 通过convert_to_tensor创建的tensor设备类型只能是cpu 

# 通过`with tensorflow.device()`语句创建设备上下文 
# 然后在该上下文内的操作都是运行在该设备上 

with tensorflow.device('GPU:0'): # 指定0号gpu 
    tensor_zero = tensorflow.zeros_like(tensor) # 创建一个全0的张量其形状与tensor一致 
    tensor_zero.device

tensor_zero.device # 离开设备上下文之后tensor_zero的设备类型依然是gpu 

