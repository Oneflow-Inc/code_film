# tensorflow v2 api 之 tf.pad

import tensorflow

# 首先我们创建一个常量 Tensor 表示 pad op 的输入

origin_tensor = tensorflow.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
origin_tensor.numpy()

# 接着再创建另一个常量 Tensor 表示 padding 的大小

paddings = tensorflow.constant([[1, 1], [2, 2]])
paddings[0].numpy() # 表示上下pad各1个像素
paddings[1].numpy() # 表示左右pad各2个像素

# tensorflow pad op 提供 3 种 padding 模式, "CONSTANT", "REFLECT" 和 "SYMMETRIC"

# "CONSTANT" 模式表示在边缘处补0

constant_padded = tensorflow.pad(origin_tensor, paddings, "CONSTANT")
constant_padded.numpy() # "CONSTANT" pad 结果

# 上下是全0

constant_padded[0, :].numpy()
constant_padded[-1, :].numpy()

# 左右也是全0

constant_padded[:, 0:2].numpy()
constant_padded[:, -2:].numpy() 

# "REFLECT" 模式表示以原图像边缘一圈像素为对称中心, pad 的内容与除去边缘的图像内容呈镜像关系

reflect_paded = tensorflow.pad(origin_tensor, paddings, "REFLECT")
reflect_paded.numpy() # "REFLECT" pad 结果

# 结果第0行与第2行相等

reflect_paded[0, :].numpy()
reflect_paded[2, :].numpy() 

# 结果第4行与第2行相等

reflect_paded[-1, :].numpy()
reflect_paded[2, :].numpy() 

# 结果第0和1列 与 第3和4列 呈镜像关系

reflect_paded[:, 0:2].numpy()
reflect_paded[:, 3:5].numpy() 

# 结果第5和6列 与 第2和3列 呈镜像关系

reflect_paded[:, -2:].numpy()
reflect_paded[:, 2:4].numpy() 

# "SYMMETRIC" 模式与 "REFLECT" 类似, 区别在于是直接复制边界处的像素然后做镜像填充

symmetric_paded = tensorflow.pad(origin_tensor, paddings, "SYMMETRIC")
symmetric_paded.numpy() # "SYMMETRIC" pad 结果

# 结果第0行与第1行相等

symmetric_paded[0, :].numpy()
symmetric_paded[1, :].numpy() 

# 结果第4行与第3行相等

symmetric_paded[-1, :].numpy()
symmetric_paded[3, :].numpy()

# 结果第0和1列 与 第2和3列 呈镜像关系

symmetric_paded[:, 0:2].numpy()
symmetric_paded[:, 2:4].numpy() 

# 结果第5和6列 与 第3和4列 呈镜像关系

symmetric_paded[:, -2:].numpy()
symmetric_paded[:, 3:5].numpy() 
