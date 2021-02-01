# 介绍 to_static 知识

import paddle
import numpy 

# 我们先简单写一个网络

class Net(paddle.nn.Layer): 
    def __init__(self): 
        super(Net, self).__init__()
        # 一个接受10维输入，输出3维的全连接层
        self.linear = paddle.nn.Linear(10, 3)
    
    # to_static装饰器能将动态图转换为静态图Program
    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 10], name='x')])
    # 接受输入张量信息，用None表示不定长的维度
    def forward(self, x): 
        return self.linear(x)

# 创建一个 shape为(1, 3, 10) 的array
a = numpy.random.randn(1, 3, 10).astype(numpy.float32)

# 初始化网络

net = Net()

# 创建一个 shape为(1, 3, 10) 的array
a = numpy.random.randn(1, 3, 10).astype(numpy.float32)

# 获取网络输出
out = net(a)

# 打印
out.numpy()

# 保存我们转换得到的静态图模型

paddle.jit.save(net, './simple_net')


