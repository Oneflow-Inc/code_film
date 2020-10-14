# Paddle api 之 fluid.dygraph.grad
import paddle.fluid as fluid

# fluid.dygraph.grad 是 Paddle 动态图下获取反向传播梯度的 API

# 拆解with语句，方便交互式展示
with_dygraph_guard = fluid.dygraph.guard()

# 进入with fluid.dygraph.guard()
with_dygraph_guard.__enter__()

# 创建一个 shape为(1)值为1的张量
x = fluid.layers.ones(shape=[1], dtype='float32')

# 允许该张量反传梯度
x.stop_gradient = False

# 计算 y = x*x
y = x * x

# dx等于y对x求导数，dx = 2*x
dx_create_True = fluid.dygraph.grad(outputs=[y], # 接受输出outputs
                                    inputs=[x], # 接受输入inputs
                                    # create graph 属性表示是否创建计算过程中的反向图，若值为False，则计算过程中的反向图会释放
                                    create_graph=True, 
                                    # retain_graph 属性表示是否保留计算梯度的前向图。如果保留则可对同一张图求两次反向
                                    retain_graph=True)[0]

# 计算 z = y + dx
z = y + dx_create_True

# 进行反向传播
z.backward()

# 当create_graph = True 则 dx 创建反向图
# z = x*x + dx，此时反向传播给x的是z第一，第二项导数，x.gradient = 2*x + 2 = 4

# 返回 x 的 梯度
x.gradient()

# 下面我们看下 create_graph = False 的情况

# 创建一个 shape为(1)值为1的张量
x_2 = fluid.layers.ones(shape=[1], dtype='float32')

# 允许该张量反传梯度
x_2.stop_gradient = False

# 计算 y_2 = x_2*x_2
y_2 = x_2 * x_2

dx_create_False = fluid.dygraph.grad(outputs=[y_2], 
                                     inputs=[x_2], 
                                     create_graph=False, 
                                     retain_graph=True)[0]

# 计算 z_2 = y + dx_create_False
z_2 = y_2 + dx_create_False

# 进行反向传播
z_2.backward()
# 当create_graph = False 则 dx 不创建反向图，因此它不会反向传播给 x
# z_2 = x_2*x_2 + dx_create_False，此时反向传播给x的仅仅只有第一项导数，x.gradient = 2*x = 2

# 返回 x 的 梯度
x_2.gradient()

