# This code is designed to construct an Echo State Network (ESN, as known as Reservoir Computing).

from Scripts.ASHEN import EchoStateNetwork
from GetData import Visualization

#################################################################
# 模块调用
DS = DynamicSystems.dynamic_systems()      # 调用DynamicSystems模块
ESN = EchoStateNetwork.EchoStateNetwork()  # 调用EchoStateNetwork模块
VI = Visualization.plot()                  # 调用Visualization模块
#################################################################

######################################################################################
# 定义要学习的动态系统
origin = [3.051522, 1.582542, 15.62388]  # 起点
parameter = [10.0, 29, 2.667]            # 系统参数
num_step = 5000                          # 总步数
step_length = 0.01                       # 步长
dynamic_system = DS.LorenzSystem(origin,parameter,num_step,step_length)

# VI.VisualizeDynamicSystem(dynamic_system)  # 初步画图以检视系统的有效性，不需要时可把此行注释掉

dynamic_system_rearranged = DS.Rearrange(dynamic_system)  # 数据重整化，方便后续分析跟可视化
# 对动态系统数据进行切片以得到我们的训练集跟预测集（应注意，python切片是左闭右开的，如[3:6]只包含下标为3，4，5的）
# 同时，应注意做切片时要用未重整化的数据
num_discard = 1000   # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
num_train = 1000     # 训练集长度
num_predict = 1000  # 预测集长度
# 训练集
train_start, train_end = [num_discard,num_discard+num_train]  # 训练集的起点跟终点
u_train = dynamic_system[train_start:train_end]
y_train = dynamic_system[train_start+1:train_end+1]
# 预测集
predict_start, predict_end = [num_discard+num_train,num_discard+num_train+num_predict]  # 预测集的起点跟终点
u_predict = dynamic_system[predict_start:predict_end]
y_predict = dynamic_system[predict_start+1:predict_end+1]

############################################################################################################
# 调用回声状态网络算法模型
reservoir_dim = 150  # N是水库矩库的边长，同时也就是水库态向量的长度

# 调用EchoStateNetwork模块中的函数进行动态系统的预测
result = ESN.ESN_for_DynamicSystems(u_train,y_train,reservoir_dim,num_predict,
                                    leaking_rate=0.5,
                                    reservoir_spectral_radius=0.5,
                                    input_scaling=0.3)
y_network_train, y_network_predict, W_in, W_reservoir = result  # 对结果进行解压

VI.GlobalSetting()  # 引入全局画图变量
VI.Analyzing_3Dsystem(dynamic_system[1000:3000], y_network_train, y_network_predict)