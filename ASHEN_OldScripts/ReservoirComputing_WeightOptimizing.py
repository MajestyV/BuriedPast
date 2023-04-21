import numpy as np
import matplotlib.pyplot as plt
from Scripts.ASHEN import Visualization

# 调用Reservoir Computing算法所需的各种模块
RC_masking = ReservoirComputing.masking()
RC_reservoir = ReservoirComputing.reservoir()
RC_readout = ReservoirComputing.readout()
RC_recurrent = ReservoirComputing.recurrent()
# 调用动态系统生成模块，此模块可以提供一些用于学习的动态系统，如洛伦兹函数，蔡氏电路等
DS = DynamicSystems.dynamic_systems()
# 调用误差衡量模块
ev = Evaluation.evaluate()
# 画图模块
vi = Visualization.plot()

# 主函数
if __name__ == '__main__':
    # 首先定义要学习的动态系统
    origin = [3.051522, 1.582542, 15.62388]  # 起点
    parameter = [10.0, 29, 2.667]            # 系统参数
    num_step = 20000                         # 步数
    step_length = 0.005                      # 步长
    dynamic_system = DS.LorenzSystem(origin,parameter,num_step,step_length)

    # 初步画图以检视系统的有效性，不需要时可把此行注释掉
    #vi.VisualizeDynamicSystem(dynamic_system)

    dynamic_system_rearranged = DS.Rearrange(dynamic_system)  # 数据重整化，方便后续分析跟可视化
    # 对动态系统数据进行切片以得到我们的训练集跟预测集（应注意，python切片是左闭右开的，如[3:6]只包含下标为3，4，5的）
    # 同时，应注意做切片时要用未重整化的数据
    num_discard = 3000   # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 5000     # 训练集长度
    num_predict = 10000  # 预测集长度
    # 训练集
    train_start, training_end = [num_discard,num_discard+num_train]  # 训练集的起点跟终点
    train_input = dynamic_system[train_start:training_end]
    train_output = dynamic_system[train_start+1:training_end+1]
    # 预测集
    predict_start, predicting_end = [num_discard+num_train,num_discard+num_train+num_predict]  # 预测集的起点跟终点
    predict_input = dynamic_system[predict_start:predicting_end]
    predict_output = dynamic_system[predict_start+1:predicting_end+1]

    # 定义RC神经网络
    N = 7  # 虚拟核的数目，number of virtual nodes
    Q = 3  # 输入变量的维数（对于动态系统，输入变量是三维坐标，所以是3）

    # 对于器件实现而已，我们还需要定义工作区域
    voltage_range = (0,3)  # 器件工作在0V-5V之间
    # 如果我们不用交流电工作，那么我们只可能得到正激励，所以要先对动态系统进行平移到第一象限，最后再移动回去
    shift_vec = np.array([0.0,0.0,0.0])

    # 对输入进行随机权重分配，即masking
    # 第一步是先对输入进行正则化（即进行平移缩放并投影到器件工作区域）
    reformed_train = RC_masking.Normalizing(train_input,voltage_range,shift_vec_manual="True",shift_vec=shift_vec)
    # reformed_predict = RC_masking.Normalizing(predict_input,voltage_range,shift_vec_manual="True",shift_vec=shift_vec)
    # 生成随机权重
    M = RC_masking.NodeWeight(Q,N)
    print(M)
    #M = np.array([[0.32912517,0.85181484,0.69611565,0.28368083,0.36596467,0.34646949,
                   #0.75406652,0.25337251,0.46326842,0.43276474,0.75487091,0.42961079],
                  #[0.76431846,0.1556842,0.02460985,0.9781836 ,0.23624687,0.21757024,
                   #0.41944901,0.17576837,0.36111495,0.10615589,0.31030393,0.14327974],
                  #[0.15371693,0.30399001,0.0998681,0.37201952,0.89950967,0.3081344,
                   #0.25899755,0.74119069,0.45263018,0.56811149,0.84339703,0.74359082]])

    # 训练阶段
    J_train = []
    #print(reformed_train)
    for i in range(num_train):
        I_normalized = [reformed_train[i]]  # GenReservoirState函数的输入必须是二维数组
        # print(I_normalized)
        J_train.append(RC_reservoir.GenReservoirState(I_normalized,M)[0])  # 同时输出也是二维数组，所以取第一项就好
    J_train = np.array(J_train)
    #print(J_train.shape)
    #print(J_train)

    Wout, S = RC_readout.LASSO(J_train,train_output)  # 利用LASSO回归进行输出权重优化

    print(Wout)

    network_train = np.dot(J_train,np.transpose(Wout))+S  # 神经网络对训练集的输出结果

    nrmse_training = ev.NRMSE(train_output, network_train)  # 训练阶段的nrmse
    print('nrmse_training')
    print(nrmse_training)

    # 权重优化
    initial_train = reformed_train[0]  # 训练集的第一个点
    I_normalized = np.array([initial_train])  # 转换为二维数组
    print(I_normalized)
    network_train_new = []
    for i in range(num_predict):
        J = RC_reservoir.GenReservoirState(I_normalized, M)
        K = np.dot(J, np.transpose(Wout)) + S  # 获得此时刻的输出
        # print(K)
        network_train_new.append(K[0])  # 同时输出也是二维数组，所以取第一项就好

        I_normalized = RC_masking.Normalizing(K,voltage_range,shift_vec_manual="True",shift_vec=shift_vec)
        print(I_normalized)
        # 此时刻的输出即为下一时刻的输入
    network_train_new = np.array(network_train_new)

    Wout, S = RC_readout.LASSO(J_train, train_output)  # 利用LASSO回归进行输出权重优化

    # 预测阶段
    initial_input = network_train[len(network_train)-1]  # 初始输入是训练集的最后一个点
    I_normalized = np.array([initial_input])  # 转换为二维数组
    print(I_normalized)
    network_predict = []
    for i in range(num_predict):
        J = RC_reservoir.GenReservoirState(I_normalized, M)
        K = np.dot(J, np.transpose(Wout))+S  # 获得此时刻的输出
        # print(K)
        network_predict.append(K[0])  # 同时输出也是二维数组，所以取第一项就好
        I_normalized = K  # 此时刻的输出即为下一时刻的输入
    network_predict = np.array(network_predict)

    #print(network_predict)
    #print(J_predict.shape)
    #print(J_predict)

    #Wout, S = RC_readout.LASSO(J_train, train_output)  # 利用LASSO回归进行输出权重优化

    #network_train = np.dot(J_train, np.transpose(Wout))+S  # 神经网络对训练集的输出结果


    network_train_rearranged = DS.Rearrange(network_train)  # training阶段，将标准的动态系统点输入模型得到的数据
    t_network_train = network_train_rearranged[0]+num_predict
    x_network_train = network_train_rearranged[1]
    y_network_train = network_train_rearranged[2]
    z_network_train = network_train_rearranged[3]

    network_new_rearranged = DS.Rearrange(network_train_new)  # training阶段，模型生成的数据
    t_network_new = network_new_rearranged[0] + num_predict
    x_network_new = network_new_rearranged[1]
    y_network_new = network_new_rearranged[2]
    z_network_new = network_new_rearranged[3]

    truth_train_rearranged = DS.Rearrange(train_output)  # training阶段，模型生成的数据
    t_truth_train = truth_train_rearranged[0] + num_predict
    x_truth_train = truth_train_rearranged[1]
    y_truth_train = truth_train_rearranged[2]
    z_truth_train = truth_train_rearranged[3]

    plt.plot(x_truth_train,z_truth_train)
    plt.plot(x_network_train,z_network_train)
    plt.plot(x_network_new,z_network_new)

    # print(train_output)

    #plt.plot(x_truth_train, z_truth_train)
    #plt.plot(x_network_train,z_network_train)

    #nrmse_training = ev.NRMSE(train_output, network_train)  # 训练阶段的nrmse
    #print('nrmse_training')
    #print(nrmse_training)


    #network_prediction = RC_recurrent.Predicting([predict_input[0]],M,coef,intercept,num_predict,
                                                 #voltage_range=voltage_range,shift_vec=shift_vec)

    #print(network_prediction)

    network_predict_rearranged = DS.Rearrange(network_predict)  # predict阶段，模型生成的数据
    t_network_predict = network_predict_rearranged[0]
    x_network_predict = network_predict_rearranged[1]
    y_network_predict = network_predict_rearranged[2]
    z_network_predict = network_predict_rearranged[3]

    #plt.plot(x_network_predict,z_network_predict)

    #print(network_prediction)

    # vi.Analyzing_3Dsystem(dynamic_system[3001:19001],network_train,network_predict)