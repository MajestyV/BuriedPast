import numpy as np
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
    num_discard = 0   # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
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
    N = 8  # 虚拟核的数目，number of virtual nodes
    Q = 3  # 输入变量的维数（对于动态系统，输入变量是三维坐标，所以是3）

    # 对于器件实现而已，我们还需要定义工作区域
    voltage_range = (0,5)  # 器件工作在0V-5V之间
    # 如果我们不用交流电工作，那么我们只可能得到正激励，所以要先对动态系统进行平移到第一象限，最后再移动回去
    shift_vec = np.array([10.0,10.0,10.0])

    # 对输入进行随机权重分配，即masking
    # 第一步是先对输入进行正则化（即进行平移缩放并投影到器件工作区域）
    reformed_train = RC_masking.Normalizing(train_input,voltage_range,shift_vec_manual="True",shift_vec=shift_vec)
    reformed_predict = RC_masking.Normalizing(predict_input,voltage_range,shift_vec_manual="True",shift_vec=shift_vec)
    # 生成随机权重
    M = RC_masking.NodeWeight(Q,N)
    # print(M)

    # 权重分配
    J = np.dot(reformed_train,M)
    J_predict = np.dot(reformed_predict,M)

    print(reformed_train.shape)
    print(J.shape)

    # 通过器件性能决定激活函数，这也决定的dynamic reservoir
    # 按照比例慢慢调大参数可以有更好的拟合效果
    X = RC_reservoir.I_Taylor(J,(0.00000000e+00, -2.84475793e-01, 3.91165776e-01, 1.21843635e-00,
                                 2.09783459e-02, -5.31299752e-02, -7.96619973e-4, 9.05348756e-4))

    coef, intercept = RC_readout.LASSO(X,train_output)  # 利用LASSO回归进行输出权重优化

    network_train = np.dot(X,np.transpose(coef))+intercept  # 神经网络对训练集的输出结果

    network_train_rearranged = DS.Rearrange(network_train)  # training阶段，模型生成的数据
    t_network_train = network_train_rearranged[0]+num_predict
    x_network_train = network_train_rearranged[1]
    y_network_train = network_train_rearranged[2]
    z_network_train = network_train_rearranged[3]

    truth_train_rearranged = DS.Rearrange(train_output)  # training阶段，模型生成的数据
    t_truth_train = truth_train_rearranged[0] + num_predict
    x_truth_train = truth_train_rearranged[1]
    y_truth_train = truth_train_rearranged[2]
    z_truth_train = truth_train_rearranged[3]

    # print(train_output)

    #plt.plot(x_truth_train, z_truth_train)
    #plt.plot(x_network_train,z_network_train)

    #nrmse_training = ev.NRMSE(train_output, network_train)  # 训练阶段的nrmse
    #print('nrmse_training')
    #print(nrmse_training)


    network_prediction = RC_recurrent.Predicting([predict_input[0]],M,coef,intercept,num_predict,
                                                 voltage_range=voltage_range,shift_vec=shift_vec)

    #print(network_prediction)

    network_predict_rearranged = DS.Rearrange(network_prediction)  # predict阶段，模型生成的数据
    t_network_predict = network_predict_rearranged[0]
    x_network_predict = network_predict_rearranged[1]
    y_network_predict = network_predict_rearranged[2]
    z_network_predict = network_predict_rearranged[3]

    #plt.plot(x_network_predict,z_network_predict)

    #print(network_prediction)

    vi.Analyzing_3Dsystem(dynamic_system,network_train,network_prediction)


#print(X)

coef, intercept = RC_readout.LASSO(X,training_output)

testing_output = np.dot(X,np.transpose(coef))+intercept

predicted_output = np.dot(X_predict,np.transpose(coef))+intercept
#print(testing_output)
#print(training_output)

testing_output_rearrange = DS.Rearrange(testing_output)  # training阶段，模型生成的数据
t_testing = testing_output_rearrange[0]+1001
x_testing = testing_output_rearrange[1]
y_testing = testing_output_rearrange[2]
z_testing = testing_output_rearrange[3]

training_output_rearrange = DS.Rearrange(training_output)  # training的ground truth
t_traning = training_output_rearrange[0]+1001
x_traning = training_output_rearrange[1]
y_traning = training_output_rearrange[2]
z_traning = training_output_rearrange[3]

predicted_output_rearrange = DS.Rearrange(predicted_output)  # 模型的预测
t_predicted = predicted_output_rearrange[0]+6002
x_predicted = predicted_output_rearrange[1]
y_predicted = predicted_output_rearrange[2]
z_predicted = predicted_output_rearrange[3]

predicting_output_rearrange = DS.Rearrange(predicting_output)  # 标准数据-ground truth
t_predicting = predicting_output_rearrange[0]+6002
x_predicting = predicting_output_rearrange[1]
y_predicting = predicting_output_rearrange[2]
z_predicting = predicting_output_rearrange[3]

#plt.plot(x_testing,z_testing)
#plt.plot(x_traning,z_traning)

#print(len(predicted_output))
#print(len(predicting_output))

nrmse_training = ev.NRMSE(training_output,testing_output)
print('nrmse_training')
print(nrmse_training)

nrmse_predicting = ev.NRMSE(predicting_output,predicted_output)
print('nrmse_predicting')
print(nrmse_predicting)

#plt.plot(x_predicted,z_predicted)
#plt.plot(x_predicting,z_predicting)

#plot.GlobalSetting()  # 载入全局绘图参数

# 画X-Z相图-训练
#plot.Visulize(x_traning,z_traning,color=np.array([7,7,7])/255.0)
#plot.Visulize(x_testing,z_testing,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Training'],xlabel='X',ylabel='Z')

# 画X训练图
#plot.Visulize(t_traning,x_traning,color=np.array([7,7,7])/255.0)
#plot.Visulize(t_testing,x_testing,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Training'],xlabel='time step',ylabel='X',
                   #xlim=(min(t_traning),max(t_traning)),ylim=(-4,6))

# 画X-Z相图-预测
#plot.Visulize(x_predicting,z_predicting,color=np.array([7,7,7])/255.0)
#plot.Visulize(x_predicted,z_predicted,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Prediction'],xlabel='X',ylabel='Z')

# 画X预测图
#plot.Visulize(t_predicting,x_predicting,color=np.array([7,7,7])/255.0)
#plot.Visulize(t_predicted,x_predicted,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Prediction'],xlabel='time step',ylabel='X',
                   #xlim=(min(t_predicting),max(t_predicting)),ylim=(-5,10))