# This code is designed to construct an Echo State Network (ESN, as known as Reservoir Computing).

import numpy as np
from ReservoirComputing import DynamicalSystems,CreatingWeightMatrix,ActivationFunction,WeightOptimization,Visualization

#################################################################
# 模块调用
DS = DynamicalSystems.dynamical_systems()  # 调用DynamicSystems模块
CWM = CreatingWeightMatrix.weight()    # 调用CreatingWeightMatrix模块
AF = ActivationFunction.ActFun()       # 调用ActivationFunction模块
WO = WeightOptimization.optimize()     # 调用WeightOptimization模块
VI = Visualization.plot()              # 调用Visualization模块
#################################################################

##################################################################
class algorithm:
    """利用这类函数，我们可以轻松调用回声状态网络算法模型"""
    def __init__(self):
        self.name = algorithm

    # 此函数可以调用Echo State Network (ESN)或者是Leaky-integrator Echo State Network (Li-ESN)模型去拟合动态系统
    def ESN_for_Dynamics(self,training_input,training_output,ReservoirDimension,predicting_step,**kwargs):
        # ESN算法的参数
        a = kwargs['leaking_rate'] if 'leaking_rate' in kwargs else 0.5  # ESN的leaking rate
        # a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        rho = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs else 0.5  # 水库矩阵的谱半径
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.3  # 输入的缩放因子
        #s_fb = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0.0  # 输出反馈的缩放因子
        reservoir_density = kwargs['reservoir_density'] if 'reservoir_density' in kwargs else 0.04  # 水库权重矩阵密度

        # 要定义我们的网络，我们需要定义三个关键变量：输入，水库态（reservoir state），输出
        # 在此，我们记：x-输入向量，u-未激活的水库输入向量，r-水库态向量，v-水库态与输入串接而得的向量，y-输出向量
        N = ReservoirDimension  # N是水库矩库的边长，同时也就是水库态向量的长度
        x_train, y_train = [training_input,training_output]  # 解压训练集数据
        D_x, D_r, D_y = [len(x_train[0]), N, len(y_train[0])]  # 输入，水库态以及输出向量的维度

        # 定义输入权重矩阵（Input Weight Matrix)
        W_in_raw = CWM.RandomWeightMatrix((D_x,D_r), (-1, 1))  # 生成随机权重矩阵
        W_in = s_in*CWM.NormalizeMatrixElement(W_in_raw)        # 进行矩阵元素归一化
        # 定义水库态权重矩阵（Reservoir Weight Matrix）
        W_reservoir_raw = CWM.GenSparseMatrix((D_r, D_r), (-1, 1), reservoir_density).todense()  # 生成稀疏矩阵并解压
        W_reservoir = CWM.NormalizeSpectralRadius(W_reservoir_raw)  # 对水库态权重矩阵进行谱半径归一化
        # 定义输出反馈权重矩阵（Output Feedback Weight Matrix）
        #W_fb_raw = WM.RandomWeightMatrix((D_y, D_x), (-1, 1))  # 生成随机权重矩阵
        #W_fb = s_fb*WM.NormalizeMatrixElement(W_fb_raw)        # 进行矩阵元素归一化

        training_step = len(x_train)  # 训练集的长度
        r0 = np.zeros((1, D_r), dtype=float)  # 水库态向量初值，为零向量
        r_train = []
        v_train = []  # 水库态与输入态串接后的态
        for i in range(training_step):
            x = np.array([x_train[i]])  # 转换成二维数组才能进行矩阵运算
            r = (1.0-a)*r0+AF.tanh(s_in*(x@W_in)+rho*(r0@W_reservoir))
            r_train.append(r.getA()[0])  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
            v_train.append(np.hstack((r.getA()[0], x_train[i])))  # 将水库态与输入串接，很关键！！！
            r0 = r

        coef, intercept = WO.RIDGE(v_train, y_train, alpha=1.0)  # 利用岭回归优化连接权重

        y_network_train = np.dot(v_train, np.transpose(coef)) + intercept

        # 预测模块
        x_i = np.array([y_train[len(y_train)-1]])  # 预测阶段的第一个输入是训练阶段的最后一个输出
        r_i = np.array([r_train[len(r_train)-1]])  # 水库态的初始值是最后输出的水库态
        # v_network = []
        y_network_predict = []
        for i in range(predicting_step):
            r_p = (1.0-a)*r_i+AF.tanh(s_in*(x_i@W_in)+rho*(r_i@W_reservoir))
            v_p = np.hstack((r_p, x_i))
            y_p = np.dot(v_p, np.transpose(coef))+intercept

            x_i = y_p
            r_i = r_p

            y_network_predict.append(y_p.getA()[0])
        y_network_predict = np.array(y_network_predict)

        return y_network_train,y_network_predict,W_in,W_reservoir

    # 此函数可以调用使用了器件激活函数的ESN或是Li-ESN模型拟合动态系统，此模型被记为Analog-ESN
    # 在此函数中，器件的激活函数（device-parameterized activation function）以多项式形式输出，专用于多项式回归拟合的器件性能
    def AnalogESN_for_Dynamics_Poly(self,training_input,training_output,ReservoirDimension,predicting_step,**kwargs):
        # ESN算法的参数
        a = kwargs['leaking_rate'] if 'leaking_rate' in kwargs else 0.5  # ESN的leaking rate
        # a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        rho = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs else 0.5  # 水库矩阵的谱半径
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.3  # 输入的缩放因子
        # s_fb = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0.0  # 输出反馈的缩放因子
        reservoir_density = kwargs['reservoir_density'] if 'reservoir_density' in kwargs else 0.04  # 水库权重矩阵密度
        # Device-parameterized activation function的参数
        poly_coef = kwargs['coefficient'] if 'coefficient' in kwargs else (0,1)  # 激活函数多项式的系数，默认器件性能为y=x
        R_ref = kwargs['reference_factor'] if 'reference_factor' in kwargs else 1  # 信号转换时的参考因子，默认为1

        # 要定义我们的网络，我们需要定义三个关键变量：输入，水库态（reservoir state），输出
        # 在此，我们记：x-输入向量，u-未激活的水库输入向量，r-水库态向量，v-水库态与输入串接而得的向量，y-输出向量
        N = ReservoirDimension  # N是水库矩库的边长，同时也就是水库态向量的长度
        x_train, y_train = [training_input, training_output]  # 解压训练集数据
        D_x, D_r, D_y = [len(x_train[0]), N, len(y_train[0])]  # 输入，水库态以及输出向量的维度

        # 定义输入权重矩阵（Input Weight Matrix)
        W_in_raw = CWM.RandomWeightMatrix((D_x, D_r), (-1, 1))  # 生成随机权重矩阵
        W_in = s_in * CWM.NormalizeMatrixElement(W_in_raw)  # 进行矩阵元素归一化
        # 定义水库态权重矩阵（Reservoir Weight Matrix）
        W_reservoir_raw = CWM.GenSparseMatrix((D_r, D_r), (-1, 1), reservoir_density).todense()  # 生成稀疏矩阵并解压
        W_reservoir = CWM.NormalizeSpectralRadius(W_reservoir_raw)  # 对水库态权重矩阵进行谱半径归一化
        # 定义输出反馈权重矩阵（Output Feedback Weight Matrix）
        # W_fb_raw = WM.RandomWeightMatrix((D_y, D_x), (-1, 1))  # 生成随机权重矩阵
        # W_fb = s_fb*WM.NormalizeMatrixElement(W_fb_raw)        # 进行矩阵元素归一化

        training_step = len(x_train)  # 训练集的长度
        r0 = np.zeros((1, D_r), dtype=float)  # 水库态向量初值，为零向量
        u_train = []  # 训练部分的未激活的水库输入向量
        r_train = []
        v_train = []  # 水库态与输入态串接后的态
        for i in range(training_step):
            x = np.array([x_train[i]])  # 转换成二维数组才能进行矩阵运算
            u = s_in*(x@W_in)+rho*(r0@W_reservoir)
            r = (1.0-a)*r0+R_ref*AF.Polynomial(u,poly_coef)
            u_train.append(u.getA()[0])  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
            r_train.append(r[0])
            v_train.append(np.hstack((r[0], x_train[i])))  # 将水库态与输入串接，很关键！！！
            r0 = r

        # print(v_train)
        coef, intercept = WO.RIDGE(v_train, y_train, alpha=1.0)  # 利用岭回归优化连接权重

        y_network_train = np.dot(v_train, np.transpose(coef)) + intercept

        # 预测模块
        x_i = np.array([y_train[len(y_train) - 1]])  # 预测阶段的第一个输入是训练阶段的最后一个输出
        r_i = np.array([r_train[len(r_train) - 1]])  # 水库态的初始值是最后输出的水库态
        # v_network = []
        u_predict = []  # 训练部分的未激活的水库输入向量
        y_network_predict = []
        for i in range(predicting_step):
            u = s_in*(x_i@W_in)+rho*(r_i@W_reservoir)
            r_p = (1.0-a)*r0+R_ref*AF.Polynomial(u,poly_coef)
            v_p = np.hstack((r_p, x_i))
            y_p = np.dot(v_p,np.transpose(coef))+intercept

            x_i = y_p
            r_i = r_p

            u_predict.append(u.getA()[0])
            y_network_predict.append(y_p[0])
        y_network_predict = np.array(y_network_predict)

        u_total = np.vstack((u_train, u_predict))  # 将训练部分与预测部分的未激活水库输入串接
        print(np.array(u_train).shape,np.array(u_predict).shape,u_total.shape)  # 此行可以进行维数检验，正常情况下注释掉就好
        # 获取每一个时间步下，未激活水库输入的范围
        u_states_range = [(u_total[i].min(),u_total[i].max()) for i in range(len(u_total))]

        return y_network_train, y_network_predict, W_in, W_reservoir,u_states_range

if __name__=='__main__':
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

    # 调用神经网络 ###############################################################################################
    N = 150  # N是水库矩库的边长，同时也就是水库态向量的长度

    ESN = algorithm()

    # 调用EchoStateNetwork模块中的函数进行动态系统的预测
    result = ESN.ESN_for_Dynamics(u_train,y_train,N,num_predict,
                                        leaking_rate=0.5,
                                        reservoir_spectral_radius=0.5,
                                        input_scaling=0.3)
    y_network_train, y_network_predict, W_in, W_reservoir = result  # 对结果进行解压

    VI.GlobalSetting()  # 引入全局画图变量
    VI.DynamicSystemApproximation(dynamic_system[1000:3000], y_network_train, y_network_predict)