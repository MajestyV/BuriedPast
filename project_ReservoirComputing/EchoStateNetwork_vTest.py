import numpy as np
from ReservoirComputing import DynamicalSystems,CreatingWeightMatrix,ActivationFunction,WeightOptimization,Visualization

########################################################################################################################
# 模块调用
DS = DynamicalSystems.dynamical_systems()  # 调用DynamicSystems模块
CWM = CreatingWeightMatrix.weight()    # 调用CreatingWeightMatrix模块
AF = ActivationFunction.ActFun()       # 调用ActivationFunction模块
WO = WeightOptimization.optimize()     # 调用WeightOptimization模块
VI = Visualization.plot()              # 调用Visualization模块
########################################################################################################################

########################################################################################################################
class ESN:
    """This code is designed to construct an Echo State Network (ESN, as known as Reservoir Computing)."""

    # 对网络参数进行初始化
    # x_train，y_train分别为输入，输出的训练数据集，应为二阶张量，维数格式为 (测试集的长度,输入/输出向量的维数)
    def __init__(self,x_train,y_train,activation,**kwargs):
        self.name = ESN

        x_shape, y_shape = [x_train.shape, y_train.shape]  # 获取输入输出的训练数据的的维度
        # 为了方便表达，我们接下来用简单的大写字母来表示各层的维数
        N, M, L = [x_shape[0], x_shape[1], y_shape[1]]  # 训练集长度，输入维数，输出维数

        # 定义ESN的基本结构：# a=1时，网络结构为最基础的ESN；a<1时，网络结构为Li-ESN
        a = kwargs['leaking_rate'] if 'leaking_rate' in kwargs else 0.5  # ESN的leaking rate

        # ESN有三个关键的层：输入，储层，以及输出；输入层与输出层的维数由我们要学习的数据决定，而储层的特性则极大影响了网络的学习能力
        # ESN的输入矩阵以及储层矩阵中的元素，即权重都是随机生成的，在此我们可以定义一开始生成的权重的范围，但是后面权重矩阵都要进行归一化处理，故影响不大
        weight_range = kwargs['weight_element_range'] if 'weight_element_range' in kwargs else (-1, 1)
        # 由于连接权重跟储层态都是随机生成的，为了方便管理，我们在一个网络中只生成一次，每次重新构建网络才会重新生产
        # 首先让我们定义储层权重，储层权重是一个随机的稀疏矩阵，更多细节详见：https://www.sciencedirect.com/science/article/pii/S1574013709000173
        K = kwargs['reservoir_dimension'] if 'reservoir_dimension' in kwargs else 100  # 定义储层维数，也是储层态向量长度
        res_den = kwargs['reservoir_density'] if 'reservoir_density' in kwargs else 0.04  # 储层矩阵密度
        rsr = kwargs['reservoir_spectral_radius'] if 'reservoir_spectral_radius' in kwargs else 0.5  # 储层矩阵的谱半径
        # 那么接下来我们正式定义储层权重矩阵（Reservoir Weight Matrix）
        W_res_init = CWM.GenSparseMatrix((K, K), weight_range, res_den).todense()  # 生成稀疏矩阵并解压
        W_res = rsr*CWM.NormalizeSpectralRadius(W_res_init)  # 对储层权重矩阵进行谱半径归一化，并根据指定谱半径重新scaling

        # 同时，我们一开始需要一个初始化的储层态来启动网络的迭代，在此，默认初始的储层态是一个零向量
        r_init = kwargs['reservoir_state_initial'] if 'reservoir_state_initial' in kwargs else np.zeros((1,K),dtype=float)

        # 同样的，让我们来定义输入层的连接权重矩阵（Input Weight Matrix)
        s_in = kwargs['input_scaling'] if 'input_scaling' in kwargs else 0.3  # 输入的缩放因子
        W_in_init = CWM.RandomWeightMatrix((M, K), weight_range)  # 初始化随机输入权重矩阵
        W_in = s_in*CWM.NormalizeMatrixElement(W_in_init)  # 对输入权重进行矩阵元素归一化，并按照指定缩放因子重新scaling

        # 有时，我们可以加入来自输出的反馈，通过输出反馈权重矩阵（Output Feedback Weight Matrix）连接
        s_fb = kwargs['feedback_scaling'] if 'feedback_scaling' in kwargs else 0.0   # 输出反馈的缩放因子
        W_fb_raw = CWM.RandomWeightMatrix((L, M), weight_range)                      # 生成随机权重矩阵
        W_fb = s_fb*CWM.NormalizeMatrixElement(W_fb_raw)                             # 进行矩阵元素归一化

        # 向我们的网络中加入噪声扰动
        self.s_noise = kwargs['noise_scaling'] if 'noise_scaling' in kwargs else 0.0  # 噪声的缩放因子


        # 将一些变量转变为实例变量，方便这个class下面的其他函数调用
        self.x_train, self.y_train, self.activation = [x_train,y_train,activation]  # 训练集输入，训练集输出，以及激活函数
        self.N, self.M, self.K, self.L = [N, M, K, L]                               # 训练集长度以及，ESN各层维数
        self.W_in, self.W_res = [W_in, W_res]                                       # 各个连接权重
        self.r_init = r_init                                                        # 初始储层态向量
        self.a = a                                                                  # leaking rate

    ####################################################################################################################
    # 训练模块
    # 此函数可以计算训练集输入生成的所有储层态
    def CalResState(self):
        N, M, K, L = [self.N, self.M, self.K, self.L]  # 训练集长度，输入维数，储层态维数，输出维数
        r_0 = self.r_init  # 初始储层态向量，为零向量
        r_train, v_train = [np.empty((N,K),dtype=float),np.empty((N,M+K),dtype=float)]  # r为训练阶段的所有储层态，v为所有储层态跟输入的串接
        for i in range(N):
            x = self.x_train[i].reshape(1,-1)  # 要将输入转换成二维数组才能进行矩阵运算
            r_1 = (1.0-self.a)*r_0+self.activation(x@self.W_in+r_0@self.W_res)  # 计算储层态
            r_train[i] = r_1.getA()[0]  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
            v_train[i] = np.hstack((r_1,x))[0]  # 将储层态与输入串接（这一步很关键！！！），并将串接向量的值记录在v_train中
            r_0 = r_1
        return r_train, v_train

    # 此函数可以利用Tikhonov regularization计算输出权重（https://en.wikipedia.org/wiki/Tikhonov_regularization），并计算网络输出
    def CalOutputWeight(self,**kwargs):
        if 'reservoir_state' in kwargs and 'concatenated_state' in kwargs:
            r_train, v_train = (kwargs['reservoir_state'],kwargs['concatenated_state'])
        else:
            r_train, v_train = self.CalResState()  # 利用CalResState()函数计算训练集输入生成的所有储层态

        # print(v_train.shape)
        W_out, threshold = WO.RIDGE(v_train, y_train, alpha=1.0)  # 利用岭回归优化连接权重
        y_train_ESN = np.dot(v_train, np.transpose(W_out)) + threshold  # 网络的输出，可用于计算各类统计指标，分析训练结果
        return W_out, threshold, y_train_ESN

    ####################################################################################################################
    # 预测模块
    def Forecasting(self,num_step,**kwargs):
        if 'entering_external_information' in kwargs:
            r_train, v_train = (kwargs['reservoir_state'], kwargs['concatenated_state'])
            W_out, threshold = (kwargs['output_connection_weight'],kwargs['threshold'])
        else:
            r_train, v_train = self.CalResState()                   # 利用CalResState()函数计算训练集输入生成的所有储层态
            W_out, threshold, y_train_ESN = self.CalOutputWeight()  # 利用CalOutputWeight()函数计算训练集输入生成的所有储层态

        Q, M, K, L = [num_step, self.M, self.K, self.L]  # 要预测的步数，输入维数，储层态维数，输出维数

        x_0 = y_train[len(y_train)-1].reshape(1,-1)  # 预测阶段的第一个输入是训练阶段的最后一个输出，要注意转换成二阶张量
        r_0 = r_train[len(r_train)-1].reshape(1,-1)  # 水库态的初始值是最后输出的水库态
        # r_predict, v_predict = [np.empty((Q,K),dtype=float), np.empty((Q,M+K),dtype=float)]  # r为预测阶段的所有储层态，v为所有储层态跟输入的串接
        y_predict_ESN = np.empty((Q,L),dtype=float)  # 创建一个二阶张量存放ESN的输出结果
        for i in range(Q):
            x_1 = x_0
            r_1 = (1.0-self.a)*r_0 + self.activation(x_1@self.W_in+r_0@self.W_res)
            v_1 = np.hstack((r_1, x_1))
            y_1 = np.dot(v_1, np.transpose(W_out))+threshold

            x_0 = y_1
            r_0 = r_1

            y_predict_ESN[i] = y_1.getA()[0]

        return y_predict_ESN

if __name__=='__main__':
    ####################################################################################################################
    # 定义要学习的动态系统
    origin = [3.051522, 1.582542, 15.62388]  # 起点
    parameter = [10.0, 29, 2.667]            # 系统参数
    num_step = 5000                          # 总步数
    step_length = 0.01                       # 步长
    dynamic = DS.LorenzSystem(origin,parameter,num_step,step_length)

    # VI.VisualizeDynamicSystem(dynamic_system)  # 初步画图以检视系统的有效性，不需要时可把此行注释掉

    dynamic_rearranged = DS.Rearrange(dynamic)  # 数据重整化，方便后续分析跟可视化
    # 对动态系统数据进行切片以得到我们的训练集跟预测集（应注意，python切片是左闭右开的，如[3:6]只包含下标为3，4，5的）
    # 同时，应注意做切片时要用未重整化的数据
    num_discard = 1000   # 前面的点可能包含初始点的信息，会是我们的拟合偏移，因此我们从一定点数之后开始取值
    num_train = 2000     # 训练集长度
    num_predict = 2000  # 预测集长度
    # 训练集
    train_start, train_end = [num_discard,num_discard+num_train]  # 训练集的起点跟终点
    x_train = dynamic[train_start:train_end]
    y_train = dynamic[train_start+1:train_end+1]
    # 预测集
    predict_start, predict_end = [num_discard+num_train,num_discard+num_train+num_predict]  # 预测集的起点跟终点
    x_predict = dynamic[predict_start:predict_end]
    y_predict = dynamic[predict_start+1:predict_end+1]

    ####################################################################################################################
    # 调用ESN
    res_dim = 150  # N是水库矩库的边长，同时也就是水库态向量的长度

    # 定义ESN网络
    ESN = ESN(x_train,y_train,np.tanh,
              leaking_rate=1,
              reservoir_dimension=res_dim,
              reservoir_spectral_radius=0.7,
              input_scaling=0.5)

    W_out, threshold, y_train_ESN = ESN.CalOutputWeight()

    y_predict_ESN = ESN.Forecasting(num_predict)

    # 调用EchoStateNetwork模块中的函数进行动态系统的预测
    # y_network_train, y_network_predict, W_in, W_reservoir = result  # 对结果进行解压

    VI.GlobalSetting()  # 引入全局画图变量
    VI.DynamicSystemApproximation(dynamic[1000:5000], y_train_ESN, y_predict_ESN)