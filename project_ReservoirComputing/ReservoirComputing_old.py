import copy
import numpy as np
from sklearn.linear_model import Lasso

class masking:
    """ This class of function serve as the input masking module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = masking

    # 生成每个虚拟节点的权重
    # ncol = N (number of virtual nodes, 虚拟节点的个数), nrow = Q (dimension of the input, 输入变量的维数)
    def NodeWeight(self,nrow,ncol,regulation="False"):
        raw = np.random.rand(nrow,ncol)  # 生成一个nrow行ncol列的、元素满足在0~1之间均匀分布的数组，每一个元素被抽中的概率都是相等的
        new = copy.deepcopy(raw)         # 直接赋值是对象的引用（别名），即浅拷贝，这时候改动某一个别名中的元素都会影响对象本身
                                         # 因此，要实现将数组复制并防止交叉影响，需要深拷贝
        if regulation == "True":         # 判断是否重整化
            for i in range(nrow):        # 重整化随机数矩阵，使其的值只有1跟-1
                for j in range(ncol):
                    if new[i,j] >= 0.5:
                        new[i,j] = 1.0
                    else:
                        new[i,j] = -1.0
        else:
            pass

        return new

    # 这个函数可以对训练集中的输入变量进行归一化处理，将其限定在某一个区间内，如：[Vmin,Vmax]
    # 归一化操作的作用是将我们的训练集映射到一个忆阻器可以处理的区间，后面可以通过输出矩阵的学习将其映射回去
    # training_input为训练集的输入变量，格式为：
    # [[x1(t0), x2(t0), x3(t0),...xq(t0)], ..., [x1(tn), x2(tn), x3(tn),...xq(tn)]]
    # output_range的格式为：(Vmin,Vmax)
    def Normalizing(self,training_input,output_range,shift_vec_manual="False",shift_vec=None):
        Q = len(training_input[0])               # Q为输入变量的维数
        length_training = len(training_input)    # 训练集的长度

        training_input = np.array([np.array(training_input[n]) for n in range(length_training)])  # 将训练集转化为一个二维数组，防止报错
        ceiling = training_input.max()  # 获取训练集中的最大值
        floor = training_input.min()    # 获取训练集中的最小值

        Vmin, Vmax = output_range       # 获取归一化后的范围
        normal_factor = float(Vmax-Vmin)/float(ceiling-floor)

        if shift_vec_manual == "False":                # 使用默认的平移矩阵
            shift_vec = np.zeros(Q)                    # 生成长度为Q的一维零数组
            for i in range(Q): shift_vec[i] = -floor   # 将数组中的每个值都换成训练集最小值的相反数
        else:
            pass

        training_input = np.array([(training_input[n]+shift_vec)*normal_factor for n in range(length_training)])  # 归一化

        return training_input

class reservoir:
    """ This class of function serve as the reservoir module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = reservoir

    # 这个函数可以用于生成reservoir states
    # I是原始的输入序列，是个二维数组，格式为[[I1, I2, I3,...Iq]]
    # N是虚拟核的个数，同时也是
    def GenReservoirState(self,I,Mask):
        # 定义激活函数
        def activation(x): return self.Sigmoid(x)

        Q,N = Mask.shape # 从Mask矩阵获得输入序列维数（Mask的行数）跟虚拟核个数（Mask的列数）

        I_masked = np.dot(I,Mask)  # 对输入进行权重分配

        # 对输入进行激活
        J = np.zeros((1,N), dtype=float)
        for n in range(N):
            J[0][n] = activation(I_masked[0][n])

        return J

    #  这个函数可以用于生成reservoir states
    def GenEchoState(self,I,Mask):
        # 定义激活函数
        def activation(x): return self.Sigmoid(x)

        # 定义回声函数
        def echo(x): return 1.0/x

        Q,N = Mask.shape # 从Mask矩阵获得输入序列维数（Mask的行数）跟虚拟核个数（Mask的列数）

        I_masked = np.dot(I,Mask)  # 对输入进行权重分配

        # 对输入进行激活
        I_echo = I_masked[0][0]
        J = np.zeros((1,N), dtype=float)
        for n in range(N):
            J[0][n] = activation(I_echo)
            I_echo = J[0][n]+echo(I_echo)

        return J

    # 以下是各种不同的激活函数（activation function）
    # Sigmoid function
    def Sigmoid(self,x): return 1.0/(1+np.exp(-x))

    # 通过Taylor展开拟合的器件性能,自变量V需要是浮点数或者是一维数组
    def I_Taylor(self,V,coefficient):
        degree = len(coefficient)                # 泰勒展开的阶数
        I_list = []
        for n in range(degree):
            I_list.append(coefficient[n]*V**n)   # 根据阶数，计算每一阶对函数总值的贡献
        I_mat = np.array(I_list)                 # 将列表转换为二维数组，即矩阵
        I_total = I_mat.sum(axis=0)              # 进行每列的内部求和，即按列将整个矩阵求和成一行矩阵，结果为一维数组
        return I_total

    def I_nonlinear(self,x,C0,C1,C2,C3):
        return C0+C1*x+C2*x**2+C3*x**3

class readout:
    """ This class of function serve as the read-out module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = readout

    def LinearRegression(self):
        return

    # LASSO（least absolute shrinkage and selection operator）回归
    def LASSO(self,input,output,alpha=0.025):
        input = np.array([np.array(input[n]) for n in range(len(input))])     # 确保输入是一个二维数组
        output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

        lasso = Lasso(alpha=alpha)  # 输入正则化系数
        lasso.fit(input,output)

        return lasso.coef_, lasso.intercept_

class recurrent:
    """ This class of function is designed to recurrent and predicting the future of a well-trained RC-network. """
    def __init__(self):
        self.name = recurrent
        self.mask = masking()  # 调用mask模块
        self.reservoir = reservoir()  # 调用reservoir模块

    # 这个函数可以对已知的参数的系统进行循环，从而获得系统的预测值
    # 应注意, initial_input必须是个二维数组而非向量
    def Predicting(self,initial_input,input_weight,output_weight,intercept,num_step,**kwargs):
        voltage_range = kwargs['voltage_range'] if 'voltage_range' in kwargs else (0,5)
        shift_vec = kwargs['shift_vec'] if 'shift_vec' in kwargs else np.array([10.0,10.0,10.0])

        predicting_result = []
        x0 = initial_input  # 定义输入初值
        for i in range(num_step):
            normalized_x0 = self.mask.Normalizing(x0,voltage_range,
                                                  shift_vec_manual="True",shift_vec=shift_vec)

            # 权重分配
            J = np.dot(normalized_x0, input_weight)

            # 激活函数
            X = self.reservoir.I_Taylor(J, (0.00000000e+00, -2.84475793e-01, 3.91165776e-01, 1.21843635e-00,
                                            2.09783459e-02, -5.31299752e-02, -7.96619973e-4, 9.05348756e-4))

            x1 = np.dot(X, np.transpose(output_weight))+intercept  # 神经网络对训练集的输出结果

            predicting_result.append(x1[0])  # 记录神经网络的输出结果，由于x1是二维数组，所以只需要去其中的向量
            x0 = x1  # 将x1的值赋予x0，开始循环

        return np.array(predicting_result)  # 保证输出为数组



if __name__ == '__main__':
    mask = masking()

    #a = mask.NodeWeight(10,8)

    test = [[1,4,5,6,7,7,8,-23,51,54,21],
            [43,56,12,3,5,-43,1,3,4,6,1]]

    b = mask.Normalizing(test,(0,10))
    print(b)

    #print(a[0])
    #print(a[1])
