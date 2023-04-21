import copy
import numpy as np
from sklearn.linear_model import Lasso

class masking:
    """ This class of function serve as the input masking module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = masking

    # 生成每个虚拟节点的权重
    # ncol = N (number of virtual nodes, 虚拟节点的个数), nrow = Q (dimension of the input, 输入变量的维数)
    def NodeWeight(self,nrow,ncol):
        raw = np.random.rand(nrow,ncol)  # 生成一个nrow行ncol列的、元素满足在0~1之间均匀分布的数组，每一个元素被抽中的概率都是相等的
        new = copy.deepcopy(raw)         # 直接赋值是对象的引用（别名），即浅拷贝，这时候改动某一个别名中的元素都会影响对象本身
                                         # 因此，要实现将数组复制并防止交叉影响，需要深拷贝
        for i in range(nrow):            # 重整化随机数矩阵，使其的值只有1跟-1
            for j in range(ncol):
                if new[i,j] >= 0.5:
                    new[i,j] = 1.0
                else:
                    new[i,j] = -1.0
        return new

    # 这个函数可以对训练集中的输入变量进行归一化处理，将其限定在某一个区间内，如：[Vmin,Vmax]
    # 归一化操作的作用是将我们的训练集映射到一个忆阻器可以处理的区间，后面可以通过输出矩阵的学习将其映射回去
    # training_input为训练集的输入变量，格式为：
    # [[x1(t0), x2(t0), x3(t0),...xq(t0)], ..., [x1(tn), x2(tn), x3(tn),...xq(tn)]]
    # output_range的格式为：(Vmin,Vmax)
    def Normalizing(self,training_input,output_range,shift_vec_manual="False",shift_vec=""):
        Q = len(training_input[0])               # Q为输入变量的维数
        length_training = len(training_input)    # 训练集的长度

        training_input = np.array([np.array(training_input[n]) for n in range(length_training)])  # 将训练集转化为一个二维数组，防止报错
        ceiling = training_input.max()  # 获取训练集中的最大值
        floor = training_input.min()    # 获取训练集中的最小值

        Vmin, Vmax = output_range       # 获取归一化后的范围
        normal_factor = float(Vmax-Vmin)/float(ceiling-floor)

        if shift_vec_manual == "False":                     # 使用默认的平移矩阵
            shift_vec = np.zeros(Q)                    # 生成长度为Q的一维零数组
            for i in range(Q): shift_vec[i] = -floor   # 将数组中的每个值都换成训练集最小值的相反数
        else:
            pass

        training_input = [(training_input[n]+shift_vec)*normal_factor for n in range(length_training)]  # 归一化

        return training_input


class reservoir:
    """ This class of function serve as the reservoir module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = reservoir

    ## 以下是各种不同的激活函数（activation function）

    # Sigmoid function
    def Sigmoid(self,x): return 1.0/(1+np.exp(-x))

    def I_nonlinear(self,x,C0,C1,C2,C3):
        pass

class readout:
    """ This class of function serve as the read-out module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = readout

    def LinearRegression(self):
        return

    def LASSO(self,input,output,alpha=0.025):
        input = np.array([np.array(input[n]) for n in range(len(input))])     # 确保输入是一个二维数组
        output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

        lasso = Lasso(alpha=alpha)  # 输入正则化系数
        lasso.fit = (input,output)

        return lasso.coef_, lasso.intercept_


if __name__ == '__main__':
    mask = masking()

    #a = mask.NodeWeight(10,8)

    test = [[1,4,5,6,7,7,8,-23,51,54,21],
            [43,56,12,3,5,-43,1,3,4,6,1]]

    b = mask.Normalizing(test,(0,10))
    print(b)

    #print(a[0])
    #print(a[1])
