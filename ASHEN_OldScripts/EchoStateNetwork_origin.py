# This code is designed to construct an Echo State Network (ESN, as known as Reservoir Computing).

import random
import numpy as np
import scipy
import scipy.sparse as ss
import sklearn.linear_model as sk_fitting
from sklearn.metrics import mean_squared_error, r2_score
from GetData import Visualization

#################################################################
# 模块调用
DS = DynamicSystems.dynamic_systems()  # 调用DynamicSystems模块
VI = Visualization.plot()              # 调用Visualization模块
#################################################################

# 定义要学习的动态系统
origin = [3.051522, 1.582542, 15.62388]  # 起点
parameter = [10.0, 29, 2.667]            # 系统参数
num_step = 5000                         # 步数
step_length = 0.01                      # 步长
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

# 定义用到的函数 ###############################################################################################
# 此函数可用于生成输入权重
# ncol = N (number of virtual nodes, 虚拟节点的个数), nrow = Q (dimension of the input, 输入变量的维数)
def NodeWeight(nrow,ncol,value_range=(-1,1),lock='False',seed=""):
    value_min, value_max = value_range
    if lock == 'True':
        # 利用seed控制每次生成的随机数一样
        random_matrix = np.random.RandomState(seed).uniform(value_min,value_max,(nrow,ncol))
    else:
        random_matrix = np.random.uniform(value_min,value_max,(nrow,ncol))  # 随机生成[-1,1)的浮点数，组成nrow*ncol的矩阵
    return random_matrix

# print(NodeWeight(3,8))

# 此函数可用于生成稀疏矩阵用作reservoir
# dim为矩阵的维数，density为稀疏矩阵的密度，value_range为非零元素的取值范围
def GenSparseMatrix(dim, density, value_range):
    x_dim, y_dim = dim  # 解压维数

    num_element = x_dim * y_dim  # 矩阵的总元素个数
    num_nonzero = round(num_element * density)  # 四舍五入取整

    value_min, value_max = value_range  # 非零元素值的范围

    # 随机产生行、列坐标和值
    x = [random.sample(range(0, x_dim), 1)[0] for i in range(num_nonzero)]  # 从range(0,x_dim)中随机选取num_nonzero个值组成x坐标
    y = [random.sample(range(0, y_dim), 1)[0] for j in range(num_nonzero)]  # 从range(0,y_dim)中随机选取num_nonzero个值组成y坐标
    values = np.random.uniform(value_min, value_max, (num_nonzero))  # 随机生成[-1,1)的浮点数，组成长度为num_nonzero的矩阵

    x_coordinates = np.array(x)  # 将x跟y的值从列表转变成数组
    y_coordinates = np.array(y)

    # coo_matrix函数生成稀疏矩阵
    Sparse_Matrix = ss.coo_matrix((values, (x_coordinates, y_coordinates)),shape=(x_dim, y_dim))

    return Sparse_Matrix

# 这个函数可以对稀疏矩阵进行归一化，即将输入的稀疏矩阵缩放为一个谱范数（即最大的奇异值）为1的稀疏矩阵
def NormalizeSpectralNorm(Sparse_Matrix):
    # 获取Sparse_Matrix的最大奇异值，以及对应的左右奇异向量
    svec_right, sval, svec_left = scipy.sparse.linalg.svds(Sparse_Matrix,k=1,which='LM')  # LM-Largest Magnitude
    spectral_norm = sval[0]  # 输入矩阵的谱范数（Spectral norm），也是将使用的归一化因子
    Normalized_Sparse_Matrix = Sparse_Matrix/spectral_norm  # 归一化
    #print(spectral_norm)
    #svec_right_n, sval_n, svec_left_n = scipy.sparse.linalg.svds(Normalized_Sparse_Matrix, k=1, which='LM')  # LM-Largest Magnitude
    #print(sval_n[0])
    return Normalized_Sparse_Matrix

# 这个函数可以对稀疏矩阵进行谱半径归一化，即将输入的稀疏矩阵缩放为一个谱半径（特征值的模的最大值）为1的稀疏矩阵
def NormalizeSpectralRadius(Sparse_Matrix):
    # 获取Sparse_Matrix的模最大的特征值，以及对应的特征向量
    eval, evec = scipy.sparse.linalg.eigs(Sparse_Matrix,k=1,which='LM')  # LM-Largest Magnitude
    spectral_radius = abs(eval[0])  # 输入矩阵的谱半径（Spectral radius），也是将使用的归一化因子
    Normalized_Sparse_Matrix = Sparse_Matrix/spectral_radius  # 归一化
    return Normalized_Sparse_Matrix

# 激活函数
def tanh(x): return np.tanh(x)

# Ridge regression（岭回归）
def RIDGE(input,output,alpha=1.0):
    input = np.array([np.array(input[n]) for n in range(len(input))])     # 确保输入是一个二维数组
    output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

    #print(input.shape)
    #print(output.shape)

    ridge = sk_fitting.Ridge(alpha=alpha)  # 输入正则化系数
    ridge.fit(input,output)

    output_predict = ridge.predict(input)

    # 结果评估
    print('Mean Squared Error (MSE): %.2f'% mean_squared_error(output, output_predict))
    print('Coefficient of determination (R^2): %.2f'% r2_score(output, output_predict))

    return ridge.coef_, ridge.intercept_

# LASSO（least absolute shrinkage and selection operator）回归
def LASSO(input,output,alpha=0.025):
    input = np.array([np.array(input[n]) for n in range(len(input))])     # 确保输入是一个二维数组
    output = np.array([np.array(output[n]) for n in range(len(output))])  # 确保输出是一个二维数组

    lasso = sk_fitting.Lasso(alpha=alpha)  # 输入正则化系数
    lasso.fit(input,output)

    output_predict = lasso.predict(input)

    # 结果评估
    print('Mean Squared Error (MSE): %.2f' % mean_squared_error(output, output_predict))
    print('Coefficient of determination (R^2): %.2f' % r2_score(output, output_predict))

    return lasso.coef_, lasso.intercept_
###############################################################################################################

# 要定义我们的网络，我们需要定义三个关键变量：输入，水库态（reservoir state），输出
N = 150  # N是水库矩库的边长，同时也就是水库态向量的长度
D_u, D_x, D_y = [len(u_train[0]),N,len(y_train[0])]  # 输入，水库态以及输出向量的维度
#u   # 输入向量
#x   # 水库态向量
#y   # 输出向量

W_in = NodeWeight(D_u,D_x,(-0.3,0.3))
W_raw = GenSparseMatrix((D_x,D_x),0.04,(-1,1)).todense()
W = NormalizeSpectralRadius(W_raw)
# print(W.shape)

a = 0.5  # a=1时，网络为最基础的Echo State Network (ESN)
gamma = 0.5

x0 = np.zeros((1,D_x),dtype=float)
x_train = []
v_train = []
for i in range(num_train):
    u = np.array([u_train[i]])  # 转换成二维数组才能进行矩阵运算
    #print(u.shape)
    #x = u@W_in
    x = (1.0-a)*x0+tanh(u@W_in+gamma*(x0@W))
    x_train.append(x.getA()[0])  # 将矩阵转换为数组后取第一个元素（即第一行）以降维
    v_train.append(np.hstack((x.getA()[0],u_train[i])))  # 将水库态与输入串接，很关键！！！
    x0 = x

print(x_train)

# coef,intercept = RIDGE(x_train,y_train_x,alpha=1.0)
#coef,intercept = LASSO(x_train,y_train,alpha=0.1)
coef,intercept = RIDGE(v_train,y_train,alpha=1.0)

y_head = np.dot(v_train,np.transpose(coef))+intercept

# 分析数据模块
# print(coef,intercept)
# VI.Analyzing(y_train,y_head)

# 预测模块
u_i = np.array([u_predict[0]])
x_i = np.array([x_train[len(x_train)-1]])  # 水库态的初始值是最后输出的水库态
# v_network = []
y_network = []
for i in range(num_predict):
    x_p = (1.0-a)*x_i+tanh(u_i@W_in+gamma*(x_i@W))
    v_p = np.hstack((x_p,u_i))
    y_p = np.dot(v_p, np.transpose(coef))+intercept

    u_i = y_p
    x_i = x_p

    y_network.append(y_p.getA()[0])

VI.Analyzing(y_predict,y_network)
#VI.VisualizeDynamicSystem(y_network)