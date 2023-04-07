#coding:utf-8
import numpy as np
import scipy
import scipy.sparse as ss
import random

# dim为矩阵的维数
# density为稀疏矩阵的密度
# value_range为非零元素的取值范围
import scipy.sparse.linalg

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

def ExpandSparseMatrix(Sparse_Matrix):
    print(Sparse_Matrix, "shape is ", Sparse_Matrix.shape)
    Full_Matrix = Sparse_Matrix.todense()  # todense函数可以将稀疏矩阵转为完全阵
    print(Full_Matrix, "#fullM,", "shape is ", Full_Matrix.shape)
    return Full_Matrix

# 这个函数可以对稀疏矩阵进行谱范数归一化，即将输入的稀疏矩阵缩放为一个谱范数（最大奇异值）为1的稀疏矩阵
def NormalizeSpectralNorm(Sparse_Matrix):
    # 获取Sparse_Matrix的最大奇异值，以及对应的左右奇异向量
    svec_right, sval, svec_left = scipy.sparse.linalg.svds(Sparse_Matrix,k=1,which='LM')  # LM-Largest Magnitude
    spectral_norm = sval[0]  # 输入矩阵的谱范数（Spectral norm），也是将使用的归一化因子
    Normalized_Sparse_Matrix = Sparse_Matrix/spectral_norm  # 归一化
    return Normalized_Sparse_Matrix

# 这个函数可以对稀疏矩阵进行谱半径归一化，即将输入的稀疏矩阵缩放为一个谱半径（特征值的模的最大值）为1的稀疏矩阵
def NormalizeSpectralRadius(Sparse_Matrix):
    # 获取Sparse_Matrix的模最大的特征值，以及对应的特征向量
    eval, evec = scipy.sparse.linalg.eigs(Sparse_Matrix,k=1,which='LM')  # LM-Largest Magnitude
    spectral_radius = abs(eval[0])  # 输入矩阵的谱半径（Spectral radius），也是将使用的归一化因子
    Normalized_Sparse_Matrix = Sparse_Matrix/spectral_radius  # 归一化
    return Normalized_Sparse_Matrix

# 这个函数可以计算Li-ESN中水库态权重矩阵的有效谱半径（Li-ESN专用）
# rho-
def CalEffectiveSpectralRadius(Weight_Matrix,rho,a):
    dim = Weight_Matrix.shape[0]  # 获取权重矩阵的维数（权重矩阵应当为方阵）
    I = np.identity(dim)
    W = rho*Weight_Matrix+(1-a)*I
    eval, evec = scipy.sparse.linalg.eigs(W,k=1,which='LM')  # 求W模最大的特征值
    return abs(eval)

M_sparse = GenSparseMatrix((150,150),0.04,(-1,1))
M_full = ExpandSparseMatrix(M_sparse)
#print(M_full.shape[0])

eig_val, eig_vec = scipy.sparse.linalg.eigs(M_full,k=1,which='LM')
right_sin_vec, sin_val, left_sin_vec = scipy.sparse.linalg.svds(M_full,k=1,which='LM')

M_normalized = NormalizeSpectralRadius(M_full)

eig_val_n, eig_vec_n = scipy.sparse.linalg.eigs(M_normalized,k=1,which='LM')
right_sin_vec_n, sin_val_n, left_sin_vec_n = scipy.sparse.linalg.svds(M_normalized,k=1,which='LM')

print(CalEffectiveSpectralRadius(M_full,0.5,0.5))

#print(sin_val)
#print(sin_val_n)
#print(eig_val_n)