# THis script is specific for the calculation of carrier effective mass.

import numpy as np
from scipy.optimize import leastsq

class effective_mass:
    def __init__(self):
        self.name = effective_mass
    
    # 能带载流子有效质量计算，详见：N. W. Ashcroft, N. D. Mermin. Solid State Physics, 1976: 213-239.
    # 以及面向维基科研：https://en.wikipedia.org/wiki/Effective_mass_(solid-state_physics)
    # 此函数可以计算在能带中运动的载流子的有效质量
    def CalculateEffectiveMass(self,Kstep,band,num_segment,**kwargs):
        num_point_total = len(band)  # 能带总点数
        num_point_segment = int(len(band)/num_segment)  # 每段能带中包含的点数
        # 每一段能带中，用于计算有效质量的点数，如不设置，则默认每段能带所有点都用于计算有效质量
        num_point_evaluating = kwargs['points_evaluating'] if 'points_evaluating' in kwargs else num_point_segment

        # 应注意，V.A.S.P.中计算能带默认的长度单位是Å，能量单位是eV，为将最后结果以电子静止质量m_{e}表示，我们需将输入数据转换为原子单位制
        # 在原子单位制中，长度单位为Bohr， 1 Bohr = 0.529177210903 Å, 1 Bohr^{-1} = 1.8897261246257702 Å^{-1}
        # 能量单位为Hartree， 1 eV = 0.0367493 Hartree
        Kstep = Kstep/1.8897261246257702  # K点路程中，每个点直接间隔的距离
        band = 0.0367493*np.array(band)

        Kpath_segment = np.array([i*Kstep for i in range(num_point_evaluating)])  # 生成衡量有效质量的能带的K空间路程点
        band_segmented = [band[i:i+num_point_evaluating] for i in range(0,num_point_total,num_point_segment)]  # 能带分段
        band_segmented_shifted = [np.array(band_segmented[i])-band_segmented[i][0] for i in range(num_segment)]  # 平移

        # 接下来我们对运动在能带上的载流子的有效质量进行计算（https://yh-phys.github.io/2019/10/26/vasp-2d-mobility/）
        # 考虑到有效质量实际上就是能带曲率的倒数，我们先利用scipy的最小二乘法模块对能带进行二次项拟合，再对二次项的系数进行计算即可
        # 要利用scipy进行拟合，我们首先要定义两个函数
        # 由于我们把能带段起点平移到了原点开始，所以我们只需要考虑形如y=a*x^2的二次多项式，不需要考虑零次项跟一次项（二次多项式的极值点为原点）
        def polynomial(coefficient,x): return coefficient[0]*x**2
        def error(coefficient,x,y): return polynomial(coefficient,x)-y  # 拟合误差
        # 能带平移会带来新的问题，因为能带从原点开始，开头几个点会比较小，所以拟合时，算法会倾向保证后面点的准确性，到时计算结果总体偏大
        # 保持零次项好像会有一点平衡效果，需要更多研究

        # scipy的最小二乘法拟合模块需要一个初猜值
        initial_guess = kwargs['initial_guess'] if 'initial_guess' in kwargs else np.array([0.5])
        EffectiveMass_list = []
        for i in range(num_segment):
            coef = leastsq(error, initial_guess, args=(Kpath_segment, band_segmented_shifted[i]))
            m_eff = 1.0/(2*coef[0][0])  # 计算有效质量
            EffectiveMass_list.append(m_eff)

        return EffectiveMass_list
