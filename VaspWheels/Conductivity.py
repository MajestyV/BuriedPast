import numpy as np

class conductivity:
    """ This class of function is design to estimate the conductivity of the materials based on DFT calculation results. """
    def __init__(self):
        self.name = conductivity

    # 通过这个函数，可以提取所需要的科学常数，所有常数都采用国际单位制
    def Constants(self,name_list):
        constants_dict = {'h': 6.62607015e-34,       # 普朗克常数，单位为J*s
                          'h_bar': 1.05457266e-34,   # 约化普朗克常数，单位为J*s
                          'kB': 1.380649e-23,        # 玻尔兹曼常数，单位为J/K
                          'e': 1.602176634e-19}      # 电子电量，单位为C
        constants_list = [constants_dict[name_list[n]] for n in name_list]
        return constants_list

    # 通过这个函数，我们可以计算本征激发情况下，本征半导体的有效载流子浓度（默认温度T=300K）
    def EffectiveChargeDensity(self,band_gap,m_electron,m_hole,T=300.0):
        Eg, m_e, m_h = [band_gap,m_electron,m_hole]  # 重命名变量，方便后续代码编写
        h_bar, kB = self.Constants(['h_bar', 'kB'])  # 提取约化普朗克常数以及玻尔兹曼常数

        N_c = 2*(m_e*kB*T/(2*np.pi*h_bar**2))**1.5   # 导带的有效状态密度
        N_v = 2*(m_h*kB*T/(2*np.pi*h_bar**2))**1.5   # 价带有效状态密度

        n_i = (N_c*N_v)**0.5*np.exp(-Eg/(2*kB*T))    # 有效载流子浓度

        return n_i

    # 通过这个函数，我们可以计算电导率
    def Conductivity(self,electron_mobility,hole_mobility,band_gap,m_electron,m_hole,T=300.0):
        e = self.Constants(['e'])                                        # 提取电子电量
        n_i = self.EffectiveChargeDensity(band_gap,m_electron,m_hole,T)  # 计算有效载流子浓度
        sigma = n_i*e*(electron_mobility+hole_mobility)                  # 计算电导率
        return sigma

    # 欧姆定律：J = sigma*E
    # 应注意，此函数的变量sigma为conductivity，即电导率，而非电导
    # 如果利用将输入的Efield换成voltage，sigma换成电导G，那么输出的则是电流I，公式变为I = G*V = V/R
    def CurrentDensity(self,Efield,sigma):
        return sigma*Efield

if __name__ == '__main__':
    pass