import numpy as np
import matplotlib.pyplot as plt

class ferroelectric:
    """这个函数类别可以模拟铁电滞回线以及相关的特性"""
    def __init__(self):
        self.name = ferroelectric

    # 计算铁电的偶极子滞回线
    # Ec-矫顽场（coercive field），Ps-饱和极化强度（saturated polarization），Pr-剩余极化强度（remnant polarization）
    def P_dipole(self,E,Ec,Ps,Pr):
        # The constant delta is defined such that P_rising(0) = -Pr
        delta = Ec*(np.log((1+Pr/Ps)/(1-Pr/Ps)))**-1  # numpy中，np.log(x)即是以e为底数的对数ln(x)

        P_rising = Ps*np.tanh((E-Ec)/(2*delta))
        P_falling = -Ps*np.tanh((-E-Ec)/(2*delta))

        return P_rising,P_falling

if __name__ == '__main__':
    ferro = ferroelectric()
    E = np.linspace(-300,300,1000)
    P1,P2 = ferro.P_dipole(E,50,20,10)
    plt.plot(E,P1)
    plt.plot(E,P2)