import numpy as np
import matplotlib.pyplot as plt

class resistor_capacitor:
    """ This class of function is design to simulate the I-V characteristic of RC parallel circuit."""
    def __init__(self):
        self.name = resistor_capacitor

    def Input(self,x):
        return np.sin(x)

    #def ClampedCubicSpline(self,x,y):
        #for i in range(len(x)):

    # 导数计算模块
    def Diff(self,x,y):
        x,y = [np.array(x),np.array(y)]  # 将输入转换为数组，防止出错
        diff = []
        for i in range(len(x)):  # x and y should have same dimension
            if i == 0:
                dy_dx = (y[1]-y[0])/(x[1]-x[0])
            elif i == len(x)-1:
                dy_dx = (y[i]-y[i-1])/(x[i]-x[i-1])
            else:
                dy_dx = ((x[i]-x[i-1])*(y[i+1]-y[i])/(x[i+1]-x[i])+(x[i+1]-x[i])*(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])
            diff.append(dy_dx)
        return np.array(diff)

    # Resistor-capacitor 并联电流仿真函数
    def Current(self,voltage,time,R,C):
        V,t = [np.array(voltage),np.array(time)]  # 将输入的电压和时间序列转换为数组，以防出错
        R,C = [float(R),float(C)]                 # 将输入的电阻和电压参数转换为浮点数，防止出现python除法中整数规则引发的错误

        I_R = V/R
        I_C = C*self.Diff(t,V)

        return I_R+I_C

    # 可变电阻测试
    def Current_test(self, voltage, time, R, C):
        V, t, R = [np.array(voltage), np.array(time), np.array(R)]  # 将输入的电压和时间序列转换为数组，以防出错
        C = float(C)  # 将输入的电阻和电压参数转换为浮点数，防止出现python除法中整数规则引发的错误

        I_R = V/R
        I_C = C * self.Diff(t, V)

        return I_R + I_C

if __name__ == '__main__':
    rc = resistor_capacitor()
    t = np.linspace(0,2*np.pi,100)
    V = rc.Input(t)
    R = np.append(np.linspace(100,10,25),np.linspace(10,100,75))
    I = rc.Current_test(V,t,R,C=0.01)
    plt.plot(V,I)