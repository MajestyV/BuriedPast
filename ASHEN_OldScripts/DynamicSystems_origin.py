import numpy as np
import matplotlib.pyplot as plt

######################################################################
#注意事项:                                                            #
#1. 利用这个函数包模拟动态系统时, 应注意步长要设置合适, 如: 0.0001~0.01之间,   #
#   太小的步长会导致数据量的暴涨, 太大的步长会使系统发散, 无法捕捉真实的动态
#

class dynamic_systems:
    """ This class of functions could produce the trajectories of different dynamic systems. """
    def __init__(self):
        self.name = dynamic_systems

    # origin = [x,y,z] specific the starting point of the system, eg. origin = [0.1,0.1,0.1]
    # parameter = [sigma, gamma, beta] ([Prandtl number, Rayleigh number, some number]) specific the parameters of the system
    # Usually, when gamma < 1, the attractor is the origin; when 1 <= gamma < 13.927, there two stable points
    def LorenzSystem(self,origin,parameter,step,step_length=0.001):
        x, y, z = origin                                       # 解压参数
        sigma, gamma, beta = parameter
        input = [x,y,z,sigma,gamma,beta,step_length]
        x,y,z,sigma,gamma,beta,dt = [float(n) for n in input]  # 将所有输入变量转换成浮点数，以防出错

        Lorenz = np.zeros([step+1,3])  # 创建一个（step+1）行，3列的零矩阵，每一行都是当前step的轨迹坐标，从origin开始
        Lorenz[0] = np.array(origin)   # 因为走step步，所以会有（step+1）个点

        for n in range(step):
            dr_dt = np.array([sigma*(y-x), x*(gamma-z)-y, x*y-beta*z])  # 计算轨迹变化导数：dr/dt = (dx/dt, dy/dt, dz/dt)
            # 更新x,y,z的值，得到下一个点的坐标, r[n+1] = r[n]+(dr/dt)*dt
            x = x + dr_dt[0]*dt
            y = y + dr_dt[1]*dt
            z = z + dr_dt[2]*dt
            Lorenz[n+1] = np.array([x,y,z])  # 将下一个点的坐标更新到数据矩阵中

        return Lorenz

    # 蔡氏电路 （Chua's circuit）
    # origin = [V1,V2,I] specific the starting point of the system
    # In this case, V1, V2 indicate the voltage applied to the capacitor C1 and C2 in the Chua's circuit, and I is the current flow through the inductor
    # parameter = [alpha, beta, c, d], where alpha, beta is decided by the circuit components, alpha = C2/C1, beta = C2*R**2/L
    # c and d is the parameter of the nonlinear resistor, we could assume c = Gb*R, d = Ga*R, where Ga and Gb is the slope of different section of the nonlinear resistor
    def ChuaCircuit(self,origin,parameter,step,step_length=0.001):
        V1, V2, I = origin                # 解压参数
        alpha, beta, c, d = parameter
        input = [V1, V2, I, alpha, beta, c, d, step_length]
        V1, V2, I, alpha, beta, c, d, dt = [float(n) for n in input]  # 将所有输入变量转换成浮点数，以防出错

        Chua = np.zeros([step + 1, 3])  # 创建一个（step+1）行，3列的零矩阵，每一行都是当前step的轨迹坐标，从origin开始
        Chua[0] = np.array(origin)  # 因为走step步，所以会有（step+1）个点

        for n in range(step):
            # 函數f描述了非線性電阻（即蔡氏二极管）的電子響應，並且它的形狀是依賴於它的元件的特定阻態
            f = c*V1+0.5*(d-c)*(np.abs(V1+1)-np.abs(V1-1))

            dr_dt = np.array([alpha*(V2-V1-f), V1-V2+I, -beta*V2])   # 计算轨迹变化导数：dr/dt = (dV1/dt, dV2/dt, dI/dt)
            # 更新V1,V2,I的值，得到下一个点的坐标, r[n+1] = r[n]+(dr/dt)*dt
            V1 = V1 + dr_dt[0] * dt
            V2 = V2 + dr_dt[1] * dt
            I = I + dr_dt[2] * dt

            Chua[n+1] = np.array([V1,V2,I])

        return Chua

    # 这个函数可以重整以上函数的输出，方便数据分析
    def Rearrange(self,data):
        num_time_step = len(data)    # 数据的长度即为时间步的个数
        num_variable = len(data[0])  # 自变量的个数
        data_rearranged = np.zeros([num_variable+1,num_time_step])  # 创建一个行为（自变量数+1），列为时间步数的零矩阵
        data_rearranged[0] = np.linspace(0,num_time_step-1,num_time_step)  # 重整后的数据的第一行即为时间步
        for i in range(num_variable):
            data_rearranged[i+1] = data[:,i]  # 从第二列开始，每一列都是某一个自变量在不同时间步下的值

        return data_rearranged

    # Calculating the spectrum of Lyapunov exponents (李亚普诺夫指数谱)
    def LyapunovExponentSpectrum(self):
        return




if __name__ == '__main__':
    #a = np.zeros([2,6])
    #a[0] = np.array([1,1,1,1,1,1])
    #print(a)

    ds = dynamic_systems()
    #a = ds.LorenzSystem([3.051522,1.582542,15.62388],[10.0,29,2.667],100000,0.0005)
    #a = ds.ChuaCircuit([0.1,0.1,0.1],[10,12.33,-0.544,-1.088],500000,0.001)  # 双漩涡混沌
    a = ds.ChuaCircuit([0.1, 0.1, 0.1], [10, 19.7226, -0.688, -1.376], 10000, 0.01)  # 混沌-周期-混沌-周期演变

    b = ds.Rearrange(a)  # 数据重整化

    t = b[0]
    x = b[1]
    y = b[2]
    z = b[3]

    ax = plt.subplot(projection='3d')
    ax.plot(x,y,z)
    #plt.xlim(-1,1)

    #b = ds.Rearrange(a)

    #plt.plot(b[0],b[1])

    #print(np.linspace(0,10-1,10))
