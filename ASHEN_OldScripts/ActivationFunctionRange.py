import numpy as np
from scipy.misc import derivative  # 使用scipy.misc模块下的derivative方法函数进行求导
from Scripts.ASHEN import ActivationFunction
from GetData import Visualization

AF = ActivationFunction.ActFun()  # 调用ActivationFunction模块
VI = Visualization.plot()         # 调用Visualization模块

x = np.linspace(-4,4,100)
y = x

# 激活函数部分
# 重新定义激活函数，方便后续求导等操作
parameters = [2.036589219,5.65E-03,3.13E+01,1.964978616]
def ActFun(input):
    output = AF.Nonlinear_IV(input,parameters)
    return output

def ActFun_float(input):
    output = AF.Nonlinear_IV_float(input,parameters)
    return output

y_ActFun = ActFun(x)
print(y_ActFun)

dy_ActFun = []
for i in range(100):
    dy_ActFun.append(derivative(ActFun_float,x[i],dx=1e-6)) # 利用derivative函数进行求导
print(dy_ActFun)


# 画图模块
VI.GlobalSetting()  # 引入全局画图参数

VI.Visualize(x,y)
VI.Visualize(x,y_ActFun)

VI.FigureSetting(xlabel='x (a.u.)',ylabel='f(x) (a.u.)')
