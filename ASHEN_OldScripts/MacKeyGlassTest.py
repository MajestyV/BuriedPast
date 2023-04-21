import numpy as np
import matplotlib.pyplot as plt

DS = DynamicSystems.dynamic_systems()

para = [10,0.2,0.1]

tau = 23   # tau越大，越混沌
step_length = 1   #迭代步长，应满足tao%h=0,否则存在误差

x0 = np.zeros(tau)
for i in range(tau):
    x0[i] = np.cos(i)   #初值是用连续函数定义的

x = DS.Mackey_Glass(x0,para,tau,6000,step_length)

plt.plot(x[1000:5000],x[1000+tau:5000+tau])