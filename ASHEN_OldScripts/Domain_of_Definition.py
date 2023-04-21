import pandas as pd
import matplotlib.pyplot as plt
from Scripts.ASHEN import VisualizationSCI

VI = VisualizationSCI.plot()  # 调用VisualizationSCI模块

data_file = 'D:/OneDrive/OneDrive - The Chinese University of Hong Kong/Desktop/Temporary_data/Analog-ESN/Demo/Unactivated_States_Range.csv'

# 利用pandas提取数据，得到的结果为DataFrame格式
data_DataFrame = pd.read_csv(data_file,header=1)  # 若header=None的话，则设置为没有列名
data_array = data_DataFrame.values  # 将DataFrame格式的数据转换为数组
print(data_array)
TimeStep = data_array[:,0]  # 时间步
U_min = data_array[:,1]  # 最小值
U_max = data_array[:,2]  # 最大值

# 画图模块
# 画图参数
markersize = 1.5
markeredgewidth = 0.25

# 设置刻度线方向
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内

plt.figure(figsize=(6.2992,1.5))

plt.plot(TimeStep,U_min,'o',markersize=markersize,markeredgewidth=markeredgewidth,markerfacecolor='none',color=VI.MorandiColor('Redred'),label='1')
plt.plot(TimeStep,U_max,'o',markersize=markersize,markeredgewidth=markeredgewidth,color=VI.MorandiColor('Paris'),label='2')

plt.hlines(-5,0,max(TimeStep)+1,colors=VI.MorandiColor('Black'),linewidth=0.75)
plt.hlines(5,0,max(TimeStep)+1,colors=VI.MorandiColor('Black'),linewidth=0.75)
plt.vlines(1000,-6,6,colors=VI.MorandiColor('Black'),linestyle='dashed',linewidth=0.75)
plt.xlim(0,max(TimeStep)+1)
plt.ylim(-6,6)

plt.legend(loc=(0.1,0.88),fontsize=6,frameon=False,ncol=2)

# 保存生成的图片结果
saving_directory = 'D:/OneDrive/OneDrive - The Chinese University of Hong Kong/Desktop/Temporary_data/'
VI.SavingFigure(saving_directory,filename='Domain_of_Definition',format='pdf')
VI.SavingFigure(saving_directory,filename='Domian_of_Definition',format='eps')

plt.show()