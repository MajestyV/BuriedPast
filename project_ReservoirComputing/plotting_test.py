import matplotlib.pyplot as plt

# 画图模块
fig = plt.figure(figsize=(15, 10))  # 控制图像大小
grid = plt.GridSpec(7, 8, wspace=0.6, hspace=0.4)  # 创建柔性网格用于空间分配，输入为(行数, 列数)
# wspace和hspace可以调整子图间距

# 分配子图位置
phase_training = fig.add_subplot(grid[:4, :4])  # X-Z相位图
phase_predicting = fig.add_subplot(grid[:4, -4:])
#t_z = fig.add_subplot(grid[6, :])  # 时序图

#t_x = fig.add_subplot(grid[4, :],sharex=t_z)  # 时序图
#t_y = fig.add_subplot(grid[5, :],sharex=t_z)

t_x = fig.add_subplot(grid[4, :])  # 时序图
t_y = fig.add_subplot(grid[5, :],sharex=t_x)
t_z = fig.add_subplot(grid[6, :],sharex=t_x)
# sharex的设置会使t_x，t_y和t_z拥有一模一样的横坐标轴

t_x.set_xlim([-100,999])
#t_x.set_xticklabels([])
#t_y.set_xticklabels([])
t_x.tick_params('x',labelbottom=False)  # 对于subplot，要调整坐标轴刻度样式的话，需要采用tick_params函数
t_y.tick_params('x',labelbottom=False)  # 如果用别的函数如set_xticklabels()，sharex的设置会把这个函数拷贝到所有share的轴上

plt.show()

# 通过循环批量调节子图参数
sub_fig_list = [sub_fig_1, sub_fig_2, sub_fig_3]
for n in sub_fig_list:
    n.set_yticklabels([])
    # n.set_yticks([])
    n.set_xlim(0, 100)
    n.set_xticks([0, 100])
    n.set_xticklabels(['K', '$\Gamma$'])

sub_fig_4.set_yticklabels([])
sub_fig_4.set_xlim(0, 100)
sub_fig_4.set_xticks([50])
# Text properties for the labels. These take effect only if you pass labels. In other cases, please use tick_params.
sub_fig_4.tick_params(color='w')  # 对于subplot，要调整刻度样式的话，需要采用tick_params函数
sub_fig_4.set_xticklabels(['DOS (a.u.)'])

#plt.show()