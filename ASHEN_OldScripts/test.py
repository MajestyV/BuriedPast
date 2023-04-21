import numpy as np
from Scripts.ASHEN import Visualization

RC_masking = ReservoirComputing.masking()
RC_reservoir = ReservoirComputing.reservoir()
RC_readout = ReservoirComputing.readout()

DS = DynamicSystems.dynamic_systems()

ev = Evaluation.evaluate()

plot = Visualization.plot()

#data = np.array([[ -2.95507616,  10.94533252],
                 #[ -0.44226119,   2.96705822],
                 #[ -2.13294087,   6.57336839],
                 #[  1.84990823,   5.44244467],
                 #[  0.35139795,   2.83533936],
                 #[ -1.77443098,   5.6800407 ],
                 #[ -1.8657203 ,   6.34470814],
                 #[  1.61526823,   4.77833358],
                 #[ -2.38043687,   8.51887713],
                 #[ -1.40513866,   4.18262786]])

#a = np.zeros([2,6])
    #a[0] = np.array([1,1,1,1,1,1])
    #print(a)

#a = ds.LorenzSystem([3.051522,1.582542,15.62388],[10.0,29,2.667],100000,0.0005)
#a = ds.ChuaCircuit([0.1,0.1,0.1],[10,12.33,-0.544,-1.088],500000,0.001)  # 双漩涡混沌
#a = ds.ChuaCircuit([0.1, 0.1, 0.1], [10, 19.7226, -0.688, -1.376], 10000, 0.01)  # 混沌-周期-混沌-周期演变

N = 6
Q = 3

ground_truth_raw = DS.ChuaCircuit([0.1, 0.1, 0.1], [10, 19.7226, -0.688, -1.376], 20000, 0.01)  # 混沌-周期-混沌-周期演变
ground_truth = DS.Rearrange(ground_truth_raw)  # 数据重整化

# 前3000点可能包含初始点的信息，会是我们的拟合偏移，因此我们从3000点之后开始取值
training_input = ground_truth_raw[1000:6000]
training_output = ground_truth_raw[1001:6001]

# 预测
predicting_input = ground_truth_raw[6001:15000]
predicting_output = ground_truth_raw[6002:15001]

training_input_new = RC_masking.Normalizing(training_input,(0,5),shift_vec_manual="True",shift_vec=np.array([3.0,3.0,3.0]))

predicting_input_new = RC_masking.Normalizing(predicting_input,(0,5),shift_vec_manual="True",shift_vec=np.array([3.0,3.0,3.0]))

M = RC_masking.NodeWeight(Q,N)

M_fixed_regulated = np.array([[-1.,  1., -1., -1.],
                              [-1., -1.,  1.,  1.],
                              [-1.,  1.,  1.,  1.]])

M_fixed = np.array([[0.31065887, 0.1368448,  0.97611208, 0.08314513, 0.00206192, 0.90721682],
                    [0.46223615, 0.71457649, 0.64289362, 0.63055467, 0.11239314, 0.74436349],
                    [0.30533918, 0.20979075, 0.23399308, 0.58935991, 0.21974417, 0.20127576]])

#M_fixed_1 =

print(M)
J = np.dot(training_input_new,M_fixed)

J_predict = np.dot(predicting_input_new,M_fixed)

# 激活函数（即器件的I-V特性）
# X = RC_reservoir.Sigmoid(J)
# X = RC_reservoir.I_nonlinear(J,0,1,1,1)
#X = RC_reservoir.I_nonlinear(J,0,6.0474993,0.4920937,0.2309307)
#X = RC_reservoir.I_nonlinear(J,0,0.33409553,0.04845448,0.03421923)
#X = RC_reservoir.I_Taylor(J,(0.00000000e+00,-2.84475793e-02,3.91165776e-02,1.21843635e-01,2.09783459e-03,
                             #-5.31299752e-03,-7.96619973e-05,9.05348756e-05))
X = RC_reservoir.I_Taylor(J,(0.00000000e+00,-2.84475793e-01,3.91165776e-01,1.21843635e-00,  # 按照比例慢慢调大参数可以有更好的拟合效果
                             2.09783459e-02,-5.31299752e-02,-7.96619973e-4,9.05348756e-4))

# X = X/1000.0

X_predict = RC_reservoir.I_nonlinear(J_predict,0,1,1,1)

#print(X)

coef, intercept = RC_readout.LASSO(X,training_output)

testing_output = np.dot(X,np.transpose(coef))+intercept

predicted_output = np.dot(X_predict,np.transpose(coef))+intercept
#print(testing_output)
#print(training_output)

testing_output_rearrange = DS.Rearrange(testing_output)  # training阶段，模型生成的数据
t_testing = testing_output_rearrange[0]+1001
x_testing = testing_output_rearrange[1]
y_testing = testing_output_rearrange[2]
z_testing = testing_output_rearrange[3]

training_output_rearrange = DS.Rearrange(training_output)  # training的ground truth
t_traning = training_output_rearrange[0]+1001
x_traning = training_output_rearrange[1]
y_traning = training_output_rearrange[2]
z_traning = training_output_rearrange[3]

predicted_output_rearrange = DS.Rearrange(predicted_output)  # 模型的预测
t_predicted = predicted_output_rearrange[0]+6002
x_predicted = predicted_output_rearrange[1]
y_predicted = predicted_output_rearrange[2]
z_predicted = predicted_output_rearrange[3]

predicting_output_rearrange = DS.Rearrange(predicting_output)  # 标准数据-ground truth
t_predicting = predicting_output_rearrange[0]+6002
x_predicting = predicting_output_rearrange[1]
y_predicting = predicting_output_rearrange[2]
z_predicting = predicting_output_rearrange[3]

#plt.plot(x_testing,z_testing)
#plt.plot(x_traning,z_traning)

#print(len(predicted_output))
#print(len(predicting_output))

nrmse_training = ev.NRMSE(training_output,testing_output)
print('nrmse_training')
print(nrmse_training)

nrmse_predicting = ev.NRMSE(predicting_output,predicted_output)
print('nrmse_predicting')
print(nrmse_predicting)

#plt.plot(x_predicted,z_predicted)
#plt.plot(x_predicting,z_predicting)

plot.GlobalSetting()  # 载入全局绘图参数

# 画X-Z相图-训练
#plot.Visulize(x_traning,z_traning,color=np.array([7,7,7])/255.0)
#plot.Visulize(x_testing,z_testing,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Training'],xlabel='X',ylabel='Z')

# 画X训练图
#plot.Visulize(t_traning,x_traning,color=np.array([7,7,7])/255.0)
#plot.Visulize(t_testing,x_testing,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Training'],xlabel='time step',ylabel='X',
                   #xlim=(min(t_traning),max(t_traning)),ylim=(-4,6))

# 画X-Z相图-预测
#plot.Visulize(x_predicting,z_predicting,color=np.array([7,7,7])/255.0)
#plot.Visulize(x_predicted,z_predicted,color=np.array([255,59,59])/255.0)

#plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Prediction'],xlabel='X',ylabel='Z')

# 画X预测图
plot.Visulize(t_predicting,x_predicting,color=np.array([7,7,7])/255.0)
plot.Visulize(t_predicted,x_predicted,color=np.array([255,59,59])/255.0)

plot.FigureSetting(legend='True',labels=['Ground Truth', 'Simulation Prediction'],xlabel='time step',ylabel='X',
                   xlim=(min(t_predicting),max(t_predicting)),ylim=(-5,10))