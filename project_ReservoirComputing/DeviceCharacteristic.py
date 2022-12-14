import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
# 多项式回归其实对数据进行预处理，给数据添加新的特征，所以调用的库在preprocessing中
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Lasso,LinearRegression

from ReservoirComputing import Visualization

###################################################################################
#
# sklearn程序包具有强大的数据拟合以及分析功能
# 每次多项式回归需要三个重复的步骤：
# （1）确定多项式特征（阶数，degree）；（2）数据的归一化：（3）回归分析
# 通过构建sklearn-Pipeline可以将这三步合在一起避免调用时的重复
#
#

class device:
    """ This function is designed for fitting the device characteristics curve and extracting device parameters. """
    def __init__(self):
        self.name = device

    # 数据提取包
    def GetExperimentData(self,data_file):
        data = pd.read_excel(data_file)   # 利用pandas读取excel中的数据
        title = data.columns              # 利用pandas的columns函数获取数据表头
        I = data[title[0]].values         # 第一列为电流，单位为A，使用.values函数将数据从DataFrame类型转换为数组
        V = data[title[1]].values         # 第二列为电压，单位为V，使用.values函数将数据从DataFrame类型转换为数组
        return np.array([I,V])            # 将电流与电压数据打包成一个二维数组输出

    # Using LASSO regression to fit the I-V characteristics of the device
    # The I-V characteristic of the device is analyzed by Taylor expansion: I = C1*V+C2*V^2+C3*V^3+......
    def DeviceFitting_lasso(self,x_data,y_data,degree=3,alpha=0.025):
        lasso = Pipeline([('poly',PolynomialFeatures(degree=degree)),  # 构造Pipeline时，传入的是一个列表
                          ('std',StandardScaler()),                    # 列表中包含：实现多项式回归的每一个步骤对应的那个类
                          ('LASSO',Lasso(alpha=alpha))])               # 每一个步骤对应的那个类以元组的形式传入
                                                                       # 每一个元组包含：（表示步骤名称的字符串,需要实例化的操作）
        lasso.fit(x_data,y_data)
        return lasso.predict

    # 记得添加备注
    def DeviceFitting_polynomial(self,x_data,y_data,degree=3):
        poly = PolynomialFeatures(degree=degree)  # 加入多项式函数特征
        poly.fit(x_data)
        x_data_poly = poly.transform(x_data)
        parameter = LinearRegression()
        parameter.fit(x_data_poly,y_data)
        return parameter.coef_
        # coef - 第一列常数项，第二列一次项系数，第三列二次项系数


if __name__ == '__main__':
    data_file = 'D:/Projects/PhaseTransistor/Data/I-V sweeping/ITO-MoS2-P(VDF-TrFE)-Gr(.15PVP)/1.14.2022/(4,5).xls'

    dv = device()

    rc = ReservoirComputing.reservoir()

    plot = Visualization.plot()

    data = dv.GetExperimentData(data_file)

    I = data[0][0:2002]*1e6 # 转换成微安
    V = data[1][0:2002]

    I_reshape = I.reshape(-1,1)
    V_reshape = V.reshape(-1,1)
    #V_2D = np.array([V])

    # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

    # 进行I-V特性拟合
    #polynomial = dv.DeviceFitting_polynomial(V_reshape,I_reshape,degree=7)  # sklearn的输入需要是二维数组

    #print(polynomial)

    x = np.linspace(-5,5,1000)
    def y(x,k,b): return k*x+b
    #def y2(x,C1,C2,C3,C4): return C1*x+C2*x**2+C3*x**3+C4*x**4
    y1 = rc.I_Taylor(x,(0.00000000e+00,  5.72491169e-01,  1.95525711e-02,  2.60109040e-02,
                        5.08139165e-03,  2.28172028e-04, -1.77237543e-04, -1.39215122e-05))

    y2 = rc.I_Taylor(x,(0.00000000e+00,-2.84475793e-02,3.91165776e-02,1.21843635e-01,
                        2.09783459e-03,-5.31299752e-03,-7.96619973e-05,9.05348756e-05))

    #print(y2)

    plot.GlobalSetting()

    plot.Visulize(V,I,color=np.array([7,7,7])/255.0)   # 器件I-V特性曲线
    plot.Visulize(x,y1,color=np.array([255,59,59])/255.0)  # 器件性能拟合曲线
    plot.Visulize(x,y2,color=np.array([255,59,59])/255.0)

    plot.FigureSetting(legend='True',labels=['Device I-V characteristic','Fitting'],xlabel='Voltage (V)',
                       ylabel='Current ($\mu$A)')
