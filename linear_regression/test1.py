import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LinearRegression import LinearRegression
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

########## 数据读取与处理 ##########
data = pd.read_csv('./data/world-happiness-report-2017.csv')
## 划分训练测试集
train_data = data.sample(frac=0.8)  # 指定训练集占80％
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 注意
x_train =  train_data[[input_param_name]].values
y_train =  train_data[[output_param_name]].values
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

## 数据展示
plt.scatter(x_train, y_train, label='Train_data')
plt.scatter(x_test, y_test, label='Test_data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()

########## 模型训练 ##########
## 参数初始化
num_iterations = 500
learning_rate = 0.01

linear_model = LinearRegression(x_train, y_train)
[theta, cost_history] = linear_model.train(learning_rate, num_iterations)
print('开始时的损失：', cost_history[0])
print('训练后的损失', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失loss')
plt.show()

########## 模型测试 ##########
y_prediction = linear_model.predict(x_test)
plt.scatter(x_train, y_train, label='Train_data')
plt.scatter(x_test, y_test, label='Test_data')
plt.plot(x_test, y_prediction, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()