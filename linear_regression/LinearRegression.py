import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    '''
    线性回归模型
    '''
    def __init__(self, data, labels, polynomial_degress=0,sinusoid_degree=0, normalize_data=True):
        '''
        :param data:  数据
        :param labels:  标签
        :param polynomial_degress: 是否要对数据进行一些额外变换
        :param sinusoid_degree:  数据的非线性变换
        :param normalize_data:  是否进行初始化 默认为是
        '''
        ## 数据预处理
        (data_processed, features_mean, features_deviation) = prepare_for_training(data,
                                                                                   polynomial_degress,
                                                                                   sinusoid_degree)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degress = polynomial_degress
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        ## 构造参数
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))  # 列向量
    def train(self, alpha, num_iterations = 500):
        '''
        模型训练——num_iterations次梯度下降
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return:
        '''
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        '''
        梯度下降
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return: 每一步的损失
        '''
        cost_history = []  #记录每一步的损失
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history
    def gradient_step(self, alpha):
        '''
        每一步梯度下降的计算
        :param alpha: 学习率
        :return:
        '''
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        # 注意要把 delta 转置后在与self.data做dot
        new_theta = self.theta - alpha*(1 / num_examples)*(np.dot(delta.T, self.data)).T
        self.theta = new_theta #更新参数
    @staticmethod  # 定义成静态 不会传递self数据
    def hypothesis(data, theta):
        '''
        预测函数
        :param theta: 参数
        :return: 预测值
        '''
        prediction = np.dot(data, theta)
        return prediction
    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(data, self.theta) - labels
        cost = (1/2)*np.dot(delta.T, delta)  # 矩阵的平方
        print(cost.shape)
        return cost[0][0]  # ???
    def get_cost(self, data, labels):
        '''
        获得误差 用于测试阶段
        要先对data进行预处理
        '''
        data_processed= prepare_for_training(data,
                                            self.polynomial_degress,
                                            self.sinusoid_degree)[0] #只需要获得data值 因此加上[0]
        return self.cost_function(data_processed, labels)
    def predict(self, data):
        '''
        预测 用于测试阶段 用训练好的模型得到回归结果
        要先对data进行预处理
        '''
        data_processed = prepare_for_training(data,
                                              self.polynomial_degress,
                                              self.sinusoid_degree)[0]  # 只需要获得data值 因此加上[0]
        return self.hypothesis(data_processed, self.theta)