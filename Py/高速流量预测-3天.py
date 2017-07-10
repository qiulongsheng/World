# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:35:58 2016

@author: byq
"""

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


data_file = r'D:\项目库\交通数据\inout3m.csv'
data = pd.read_csv(data_file, delimiter=',', engine='python')

#排除春节假期
def drop(d):
    date = d.index
    t1 = pd.to_datetime('2016-02-07')
    t2 = pd.to_datetime('2016-02-13')
    return d[[i or j for i,j in zip(date<t1, date>t2)]].values.astype('float32')

#把单列数据转换为数据集矩阵
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

data.dropna(subset=['Entry_time', 'Total_weigh'], inplace=True) #排除缺失值
data['Entry_time'] = pd.to_datetime(data['Entry_time']) #转换为规范时间格式
data = data[data['Entry_time'] >= pd.to_datetime('2016-01-01 00:00:00')] #指定时间范围
data.sort_values(by='Entry_time', inplace=True) #根据时间排序

#选取指定数据
data = pd.Series(index=data['Entry_time'].values, data=data['Total_weigh'].values)
traffic = data.resample('1d').count() #流量数据
traffic = drop(traffic)
freight = data.resample('1d').sum() #货运量数据
freight = drop(freight)

#选取目标变量，数据预处理
target = traffic  #traffic流量可以替换成freight货运量
data = target.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# split into train and test sets
timesteps = 7 #根据前面timesteps天的数据预测下一天
train_size = 60
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]

X_train, y_train = create_dataset(train, timesteps)
X_test, y_test = create_dataset(test, timesteps)

#创建GBRT模型
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                max_depth=1, random_state=0, loss='ls')
model.fit(X_train, y_train)
print('mse:', mean_squared_error(y_test, model.predict(X_test)))

trainPredict = model.predict(X_train).reshape(-1, 1)
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = model.predict(X_test).reshape(-1, 1)
testPredict = scaler.inverse_transform(testPredict)

print('训练平均绝对误差:', mean_absolute_error(scaler.inverse_transform(train[timesteps:]), trainPredict))
print('测试平均绝对误差:', mean_absolute_error(scaler.inverse_transform(test[timesteps:]), testPredict))

trainPredictPlot = np.empty_like(target)
trainPredictPlot[:] = np.nan
trainPredictPlot[timesteps:len(X_train)+timesteps] = trainPredict[:, 0]

testPredictPlot = np.empty_like(target)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2):len(target)] = testPredict[:, 0]

#输出预测结果到csv文件
outcome = {}
outcome['原始数据'] = target
outcome['训练预测结果'] = trainPredictPlot
outcome['测试预测结果'] = testPredictPlot
outcome = pd.DataFrame(outcome)
outcome.to_csv(r'D:\项目库\交通数据\通行量预测结果.csv')

#预测接下来三天
predict = []
def pred():
    global data
    x = data[-timesteps:].reshape(1,-1)
    y = model.predict(x)
    predict.append(y)
    data = np.concatenate((data, [y]))

k = 5 #参数k指定预测天数
for i in range(k):
    pred()
predict = scaler.inverse_transform(np.reshape(predict, (-1, 1)))
print('%d天的预测结果' % k, predict)

#输出预测结果的图表形式
plt.plot(target)
plt.plot(trainPredictPlot)
plt.plot(np.concatenate((testPredictPlot, predict[:,0])))
plt.title('货运量预测结果')
plt.show()