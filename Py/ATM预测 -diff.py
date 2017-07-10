# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:10:50 2016

@author: Atlantis
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

scaler = MinMaxScaler(feature_range=(-1, 1))


timesteps = 10
VALID = 351
date_end = datetime.strptime('2016-05-29','%Y-%m-%d')
date_start= date_end - timedelta(7*(5+timesteps)-1) #diff后会少一周,所以设为5
sr = pd.read_csv('ATM.csv', encoding='GBK',
                 index_col = ['交易日期'],
                 parse_dates = ['交易日期'])
#趋势图
#sr[sr.id==51300071]['sum'].sort_index().plot()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return pd.DataFrame(dataX), pd.Series(dataY)

def data_collect(data):
    atm = data.groupby('id')
    Xs = pd.DataFrame()
    ys = pd.Series()
    for _, g in atm:
        if g.shape[0] < VALID:
            continue
        g = g['sum'].sort_index()
        g = g[g.index < date_start] #去掉验证集
        g = g.resample(rule='1w').sum()[:-1] #去掉最后一条
        g.dropna(inplace=True) #g.fillna(g.mean(), inplace=True)
        g = g.diff()[1:].values
        g = scaler.fit_transform(g[:, np.newaxis])[:, 0]
        X, y = create_dataset(g, timesteps)
        Xs = Xs.append(X)
        ys = ys.append(y)
    return Xs, ys

Xs_train, ys_train = data_collect(sr)
X_train, X_test, y_train, y_test = train_test_split(
        Xs_train, ys_train, test_size=0.3, random_state=21)

param_grid = [
    {'n_estimators': [80, 100, 120, 140, 160],
    'learning_rate': [1, 0.1, 0.01],
     'max_depth': [2, 3, 4, 5]}
    ]
model = GradientBoostingRegressor(
#                n_estimators=100,
#                random_state=1,
                    loss='ls')
#model = SVR()
#param_grid2 = [{'C':[0.1, 1, 10, 15, 20, 25, 30]}]

GCV = GridSearchCV(model, param_grid)
print('开始GridSearch:')
GCV.fit(X_train, y_train)
print(GCV.best_estimator_, GCV.best_params_, GCV.best_score_)
model = GCV.best_estimator_

#Returns the coefficient of determination R^2 of the prediction.
trainScore = model.score(X_train, y_train)
print('Train Score: ', trainScore)
testScore = model.score(X_test, y_test)
print('Test Score: ', testScore)

joblib.dump(model, 'model.pkl')

#模型验证
atm = sr.groupby('id')
res = pd.DataFrame()

for g, d in atm:
    if d.shape[0] < VALID:
        continue
    d = d['sum'].sort_index()
    d = d[np.logical_and(d.index >= date_start, d.index <= date_end)]
    if d.index[0] != date_start:
        continue
    d = d.resample(rule='1w').sum() #从周一到周日
    base = d[timesteps]
    if d.isnull().any():
        continue
    d = d.diff()[1:].values #第一条是NaN
    d = scaler.fit_transform(d[:, np.newaxis])[:, 0]
    X, y = create_dataset(d, timesteps)
    pred = model.predict(X)
    y = scaler.inverse_transform(y[:, np.newaxis])[:, 0]
    y = base + y.cumsum()
    pred = scaler.inverse_transform(pred[:, np.newaxis])[:, 0] #逆转换
    pred_base = np.hstack((base, y[:len(y)-1]))
    pred = pred_base + pred
    res[g] = np.hstack((y, pred))

res = res.transpose()
for i in range(4):
    res['acu'+str(i)] = (res[i+4] - res[i]) / res[i] #相对偏差
    print(res['acu'+str(i)].abs().mean())

res.to_csv('预测结果.csv')



