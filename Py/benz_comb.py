# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:09:16 2017

@author: byq
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

#import xgboost as xgb
#import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

enc = LabelEncoder()
enc2 = OneHotEncoder()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = train.iloc[:, 1]
'''
X1 = train.iloc[:, 2:10].append(test.iloc[:, 1:9])
X2 = (train.iloc[:, 10:].append(test.iloc[:, 9:])).values
X1_encoded = enc2.fit_transform(X1.apply(enc.fit_transform).values).toarray()
X = np.hstack([X1_encoded, X2])
trainX, testX = X[:4209, :], X[4209:, :] #baseline, 未降维

n_component = 12
dim_reduce = [PCA, FastICA, TruncatedSVD, NMF, GaussianRandomProjection,
              SparseRandomProjection, FeatureAgglomeration]

for reducer in tqdm(dim_reduce):
    ruduce_name = str(reducer).strip("'>").split('.')[-1]
    pca1 = reducer(n_component)
    pca2 = reducer(n_component)

    #
    X1_pca = pca1.fit_transform(X1_encoded[:4209, :])
    X2_pca = pca2.fit_transform(X2[:4209, :])
    trainX_dr = np.hstack([X1_pca, X2_pca])
    #
    X1_pca = pca1.transform(X1_encoded[4209:, :])
    X2_pca = pca2.transform(X2[4209:, :])
    testX_dr = np.hstack([X1_pca, X2_pca])

    #组合
    trainX_dr = np.hstack([trainX, trainX_dr])
    testX_dr = np.hstack([testX, testX_dr])

X1 = train.iloc[:, 2:10].append(test.iloc[:, 1:9])
X1_encoded = X1.apply(enc.fit_transform).values
trainX = np.hstack([X1_encoded[:4209, :], train.iloc[:, 10:]])
testX = np.hstack([X1_encoded[4209:, :], test.iloc[:, 9:]])
'''

#二: 不对X1作one-hot
X1 = train.iloc[:, 2:10].append(test.iloc[:, 1:9])
trainX2 = train.iloc[:, 10:]
testX2 = test.iloc[:, 9:]
X1_encoded = MinMaxScaler().fit_transform(X1.apply(enc.fit_transform).values)
trainX1 = X1_encoded[:4209, :]
testX1 = X1_encoded[4209:, :] #baseline, 未降维
trainX = np.concatenate([trainX1, trainX2], axis=1)
testX = np.concatenate([testX1, testX2], axis=1)


n_component = 16
dim_reduce = [PCA, FastICA, TruncatedSVD, NMF, GaussianRandomProjection,
              SparseRandomProjection, FeatureAgglomeration]

dims = []
for reducer in tqdm(dim_reduce):
    ruduce_name = str(reducer).strip("'>").split('.')[-1]
    pca = reducer(n_component)

    X_pca = pca.fit_transform(trainX)
    trainX_dr = X_pca #np.hstack([trainX1, X2_pca])

    X_pca = pca.transform(testX)
    testX_dr = X_pca #np.hstack([testX1, X2_pca])

    dims.append([ruduce_name, trainX_dr, testX_dr])

train_, test_, id_ = list(), list(), list()
for dim in dims:
    train_.append(dim[1])
    test_.append(dim[2])
    id_.append(dim[0])

train_.append(trainX)
test_.append(testX)
train_ = np.concatenate(train_, axis=1)
test_ = np.concatenate(test_, axis=1)

#组合
#    trainX_dr = np.hstack([trainX, trainX_dr])
#    testX_dr = np.hstack([testX, testX_dr])

X_train, X_test, y_train, y_test = train_test_split(train_, y, test_size=0.2)

#GridSearch:
param = {
    'loss': ['ls', 'huber'],
    'n_estimators': [60, 100, 200],
    'learning_rate': [0.005, 0.05, 0.5],
    'max_depth': [3, 4],
    'min_samples_split': [2],
    'verbose': [1]
    }

model = GradientBoostingRegressor()
GV = GridSearchCV(model, param, verbose=1, n_jobs=1)
print('开始GV......')
GV.fit(X_train, y_train)
print(GV.best_params_)
model = GV.best_estimator_

#评估
score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)
score_all = model.score(train_, y)
overall_cvs = cross_val_score(model, train_, y).mean()
print('train score: {:.4f}'.format(score_train))
print('test score: {:.4f}'.format(score_test))
print('overall score: {:.4f}'.format(score_all))
print('overall cvs: {:.4f}'.format(overall_cvs))


index = test['ID'].values
pred = model.predict(test_).round(4)
pred = pd.DataFrame({'ID': index, 'y': pred})
pred.to_csv(ruduce_name + '.csv', index=False)


with open('res.txt', 'a') as f:
    cont = [ruduce_name]
    cont.append(param.values())
    cont.append(GV.best_params_)
    cont.append([trainX_dr.shape, n_component, score_train, score_test, score_all, overall_cvs])
    f.write(str(cont) + '\n')


