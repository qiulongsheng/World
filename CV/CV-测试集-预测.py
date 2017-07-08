# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 22:49:42 2016

@author: Atlantis
"""


#from time import time
import numpy as np
import pandas as pd
import tables
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D


test = tables.open_file('TEST5_features.h5', 'r')
test_table = test.root.TEST5_dataset

#生成模型
def VGG_16(weight='0921weights_Test.h5'): # 0921weights_Test, 0921weights_Error
	model = Sequential()
	model.add(ZeroPadding2D((1,1), input_shape=(512, 14, 14)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(12, activation='softmax'))
	model.load_weights(weight)
	return model	
model = VGG_16()
print('数据模型准备完毕, 模型开始预测...')

#运行预测程序
standard = [6, 7, 4, 3, 8, 0, 10, 9, 2, 1, 11, 5] #将预测标签映射到提交标准，Test集的模型基于旧标准训练的，需要转换;Error集的模型不需要转换
preds = {'name':[], 'top1':[], 'prob1':[], 'top2':[], 'prob2':[]}
for row in tqdm(test_table.iterrows()):	
	name = row['name']
	name = str(name, encoding = 'utf-8') #因为name是bytes格式(b'065ce5b41a05423ebd3765cf13f49f42'),转换为str格式
	preds['name'].append(name) #存入图片名称
	#模型预测
	y_proba = model.predict_proba(row['value'])[0] #因为返回结果是array([[]]) 形状
	order = y_proba.argsort() #Returns the indices that would sort this array.
	preds['top1'].append(standard[order[-1]]) #存入 Top2 预测结果,order[-1]是(概率)最大值的索引，即预测标签
	preds['prob1'].append(y_proba[order[-1]]) #Top1 概率
	preds['top2'].append(standard[order[-2]]) #存入 Top2 预测结果
	preds['prob2'].append(y_proba[order[-2]]) #Top2 概率
preds = pd.DataFrame(preds) #将预测结果dict转换为DataFrame
preds.to_csv('Test5预测结果_Test.csv')


