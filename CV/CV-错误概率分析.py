# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 00:58:07 2016

@author: Atlantis
"""

'''
把预测错误图片复制出来到error文件夹
'''

#from time import time
import os
from shutil import copyfile, copy2
import numpy as np
import pandas as pd
import tables
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

k = 4
test = tables.open_file('TEST%d_features.h5' % k, 'r')
test_table = test.root.TEST4_dataset 

#生成模型
def VGG_16(weight='0917weights_Test.h5'):
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
standard = [6, 7, 4, 3, 8, 0, 10, 9, 2, 1, 11, 5] #将预测标签映射到提交标准
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
#preds.to_csv('Test4预测结果.csv')

answer = pd.read_csv('F:\CV大赛\测试集\Answer\BOT_Image_Testset %d.txt' % k, header='infer', delimiter='\t')
#Testset1的 header 设为'None', 2 ,3, 4...设为 'infer'
top1_acc =[]
scores = []
error1 = {} #top1错误, top2正确
error2 = {} #top1,2错误
hidden = {} #隐藏任务	
for item in answer.iterrows(): #item是长度为2的tuple
	item = item[1] # Series，item的值，[0]是索引,形式:065ce5b41a05423ebd3765cf13f49f42 6 NaN 0
	y_true = item[1]
	pred = preds[preds.icol(0)==item[0]] # preds中对应item的行，1行5列的DataFrame,注意：顺序已经乱了
	top1 = pred['top1'].values[0]  #pred['top1']为Series,其Values为array([9], dtype=int64)形式
	top2 = pred['top2'].values[0]
	top1_score = (top1==y_true) #
	top1_acc.append(top1_score)
	top2_score = (1 * (top1==y_true) + 0.4 * (top2==y_true)) * 2**item[3] #如果item[3]=1,隐藏任务
	top2_score = round(top2_score, 1)
	scores.append(top2_score)
	
	#把分类错误的图片名称提取出来	
	animal = ['天竺鼠','松鼠','梅花鹿','狐狸','狗','狼','猫','花栗鼠','长颈鹿','驯鹿','鬣狗','黄鼠狼']	
	if top2_score == 0.4: #top1错误， top2正确
		error1[item[0]] = []
		error1[item[0]].append(pred['prob1'].values[0])
		error1[item[0]].append(pred['prob2'].values[0])
	elif top2_score == 0.0: #	top1、2都错误	
		error2[item[0]] = []
		error2[item[0]].append(pred['prob1'].values[0])
		error2[item[0]].append(pred['prob2'].values[0])


top1_accuracy = np.mean(top1_acc)
score = np.mean(scores)* 100
print('top1_acc:', top1_accuracy)
print('Score:', score)

e=pd.DataFrame(error2)
e=e.transpose()


