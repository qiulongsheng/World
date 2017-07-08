# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 00:59:54 2016

@author: Atlantis
"""


import numpy as np
from time import time, sleep
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from sklearn import metrics
from tqdm import tqdm
import tables
#import h5py
import matplotlib.pyplot as plt


def VGG_16(weight='0915weights.h5'):
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

#sgd = SGD(lr=0.00001, decay=0, momentum=0.9, nesterov=True)#sgd = rmsprop(lr=0.0001)
#sgd = RMSprop(lr=0.000001, rho=0.9, epsilon=1e-8)
sgd = Adam(lr=0.00001)
model = VGG_16()
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
time2 = time()
print('模型编译完毕，\n开始载入数据...')


train = tables.open_file(r'F:\CV大赛\bottleneck_features_train.h5', 'r')
test = tables.open_file(r'F:\CV大赛\bottleneck_features_test.h5', 'r')
train_dataset = train.root.Train_dataset
test_dataset = test.root.Test_dataset


#nb_classes = 12
batch_size = 64
step = batch_size // 32 #4
train_rows = train_dataset.nrows #2900
test_rows = test_dataset.nrows 
nb_epoch = 2 #训练 epoch 数

			
#训练模型
print('模型开始训练...')
time0 = time()
losses = {'loss':[],'acc':[],'time':[]}
nb_batch = train_rows // step #725
for k in range(nb_epoch):	
	for i in tqdm(range(0, train_rows, step)):
		Xs = []
		Ys = []
		for row in train_dataset.iterrows(i, i+step):
			Xs.append(row['value'])
			Ys.append(row['labels'])
		Xs =np.concatenate(Xs)
		Ys =np.concatenate(Ys)	
		loss = model.train_on_batch(Xs, Ys)	
		losses['loss'].append(float(loss[0]))
		losses['acc'].append(float(loss[1]))		
		print('k: %d,' % k, 'loss: %f;' % loss[0], 'accuracy: %f' % loss[1])
#		update_line(fig[0], i//step, float(loss[0]))			
print('模型训练完毕，耗时：%d秒。\n模型开始运行预测程序...' % (time()-time0))
losses['time'].append(time()-time0)

#在训练集上评估模型
Y_train = []
Y_train_pred = []
time0 = time()
for i in tqdm(range(0, train_rows, step)):
	Xs = []
	Ys = []
	for row in train_dataset.iterrows(i, i+step):
		Xs.append(row['value'])
		Ys.append(row['labels'])
	Xs =np.concatenate(Xs)
	Ys =np.concatenate(Ys)	
	Y_train.append(Ys)
	y = model.predict_classes(Xs)	
	Y_train_pred.append(y)	
losses['time'].append(time()-time0)		

#在测试集上评估模型	
time0 = time()
Y_test = []
Y_pred = []
nb_batch = test_rows // step
for i in tqdm(range(0, test_rows, step)):
	Xs = []
	Ys = []
	for row in test_dataset.iterrows(i, i+step):
		Xs.append(row['value'])
		Ys.append(row['labels'])
	Xs =np.concatenate(Xs)
	Ys =np.concatenate(Ys)	
	Y_test.append(Ys)
	y = model.predict_classes(Xs)	
	Y_pred.append(y)	
losses['time'].append(time()-time0)		
		
Y_train = np.concatenate(Y_train)
Y_test = np.concatenate(Y_test)		
Y_train_pred = np.concatenate(Y_train_pred)
Y_pred = np.concatenate(Y_pred)	
#Y_train = train_dataset.col('labels')
#Y_test = test_dataset.col('labels')
print('训练预测准确度: ', metrics.accuracy_score(Y_train_pred, Y_train.argmax(1)))
print('测试预测准确度: ', metrics.accuracy_score(Y_pred, Y_test.argmax(1)))
plt.plot(losses['loss'])
plt.plot(losses['acc'])
print(losses['time'])
model.save_weights('0915weights.h5')
#model.save('0907model.h5')

