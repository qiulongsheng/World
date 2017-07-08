# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:21:24 2016

@author: Atlantis
"""
import os
import numpy as np
import h5py
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import tables
from PIL import Image
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical


#创建模型
print('模型开始编译...')
time1 = time()
sgd = Adam(lr=0.00001)#sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)#sgd = rmsprop(lr=0.0001)
weights = r'C:\Users\Atlantis\.keras\models\inception_v3_weights_th_dim_ordering_th_kernels.h5'
model = InceptionV3(weights=None, nb_classes=12)
f = h5py.File(weights)
layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
for k in range(len(layer_names)-1):
	layer_name = layer_names[k]
	g = f[layer_name]
	weight_values = [g[weight_name.decode('utf8')] for weight_name in g.attrs['weight_names']]
	model.layers[k].set_weights(weight_values)
f.close()
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
time2 = time()
print('模型编译完毕，耗时：%d秒。\n模型开始训练...' % (time2-time1))

 ##input_shape: (None, 3, 299, 299)


nb_classes = 12

print('读取并准备图片数据...')
time0 = time()
pics = []
labels = []
for dir_path, dir_name, files in os.walk('F:\CV大赛\计算机视觉大赛'):
#ubuntu: '/media/atlant/D238FDCE38FDB199/MyPy/数据/转换图片'
	#类别名称
	if dir_name:
		class_names = dir_name
	#类别目录下的图片
	if not dir_name:
		for file in files:
			pic_name = os.path.join(dir_path, file)
			pics.append(pic_name)
			labels.append(class_names.index(os.path.split(dir_path)[1]))
pics = np.array(pics)
labels = to_categorical(labels, nb_classes)
X_train, X_test, Y_train, Y_test = train_test_split(pics, labels, test_size=0.2, random_state=5)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

nb_sample = len(X_train)
batch_size = 32
nb_batch = batch_size * 1 #3200
nb_epoch = nb_sample // nb_batch
m = 0
rgb_mean = np.array([103.939, 116.779, 123.68])
rgb_mean = rgb_mean.reshape((1,3,1,1))
#将单张图片转换为标准input格式：（3，width，height），one-hot
def pic_trans(X):
	Xs = []
	for p in X:
		img = Image.open(p).convert('RGB')
		img = img.resize((299,299))
		value = img_to_array(img)
		Xs.append(value)
	Xs -= rgb_mean
	Xs = np.array(Xs)
	return Xs

#生成nb_batch个样本图片数据
#在程序中每次调用该函数之前，必须把 m 重置为0
def BatchGenerator(X, Y):
	global m
	X_batch = X[m*nb_batch:(m+1)*nb_batch]
	Y_batch = Y[m*nb_batch:(m+1)*nb_batch]
	X_batch = pic_trans(X_batch)
	m += 1
	if m >= nb_epoch:
		m = 0
	return X_batch, Y_batch

#训练模型
#可以把 nb_epoch 扩大为 k*nb_epoch，则模型循环训练 k 次，程序不需要修改
m = 0
for i in tqdm(range(nb_epoch)):
	Xs, Ys = BatchGenerator(X_train, Y_train)
	for j in range(len(Xs)//batch_size):
		X, Y = Xs[j*batch_size:(j+1)*batch_size], Ys[j*batch_size:(j+1)*batch_size]
		loss = model.train_on_batch(X, Y)
		print('loss: ',loss)
print('模型训练完毕，耗时：%d秒。\n模型开始运行预测程序...' % (time()-time2))

#在训练集上评估模型
m = 0
Y_train_pred = []
for i in tqdm(range(1)): #range(nb_epoch)，只评估一个 nb_epoch
	Xs, Ys = BatchGenerator(X_train, Y_train)
	for j in range(len(Xs)//batch_size):
		X = Xs[j*batch_size:(j+1)*batch_size]
		y = model.predict_classes(X)
		Y_train_pred.append(y)

#在测试集上评估模型
m = 0
Y_pred = []
nb_epoch = len(X_test) // nb_batch
for i in tqdm(range(nb_epoch)):
	Xs, Ys = BatchGenerator(X_test, Y_test)
	for j in range(len(Xs)//batch_size):
		X = Xs[j*batch_size:(j+1)*batch_size]
		y = model.predict_classes(X)
		Y_pred.append(y)

Y_train_pred = np.concatenate(Y_train_pred)
Y_pred = np.concatenate(Y_pred)
print('训练预测准确度: ', metrics.accuracy_score(Y_train_pred, Y_train[:len(Y_train_pred)].argmax(1)))
print('测试预测准确度: ', metrics.accuracy_score(Y_pred, Y_test[:len(Y_pred)].argmax(1)))

model.save_weights('0924weights.h5')
#model.save('0907model.h5')
