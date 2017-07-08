# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 10:48:02 2016

@author: Atlantis
"""



import os
import numpy as np
from PIL import Image
from time import time
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD #, Adam, Adadelta,rmsprop
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
import h5py, cv2
from sklearn import metrics
from tqdm import tqdm

#将单张图片转换为标准input格式：（3，width，height），one-hot
def pic_trans(X):	
	Xs = []
	rgb_mean = np.array([103.939, 116.779, 123.68])
	rgb_mean = rgb_mean.reshape((1,3,1,1))
	if isinstance(X, str):
		X = [X]
	for pic_name in X:
		pic = Image.open(pic_name)			
		value = img_to_array(pic)
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
	
#创建模型
def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    return model	
				
				
nb_classes = 12

print('读取并准备图片数据...')
time0 = time()
pics = []
labels = []
for dir_path, dir_name, files in os.walk('数据\转换图片'):
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

#设置参数
nb_sample = len(X_train)
batch_size = 32
nb_batch = batch_size * 100 #3200
nb_epoch = nb_sample // nb_batch
m = 0

print('模型开始编译...')
time1 = time()
model = VGG_16()
f = h5py.File('vgg16_weights.h5')
for k in range(f.attrs['nb_layers']-13):
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()


import tables 
class Weights(tables.IsDescription):
	value  = tables.Float32Col(shape=(batch_size, 512, 14, 14))   # float  (single-precision)
	labels = tables.Int8Col(shape=(batch_size, nb_classes))
	

train = tables.open_file('bottleneck_features_train.h5', 'w')
test = tables.open_file('bottleneck_features_test.h5', 'w')
train_table = train.create_table('/', 'Train_dataset', Weights)
test_table = test.create_table('/', 'Test_dataset', Weights)
train_row = train_table.row
test_row = test_table.row

#训练数据
m = 0
for i in tqdm(range(nb_epoch)):
	Xs, Ys = BatchGenerator(X_train, Y_train)
	for j in range(len(Xs)//batch_size):
		X, Y = Xs[j*batch_size:(j+1)*batch_size], Ys[j*batch_size:(j+1)*batch_size]
		bottleneck_features_train = model.predict(X)
		train_row['value'] = bottleneck_features_train
		train_row['labels'] = Y
		train_row.append()
	train_table.flush()
train.close()
#测试数据	
m = 0
nb_epoch = len(X_test) // nb_batch
for i in tqdm(range(nb_epoch)):
	Xs, Ys = BatchGenerator(X_test, Y_test)
	for j in range(len(Xs)//batch_size):
		X, Y = Xs[j*batch_size:(j+1)*batch_size], Ys[j*batch_size:(j+1)*batch_size]
		bottleneck_features_test = model.predict(X)
		test_row['value'] = bottleneck_features_test
		test_row['labels'] = Y
		test_row.append()
	test_table.flush()
test.close()	



