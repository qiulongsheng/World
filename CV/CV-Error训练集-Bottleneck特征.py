# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:59:04 2016

@author: Atlantis
"""

'''
Test_set1~4预测错误的图片Augment后，用作训练集
'''


import os
import numpy as np
from PIL import Image
from time import time
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
import h5py
from tqdm import tqdm
import tables 

	
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
				
print('模型开始编译...')
time1 = time()
model = VGG_16()
f = h5py.File('vgg16_weights.h5')
for k in range(f.attrs['nb_layers']-13):
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()


#把所有Testset的图片路径和标签存入列表
print('读取并准备图片数据...')
time0 = time()
#读取图片分类标签; Testset1的 header 设为'None', 2 ,3, 4...设为 'infer'
path = 'F:\CV大赛\测试集\error\generate'

animal = {'天':0,'松':1,'梅':2,'狐':3,'狗':4,'狼':5,'猫':6,'花':7,'长':8,'驯':9,'鬣':10,'黄':11}	
nb_classes = 12
pics = [] #保存图片的完整文件路径
labels = []
for pic in tqdm(os.listdir(path)):
	pics.append(os.path.join(path, pic))
	label = animal[pic[0]]
	labels.append(label)
pics = np.array(pics)		
labels = to_categorical(labels, nb_classes)
X_train, X_test, Y_train, Y_test = train_test_split(pics, labels, test_size=0.1, random_state=5)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
time1 = time()
print('图片文件信息获取完毕, 耗时%d秒, 开始提取特征...' % (time1-time0))

#创建hdf5文件和表格
class Weights(tables.IsDescription):
	value  = tables.Float32Col(shape=(1, 512, 14, 14))   # float  (single-precision)
	label = tables.Int8Col(shape=(nb_classes))	
train = tables.open_file('Error_train.h5', 'w')
test = tables.open_file('Error_test.h5', 'w')
train_table = train.create_table('/', 'Train_dataset', Weights)
test_table = test.create_table('/', 'Test_dataset', Weights)
train_row = train_table.row
test_row = test_table.row

rgb_mean = np.array([103.939, 116.779, 123.68])
rgb_mean = rgb_mean.reshape((1,3,1,1))
#获取图片的值
def pic_trans(X):			
	img = Image.open(X).convert('RGB')
	img = img.resize((224,224))
	img = img_to_array(img)	
	img = np.expand_dims(img, axis=0)  # is equivalent to ``x[np.newaxis,:]`` or ``x[np.newaxis]``
	img -= rgb_mean
	return img

#提取训练数据Bottleneck特征	
nb_train = len(X_train)
nb_test = len(X_test)
for i in tqdm(range(nb_train)):	
	img = X_train[i]
	value = pic_trans(img)
	feature = model.predict(value) #形状(1, 512, 14, 14)
	label = Y_train[i]
	train_row['value'] = feature
	train_row['label'] = label	
	train_row.append()
	train_table.flush()
train.close()

#提取测试数据Bottleneck特征	
for i in tqdm(range(nb_test)):	
	img = X_test[i]
	value = pic_trans(img)
	feature = model.predict(value) #形状(1, 512, 14, 14)
	label = Y_test[i]
	test_row['value'] = feature
	test_row['label'] = label		
	test_row.append()
	test_table.flush()
test.close()
print('图片特征提取完毕, 耗时%d秒' % (time()-time1)) 









