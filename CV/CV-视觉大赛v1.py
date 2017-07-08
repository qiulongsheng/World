# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:50:18 2016

@author: Atlantis
"""

import os#, cv2
import numpy as np
from PIL import Image
from time import time
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD #, Adam, Adadelta,rmsprop

from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split

from sklearn import metrics
import h5py
from tqdm import tqdm


batch_size = 32
nb_classes = 12
nb_epoch = 200

print('读取并准备图片数据...')
time0 = time()

pics = []
labels = []
for dir_path, dir_name, files in os.walk('数据\转换图片'):
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
	

def VGG_16(weights_path=None):
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

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=True))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model
				
				
print('模型开始编译...')
time1 = time()
model = VGG_16()
f = h5py.File('vgg16_weights.h5')
#Ubuntu: f = h5py.File('/media/atlant/D238FDCE38FDB199/MyPy/'vgg16_weights.h5')
for k in range(f.attrs['nb_layers']-1):
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)#sgd = rmsprop(lr=0.0001)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
time2 = time()

print('模型编译完毕，耗时：%d秒。\n模型开始训练...' % (time2-time1))
nb_epoch = X_train.shape[0] // batch_size
for i in tqdm(range(nb_epoch)):
	X = pic_trans(X_train[i*batch_size:(i+1)*batch_size])
	Y = Y_train[i*batch_size:(i+1)*batch_size]	
	loss = model.train_on_batch(X, Y)
	print('loss: ',loss)
print('模型训练完毕，耗时：%d秒。\n模型开始运行预测程序...' % (time()-time2))

Y_train_pred = []
for X in tqdm(X_train[:500]):
	y = model.predict_classes(pic_trans([X]))
	Y_train_pred.append(y)	
Y_pred = []
for X in tqdm(X_test[:500]):
	y = model.predict_classes(pic_trans([X]))
	Y_pred.append(y)
print('训练预测准确度: ', metrics.accuracy_score(Y_train_pred,Y_train[:500].argmax(1)))
print('测试预测准确度: ', metrics.accuracy_score(Y_pred,Y_test[:500].argmax(1)))

model.save('CV0904.h5')

