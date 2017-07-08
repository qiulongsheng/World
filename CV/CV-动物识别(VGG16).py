# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:55:01 2016

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
from keras.optimizers import SGD, Adam, Adadelta,rmsprop

from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split

from sklearn import metrics
import h5py

batch_size = 32
nb_classes = 12
nb_epoch = 200
#data_augmentation = True

print('读取并准备图片数据...')
time0 = time()
labels = []
data = []
for dir_path, dir_name, files in os.walk('数据\转换图片'):
#for dir_path, dir_name, files in os.walk('数据/转换图片(大)'):
	#类别名称
	if dir_name:
		class_names = dir_name		
	#类别目录下的图片
	if not dir_name:			
		for file in files:			
			pic_name = os.path.join(dir_path, file)
			pic = Image.open(pic_name)			
			values = img_to_array(pic)
			values = values.astype('float32')
#			values /= 255
			data.append(values)
			labels.append(class_names.index(os.path.split(dir_path)[1]))
			
data = np.array(data)
#rgb_mean = data.mean(0).mean(2).mean(1)
rgb_mean = np.array([103.939, 116.779, 123.68])
rgb_mean = rgb_mean.reshape((1,3,1,1))
data -= rgb_mean
labels = to_categorical(labels, nb_classes)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=5)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('数据准备完毕，耗时%d秒' % (time()-time0))

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
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    #Remove the last two layers to get the 4096D activations
#    model.layers.pop()
#    model.layers.pop()
#    model.add(Dense(12, activation='softmax'))
    return model
				
print('模型开始编译...')
time1 = time()
model = VGG_16()
f = h5py.File('vgg16_weights.h5')
#f = h5py.File('/media/atlant/D238FDCE38FDB199/MyPy/'vgg16_weights.h5')
for k in range(f.attrs['nb_layers']-1):
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()

sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = rmsprop(lr=0.0001)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
time2 = time()
print('模型编译完毕，耗时：%d秒。\n模型开始训练...' % (time2-time1))

model.fit(X_train, Y_train,  batch_size=64, nb_epoch=32, verbose=1)
print('模型训练完毕，耗时：%d秒。\n模型开始运行预测程序...' % (time()-time2))

Y_pred = model.predict_classes(X_test)
Y_train_pred = model.predict_classes(X_train)
print('预测结果Y_pred: ', Y_pred)
print('实际结果Y_test: ', Y_test.argmax(1))
print('训练预测准确度: ', metrics.accuracy_score(Y_train_pred,Y_train.argmax(1)))
print('测试预测准确度: ', metrics.accuracy_score(Y_pred,Y_test.argmax(1)))


