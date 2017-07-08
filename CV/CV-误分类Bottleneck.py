# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:07:10 2016

@author: Atlantis
"""


import os
import numpy as np
from PIL import Image
from time import time
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
from tqdm import tqdm

	
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


import tables 
class Weights(tables.IsDescription):
	#图片的名称和值都存入，用于对照answer评分
	name = tables.StringCol(itemsize=32)
	value  = tables.Float32Col(shape=(1, 512, 14, 14))   # float  (single-precision)
	
test = tables.open_file('error_features.h5', 'w')
test_table = test.create_table('/', 'error_dataset', Weights)
test_row = test_table.row


#准备数据
#Xs = np.zeros((8518, 3, 224, 224))
rgb_mean = np.array([103.939, 116.779, 123.68])
rgb_mean = rgb_mean.reshape((3,1,1))
print('读取并准备图片数据...')
time0 = time()
#Testset_3
path = 'F:\CV大赛\测试集\error'
imgs = os.listdir(path)
for img in tqdm(imgs):
	test_row['name'] = img
	img = os.path.join(path, img)
	img = Image.open(img).convert('RGB')
	img = img.resize((224,224))
	img = img_to_array(img)	
	img -= rgb_mean
	img = img.reshape((1,3,224,224))
	bottleneck_features_train = model.predict(img) #形状(1, 512, 14, 14)
	test_row['value'] = bottleneck_features_train
	test_row.append()
	test_table.flush()
test.close()	









