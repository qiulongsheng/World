# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:15:54 2016

@author: Atlantis
"""
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

for directory in ['error1', 'error2', 'hidden']:
	for p in tqdm(os.listdir(os.path.join('F:\CV大赛\测试集\error', directory))):
		img = load_img(os.path.join('F:\CV大赛\测试集\error', directory, p)) # this is a PIL image
		x = img_to_array(img)  
		x = np.expand_dims(x, 0) 
		# the .flow() command below generates batches of randomly transformed images
		# and saves the results to the `preview/` directory
		i = 0
		for batch in datagen.flow(x, batch_size=1, save_prefix=p[0],
		                          save_to_dir='F:\CV大赛\测试集\error\generate', save_format='jpeg'):
		    i += 1
		    if i > 10:
		        break  # otherwise the generator would loop indefinitely