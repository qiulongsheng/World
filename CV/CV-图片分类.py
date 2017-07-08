# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 01:39:53 2016

@author: Atlantis
"""

import os
import PIL
import cv2

pic_dir = 'F:\快盘\[Pictures]\图片\Sample Pictures'
names = []
pics = []
for pic in os.listdir(pic_dir):
	names.append(pic)
	img = cv2.imread(os.path.join(pic_dir, pic))