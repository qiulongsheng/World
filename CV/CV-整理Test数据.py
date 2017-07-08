# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:58:06 2016

@author: Atlantis
"""

import os
import pandas as pd


path = 'F:\CV大赛\测试集\Testset 3'
pics = os.listdir(path)
answer = pd.read_csv('F:\CV大赛\测试集\Answer\BOT_Image_Testset 3.txt', header='infer', delimiter='\t')
desire = answer.icol(0).values
print('图片集数量: %d' % len(pics))
print('正确个数: %d' % len(desire))

i = 0
rm = []
for pic in pics:
	if pic[-4] == '.':		
		if pic[:-4] not in desire:
	
			i += 1
			rm.append(pic)
	else:
		if pic[:-5] not in desire:
			i += 1			
			rm.append(pic)
#for p in rm:
#	os.remove(os.path.join(path, p))			
print('移除张数: %d' % i)
print(rm)	
		

		