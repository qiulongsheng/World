# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:48:41 2016

@author: Atlantis
"""

from PIL import Image
import os
from tqdm import tqdm

#path = 'C:\MyPy\数据\计算机视觉大赛\狐狸'
#for pic in os.listdir(path):
#	img = Image.open(os.path.join(path,pic))
#	print(img.format,img.mode,img.size)
#	w, h = img.size
#print('平均宽度: %f,平均高度: %f' % (np.mean(w), np.mean(h)))

#以下程序用于原始VGG模型
#classes_sort = pd.value_counts(classes)
#f=open('数据\转换图片\classes.txt')
#c=np.array(f.read().split('\n'))
#result = c[classes_sort.index[:20]]
#print('预测结果', result)	
	
#转换并保存图片
#'C:\MyPy\数据\计算机视觉大赛'
path = r'C:\MyPy\数据\计算机视觉大赛'
#d=list(os.walk(path))
#创建12个存放图片的文件夹
#for i in d[0][1]:
#	os.mkdir('C:\MyPy\数据\转换图片\\' + i +'_')
size = []
for dir_path, dir_name, files in os.walk(path):
	#类别名称
	if dir_name:
		class_names = dir_name		
	#类别目录下的图片
	if not dir_name:
		i = 1		
		for file in tqdm(files):			
			pic_name = os.path.join(dir_path, file)
			pic = Image.open(pic_name)
			if not pic_name.endswith('jpg'):
				pic = Image.open(pic_name).convert('RGB')
			else:		
				pic = Image.open(pic_name)
#			统计size
			#size.append(pic.size)
			#array([ 494.07721725,  411.98889692])
			pic = pic.resize((224,224))
			save_path = os.path.join('C:\MyPy\数据\转换图片',os.path.split(dir_path)[1]+'\\')
			pic.save(save_path + str(i) +'.jpg')
			i += 1	
			
#img=Image.open(r'数据\转换图片(大)\狐狸_\86.jpg')	
#values = img_to_array(img)
##values = values.astype('float32')
#values=values.reshape(1,3,244,195)
#v = values- rgb_mean
#model.predict(v)
#model.predict(v).argmax()	