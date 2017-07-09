# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:46:00 2016

@author: byq
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cosine
from sklearn import cluster
from PIL import Image
from skimage.io import imread, imshow
import logging
import pandas


#cv2.imread色彩通道还原
def color(img):
    k = np.empty_like(img)
    k[:,:,0], k[:,:,1], k[:,:,2] = img[:,:,2], img[:,:,1], img[:,:,0]
    return k

#特征匹配
def match(img):
    features = r'D:\KSNSH\feature'
    records = []
    locations = {
                    '1.jpg': ['交易菜单', (10, 225)],
                    '4.jpg': ['理财产品购买', (802, 952)],
                    '5.jpg': ['宣读', (439, 541)],
                    '6.jpg': ['理财业务签解约', (533, 700)],
                    '7.jpg': ['综合业务系统', (519, 722)],
                    '8.jpg': ['信息补全窗口', (87, 179)]
                }

    for i in os.listdir(features):
        feature = cv2.imread(os.path.join(features, i))
        w = feature.shape[1] #, h =  feature.shape[0]
        res = cv2.matchTemplate(img, feature, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        if locations[i][1][0] - 10 < top_left[0] < locations[i][1][0] + 10 \
            and locations[i][1][1] - 10 < top_left[0] + w < locations[i][1][1] + 10:
            records.append([i, locations[i]])
    return records


#2016.12.22添加
features = []
local = r'D:\KSNSH\Features'
for f in os.listdir(local):
    p = os.path.join(local, f)
    img = cv2.imread(p)
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
    features.append(hist)

def hist_compare(h):
    difs = []
    for i in features:
        dif = cosine(i, h) #余弦距离度量屏幕画面的变化
        difs.append(dif)
    if (np.asarray(difs) >= 0.004).all():
        return True
    else:
        return False




#日志
logname = '录屏分析记录' +'.log' #日志文件名称
logfile = r'D:\KSNSH\lp.log' #os.path.join(flv_dir, logname)
logging.basicConfig(level = logging.INFO,
                    filename = logfile
                    )

file = {
        '文件名':[],
        '不合规类型':[],
        '宣读时间':[],
        '截图':[],
        '起始时间':[]
        }

rule = 10 #要求宣读时间
flv_dir = r'D:\KSNSH\录屏'
#创建目录保存未知界面
unknown_dir = r'D:\KSNSH\Unknown_screenshot'
if not os.path.exists(unknown_dir):
    os.mkdir(unknown_dir)

for f in os.listdir(flv_dir):
    if f.endswith('flv'):
        flv = os.path.join(flv_dir, f)
    else:
        continue
    cap = cv2.VideoCapture(flv)
    k = 0
    hists = []
    distance = []

    xd_when = [] #记录宣读发生时间
    xd_length = [] #记录宣读持续时间
    screenshot = {} #记录未知截图文件名
    while True:
        ret, img = cap.read()
        num_frame = cap.get(7) #帧数
        rate = cap.get(5) #码率
        if ret == True:
            #记录时间
            seconds = k // rate
            time = '%d-%d' % (seconds//60, seconds%60)
            #直方图
            hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
            if k == 0:
                record = match(img)
                if not record:
                    logging.info('录屏%s的%s处没有任何匹配特征' % (f, time))
                else:
                    action = [i[1][0] for i in record]
                    logging.info('\n录屏%s在%s处进行%s操作' % (f, time, str(action)))
            elif k > 0:
                diff = cosine(hist, hists[-1]) #余弦距离度量屏幕画面的变化
                distance.append(diff)
                if diff > 0.2:
                    if hist_compare(hist):
                        logging.info('\n录屏%s在%s处检测到未知界面' % (f, time))
                        unknown_pic = os.path.join(unknown_dir, '%s-%s.png' % (f, time))
                        screenshot[time] = unknown_pic
                        cv2.imwrite(unknown_pic, img)
#                    else:
                    record = match(img)
                    if not record:
                        logging.info('录屏%s的%s处没有任何匹配特征' % (f, time))
                    else:
                        action = [i[1][0] for i in record]
                        if not xd_when:
                            if '宣读' in action:
                                xd_when.append(seconds)
                                xd_start = '%d分%d秒' % (seconds//60, seconds%60)
                        else:
                            if not xd_length and '宣读' not in action:
                                xd_length.append(seconds - xd_when[0])
                        logging.info('录屏%s在%s处进行%s操作' % (f, time, str(action)))
            hists.append(hist)
            k += 1
        else:
            break


    if not xd_length:
        xd_length.append(0)
    elif xd_length[0] < rule:
        file['文件名'].append(f)
        file['不合规类型'].append(10)
        file['宣读时间'].append(xd_length[0])
        file['截图'].append('-')
        file['起始时间'].append(xd_start)

    if screenshot:
        for i in screenshot.items():
            file['文件名'].append(f)
            file['不合规类型'].append(20)
            file['宣读时间'].append('-')
            file['截图'].append(i[1])
            file['起始时间'].append(i[0].split('-')[0]+'分'+i[0].split('-')[1]+'秒')


    cap.release()
    cv2.destroyAllWindows()

#把违规记录存入csv文件
table = pandas.DataFrame(file)
table.to_csv('违规记录.csv', index_label='序号')




