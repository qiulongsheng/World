# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:55:58 2016

@author: Atlantis
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t1 = pd.read_csv('Test5预测结果_Test.csv')
t2 = pd.read_csv('Test5预测结果_Error.csv')

t = pd.merge(t1, t2, on='name', how='outer')
t.to_csv('Test5预测结果_合并.csv')

top1 = t['prob1_x']
#print(top1.value_counts())
#预测概率小于0.7的
t[top1<0.7]
