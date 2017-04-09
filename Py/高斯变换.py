# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:19:29 2017

@author: byq
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_boston
data = load_boston()

X = data['data']
y = data['target']