# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 23:15:18 2016

@author: qiulongsheng@beyondsoft.com
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import subprocess

os.chdir('C:\\')
name = raw_input('输入爬虫项目名称(如 zhaoshang): '.encode('mbcs'))
a = 'scrapy startproject ' + name
subprocess.call(a)
os.chdir(os.path.join(os.getcwd(), name))
domain = raw_input('输入理财产品网址域名(如 cmbchina.com): '.encode('mbcs'))
subprocess.call('scrapy genspider a' + name + ' ' + domain)
