# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 23:08:53 2016

@author: qiulongsheng@beyondsoft.com
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import subprocess

spiderdir = raw_input('输入爬虫项目的完整路径(如 C:\zhaoshang): '.encode('mbcs'))
spidername = raw_input('输入爬虫名称(如zhaoshang): '.encode('mbcs'))
os.chdir(spiderdir)
subprocess.call('scrapy crawl %s -o %s.csv' % ('a' + spidername,spidername))