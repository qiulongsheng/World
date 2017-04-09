# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:14:58 2017

@author: Atlantis
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

client = pd.read_csv(r'S11综合理财20170102\tbclient_20170102.csv',
                     encoding='mbcs', header=None)
product = pd.read_csv(r'S11综合理财20170102\tbproduct_20170102.csv',
                     encoding='mbcs', header=None)
share = pd.read_csv(r'S11综合理财20170102\tbshare0_20170102.csv',
                     encoding='mbcs', header=None)

#返回指定id客户的指定列信息
def get_client(num, *column):
    client_no = client.ix[:, 1]
    return client[client_no==num].ix[:, column]

#返回指定代码产品的指定列信息
def get_product(code, *column):
    prd_code = product.ix[:, 1]
    return product[prd_code==code].ix[:, column]

def sort_by_date(df, column=14):
    return df.sort_values(by=column)


#所有拥有购买记录的客户编号
client_no = share.ix[:,1] #客户编号
unique = client_no.value_counts().index #34402个
#plt.plot(unique.values)

#添加“投资期限”列--154列
product[154]=[(pd.datetime.strptime(str(i),'%Y%m%d') -
        pd.datetime.strptime(str(j),'%Y%m%d')).days
         for i,j in zip(product[26],product[24])]

left_share = share.ix[:, [1,12,14,29]]
right_product = product.ix[:, [1,2,4,14,15,24,26,55,141,154]]
share_prod = pd.merge(left_share, right_product, left_on=12, right_on=1)
#print(share_prod.ix[0])

#两个数据库融合
groups = share_prod.groupby('1_x', as_index=True)
merge_grouped = groups.apply(sort_by_date, '14_x')
#print(merge_grouped.ix[:2])

#客户编号
client_no = merge_grouped.index.levels[0]


'''
可用列：
share，product :

'1_x' -- 客户编号,
 12   -- 产品代码,
'14_x'-- 购买日期,
 29   -- 购买金额,

'1_y'  -- 产品代码
  2    -- 产品归属类别
  4    -- 产品类别
'14_y' -- 产品面值
  15   -- 发行价格
  24   -- 产品成立日期
  26   -- 产品结束日期
  55   -- 风险等级
  141  -- 预期收益率#
  154  -- 投资期限
'''

#所有客户的单项记录矩阵
def clients_item(column, dtype):
    records = np.zeros([len(client_no), 120], dtype=dtype)
    for i,n in enumerate(client_no):
        client = merge_grouped.ix[n]
        record = client[column]
        length = record.shape[0] #记录个数
        records[i, :length] = record
    return records

#购买金额
cost = clients_item(29, float)
#风险等级
risk = clients_item(55, str)
#投资期限
days = clients_item(154, int)
d = days.ravel().nonzero()[0] #转化为一维列表
#收益率
#yields = clients_item(141, float)
#首次购买产品
first_buy = []
for i in client_no:
    r = merge_grouped.ix[i]
    first = r[12].iloc[0]
    first_buy.append(first.strip())
first_buy = pd.Series(first_buy)
#print(first_buy.value_counts()[:10]) #首次购买前10热门产品

#热门产品
prods = share.ix[:, 12]
#print(prods.value_counts())








