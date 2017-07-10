# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:48:54 2016

@author: qiulongsheng
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.feature_selection import f_classif


data = pd.read_excel('理财产品数据.xlsx')
dic = {'预期收益率':'(%)','投资期限':'(天)','总分':''}

#查看某个字段的频率统计:
def xdes(x):
    f = open(x+'字段统计.csv', 'w')
    #1 统计描述
    des = {}
    des['个案数量'] = data[x].count()
    des['均值'] = data[x].mean()
    des['众数'] = data[x].mode().values[0]
    des['标准差'] = data[x].std()
    des['最小值'] = data[x].min()
    des['最大值'] = data[x].max()
    des['分位数'] = list(data[x].quantile([.25, .5, .75]).items())
    f.writelines(x+'信息统计\n')
    des = pd.Series(des)
    print(des)
    des.to_csv(f)
    f.write('\n')
    f.close()

def fig(x='预期收益率'):   #预期收益率分布图
    d = data[x].dropna()
    d.sort_values(inplace=True)
    ordered = d[5:-5]
    n = ordered.size
    value = ordered.value_counts()
    value_sort = value.sort_index()

    ax = plt.subplot()
    ax.hist(ordered,bins=value_sort.index,color='g')
    ax.set_xlabel(x+dic[x],fontsize='x-large')
    ax.set_ylabel('产品数量',fontsize='x-large')
    ax.set_xticks(np.linspace(ordered.min(),ordered.max(),20))
    ax.set_yticks(value_sort)

    ax2 = ax.twinx()    #双y轴
    ax2.plot(value_sort.index,np.cumsum(value_sort/n),color='r',linewidth=2.0)
    ax2.set_ylabel('累积百分比',fontsize='x-large')
    ax2.set_yticks(np.linspace(0,1,11))
    ax2.set_yticklabels([(str(i)+'%') for i in np.arange(0,101,10)])
    ax.grid(True)
    plt.subplots_adjust(left=0.04,right=0.95,top=0.96,bottom=0.07)
    font = {#'family': 'serif',
        'color':  'b',
        'weight': 'bold',
        'size': 12,
                }
    k = 5
    tx = value.index[:k]
    ty = value.values[:k]
    for h,v in zip(tx,ty):
        ax.text(h,v,str(h),fontdict=font)
    plt.title(x, fontdict=font,position=(0.475,0.9))
#    plt.savefig('x.png',bbox_inches='tight')

def scat(a='投资期限',b='预期收益率'):  # 投资期限与预期收益率的散点图
    data_ = data[data['投资期限']<800]
    plt.scatter(data_[a],data_[b],color='b',s=50,alpha=0.6)
#    plt.xlim(data[a].min()-10,data[a].max()+10)
    plt.xlabel(a,fontsize='x-large')
    plt.ylabel(b,fontsize='x-large')
    plt.xlim([data_[a].min(),data_[a].max()])
    plt.ylim([data_[b].min(),data_[b].max()])
    plt.grid(True)
    plt.subplots_adjust(left=0.04,right=0.98,top=0.98,bottom=0.07)

def pie(x): #分类变量饼状图
    value = data[x].value_counts()
    index = value.index
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    plt.pie(data[x].value_counts(),autopct='%.2f%%',labels=index,colors=colors
            , shadow=True, startangle=90)
    plt.title(x,weight='bold',size=18)
#    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(x+'.png',bbox_inches='tight')
#    plt.close()


def A2b(a,b='预期收益率'):   #以指定分类字段查看另一个字段平均值,变量关系直方图
    f = open(a+'对'+b+'影响分析.csv', 'w')
    b_avg = data.groupby(a)[b].mean()
    top = b_avg.sort_values(ascending=False).dropna()
    pd.DataFrame(top, columns=['预期收益率']).to_csv(f)
    f.write('\n')
    f.close()

    k = top.size
    plt.barh(np.arange(k),top.ix[-k:],color='y',height=0.8,align='center')#height=0.2,
    plt.yticks(np.arange(k),top.index[-k:])
    plt.xlabel('平均'+b+dic[b])
    plt.ylabel(a)
    plt.grid(True)

def bank(x='发行银行'): #top银行产品数量
    top = data[x].value_counts(ascending=True)
    k = 20
    plt.barh(np.linspace(0,k-1,k),top.iloc[-k:],color='y',height=0.8,align='edge')
    #如果这里x是名义变量，则ix和iloc都可以用，连续数值变量只能用iloc！！
    plt.xlabel('产品数量')
    plt.ylabel(x+'(%)')
    plt.yticks(np.linspace(0,k-1,k),top.index[-k:])
    plt.grid(True)

def topscore():
    score(k=10)
    top = data['总分'].sort_values().dropna()
    mins = data.ix[top[:20].index]
    maxs = data.ix[top[-20:].index]
    f = open('得分Top产品分析.csv', 'w')
    f.write('得分最低的20个产品\n')
    mins.to_csv(f)
    f.write('\n得分最高的20个产品\n')
    maxs.to_csv(f)
    f.close()


def yields():
    yields = data['预期收益率']
    yields.fillna(yields.mean(),inplace=True)
    ranges = yields.max()-yields.min()
    yields_score = (yields-yields.min())/ranges
    data['收益率得分'] = yields_score


def liquidity():
    data['投资期限'].replace(0,np.nan,inplace=True)
    liquidity = data['投资期限']
    liquidity.fillna(liquidity.mean(),inplace=True)
    ranges = liquidity.max()-liquidity.min()
    liquidity_score = (liquidity.max()-data['投资期限'])/ranges
    data['流动性得分'] = liquidity_score
#    data[data['客户是否有权提前赎回']=='是']['流动性得分']=1
#    data['流动性得分'].where(data['客户是否有权提前赎回']!='是',1,inplace=True)

def cluster(n_clusters=5):
    X = data[['投资期限','预期收益率']]
    X = X.fillna(X.mean())
    x = scale(X)
    clf = KMeans(n_clusters=n_clusters,random_state=0)
    y_pred= clf.fit_predict(x)
    data['分组'] = y_pred

    plt.scatter(X.ix[:,0],X.ix[:,1],c=y_pred,s=40,label=y_pred,cmap=plt.cm.Paired)
    plt.colorbar()

    scores = f_classif(x,y_pred)[0]
    print('f_classif_scores: ',f_classif(x,y_pred))
    data.to_excel('licai3.xlsx',index=False)
    return scores

def score(k):
    yields()
    liquidity()
    weights = cluster(k)
    weights = np.mat(weights)
    a = np.mat([data['流动性得分'],data['收益率得分']])
    scores = a.T * weights.T
    scores /= scores.max() * 0.01
    data['总分'] = np.round(scores,0)
    data.to_excel('licai3.xlsx',index=False)


def recommend(k):
    f = open('理财产品推荐.csv', 'w')
    f.write('\n产品推荐\n')
    score(k)
    groups = data['分组'].unique()
    for i in groups:
        group = data[data['分组']==i]
        qixian = group['投资期限'].min(),group['投资期限'].max()
        syl = group['预期收益率'].min(),group['预期收益率'].max()
        leixing = group['收益类型'].mode()[0]
        scores = group['总分'].mean()
        f.write('投资期限%d ~ %d天、预期收益率%.2f%% ~ %.2f%%、投资类型为"%s"的产品，推荐指数为：%d\n'%(qixian[0],qixian[1],syl[0],syl[1],leixing,scores))
        print('投资期限%d ~ %d天、预期收益率%.2f%% ~ %.2f%%、投资类型为"%s"的产品，推荐指数为：%d'%(qixian[0],qixian[1],syl[0],syl[1],leixing,scores))
    f.close()



