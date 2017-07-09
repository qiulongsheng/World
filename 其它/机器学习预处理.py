# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import os
import numpy as np
#import bolt
from sklearn import preprocessing
#from pyspark import SparkContext, SQLContext
import pandas as pd


x=pd.DataFrame(np.arange(20).reshape((4,5)))

#Spark的Dateframe转换为pandas.DataFrame
def spark2df(path):
    path = 'file:///usr/spark/spark-1.6.1-bin-hadoop2.6/df/mllib/sample_svm_df.txt'
    sc = SparkContext("local", "Simple App")
    sqlContext = SQLContext(sc)
    data  =  sqlContext.read.text(path)
    rdd = data.map(lambda line:line[0].split())
    pd = rdd.toDF().toPandas().astype(float)
    df = pd.ix[:, 1:]
    return df

'''
parameters:
df: pandas.dfFrame
column: 列名
'''
def log(df, *column):
    return df[list(column)].apply(np.log10)

def exp(df, *column):
    return df[list(column)].apply(np.exp)

def ln(df, *column):
    return df[list(column)].apply(np.log)

def power(df, **kwargs):
    return df[list(kwargs['column'])].apply(np.power(kwargs['p']))

def sqrt(df, *column):
    return df[list(column)].apply(np.sqrt)

#类型转换
def tostring(df, *column):
    return df[list(column)].apply(np.str)

def todecimal(df, *column):
    return df[list(column)].apply(np.float)

#日期运算
import datetime
def datediff(date1, date2):
    diff = date1 - date2
    return diff.days

#创建日期
def createdate(year, month, day):
    return datetime.datetime(year, month, day)

#字符串
'''
使用说明：
In [32]: a=pd.DataFrame([['dqww','erwg'],['geger','hterhg']],columns=['a','b'])
ltrim(a,'a','b',k=2)

Out[32]:
     a     b
0   ww    wg
1  ger  erhg
'''
def ltrim(df, *column, k):
    return df[list(column)].applymap(lambda item: item[k:])

def rtrim(df, *column, k):
    return df[list(column)].applymap(lambda item: item[:k])

def lenghth(df, *column):
    return df[list(column)].applymap(lambda item: len(item))

def lower(df, *column):
    return df[list(column)].applymap(lambda item: item.lower())

def upper(df, *column):
    return df[list(column)].applymap(lambda item: item.upper())

#重编码
def encode(df, *col):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(df[list(col)])

#一位有效编码one-of-K or one-hot encoding
def onehotencode(df, *col):
    enc = preprocessing.OneHotEncoder()
    return enc.fit_transform(df[list(col)])

#标准化
def std(df, *col):
    return preprocessing.scale(df[list(col)])

#归一化
def normal(df, *col):
    return preprocessing.MinMaxScaler().fit_transform(df[list(col)])

#二元化
#Binarize df (set feature values to 0 or 1) according to a threshold
def binarize(df, *col, th):
    binarizer = preprocessing.Binarizer(threshold=th)
    return binarizer.transform(df[list(col)])

#离散化
def discrete(df, col, bins, labels):
    '''
    x
Out[12]:
    0   1   2   3   4
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19

discrete(x,2,bins=[0,5,10,20],labels=['L','M','H'])
Out[13]:
0    L
1    M
2    H
3    H
Name: 2, dtype: category
Categories (3, object): [L < M < H]
    '''
    X = df[col]
    return pd.cut(X, bins=[-np.inf]+bins+[np.inf], labels=labels)


#缺失值
def fillna(df, *col, method):
    '''
    method:删除，均值，中位数，线性插值，点处线性趋势，序列均值
    '''
    if method == '删除':
        #按行删除
        return df[list(col)].dropna(axis=0, inplace=True)
        #按列删除
        return df[list(col)].dropna(axis=1, inplace=True)
    if method == '均值':
        fill_value = df[list(col)].mean(0)
        return df[list(col)].fillna(fill_value, inplace=True)
    elif method == '中位数':
        fill_value = df[list(col)].median(0)
        return df[list(col)].fillna(fill_value, inplace=True)
    elif method == '线性插值':
        return df[list(col)].interpolate(method='linear', inplace=True)
#   未实现
#if method == '点处线性趋势':
#        return df[list(col)].interpolate(method='polynomial',order=3)


#异常值
# 把异常值标记为np.inf, 再进行后续处理
def outlier(df, identify, method, *col, **keys):
    '''
    identify: 标记异常值的方法
    method: 处理异常值的方法
    **keys: 关键词(字典)参数, 依 赖上下文
    '''
    cols = df[list(col)]
    if identify == '标准差':
        #k个标准差以外的样本标为异常值
        mean = cols.mean()
        std_var = cols.std()
        lower = mean - keys['k']*std_var
        upper = mean + keys['k']*std_var
        cols[cols <= lower] = np.nan
        cols[cols >= upper] = np.nan
    elif identify == '百分比':
        cols = cols.sort_index()
        lower = cols.ix[int(keys['lower']*cols.shape[0]), :]
        upper = cols.ix[int(keys['upper']*cols.shape[0]), :]
        cols[cols <= lower] = np.nan
        cols[cols >= upper] = np.nan
    elif identify == '分界值':
        lower = keys['lower']
        upper = keys['upper']
        cols[cols <= lower] = np.nan
        cols[cols >= upper] = np.nan
    elif identify == '固定数量':
        # k指定该数量
        cols = cols.sort_index()
        cols.iloc[:keys['k']] = np.nan
        cols[keys['k']:] = np.nan

    #替换值通过keys参数传递
    if method == '替换':
        if keys['value'] == '均值':
            cols.replace(np.nan, df[list(col)].mean(axis=0), inplace=True)
        elif keys['value'] == '中位数':
            cols.replace(np.nan, df[list(col)].median(axis=0), inplace=True)
        df[list(col)] = cols
        return df
    elif method == '删除':
        df[list(col)] = cols
        #按行删除
        df = df.drop(df[pd.isnull(df).any(1)].index, axis=0, inplace=False)
        #按列删除
        df = df.drop(df.columns[pd.isnull(df).any(0)], axis=1, inplace=False)
        return df

    #返回处理完成的Dateframe


#特征加工
#特征过滤
from sklearn.feature_selection import VarianceThreshold
#方差阀值
def feature_sel(df, *col, threshold):
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    return sel.fit_transform(col)

#特征选择
from sklearn.feature_selection import SelectKBest, SelectPercentile
# k个特征
def selectBest(df,score_func, k=10, **col):
    '''
    score_func:
        f_classif, 分类任务
        f_regression, 回归任务
    k: 选择保留的特征个数
    返回: 选择的特征
    '''
    X_new = SelectKBest(score_func, k=k).fit_transform(df[col['X']], df[col['y']])
    return X_new
#百分比特征
def selectPercentile(df, score_func, percentile=10, **col):
    '''
    percentile: 选择保留的特征比例
    '''
    X_new = SelectPercentile(score_func, percentile=percentile).fit_transform(df[col['X']], df[col['y']])
    return X_new

#主成分分析
from sklearn.decomposition import PCA
def pca(df, *col, n_components, svd_solver):
    pca = PCA(n_components=n_components, whiten=True)
    return pca.fit_transform(df[col])


#穷举法特征搜索
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

'''
x = pd.DataFrame(np.arange(20).reshape((4,5)))
Out[36]:
    0   1   2   3   4
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19

selectFromModel(x, X=[0,1,2,3],y=4,method='逻辑回归')
Out[37]:
array([[ 0,  3],
       [ 5,  8],
       [10, 13],
       [15, 18]])
'''

def selectFromModel(df, method, **col):
    models = {
                '线性回归': LinearRegression,
                '朴素贝叶斯': GaussianNB,
                '逻辑回归': LogisticRegression,
                'SVM': SVC
                }
    model = SelectFromModel(models[method]())
    X_new = model.fit_transform(df[col['X']].values, df[col['y']].values.ravel())
    return X_new



import json
def reindex(df, js_file):
#    js = [{"k":"0","v":"0_"},{"k":"1","v":"1_"}]

    js = json.load(open('na.json'))
    mapping = dict([(i['k'],i['v']) for i in js])
    df_ = df.rename_axis(mapping, axis="columns")
    return pd.concat([df,df_], axis=1)

