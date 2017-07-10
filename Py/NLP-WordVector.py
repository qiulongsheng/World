# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:43:42 2016

@author: Atlantis
"""
import numpy as np
import re
import os
import string
import pandas
import gensim
import jieba
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def fenci():    #对单个文本文件进行分词 ，返回 一个词袋
    empty = '将 于 一 地 的 和 及 也 之 在 只  仅 与 了 即 也 若 比 及 我 为 他 是 他们 就 都 以 到 她 '.split()
    symbol = string.punctuation +'－-[]％（） ～·，：、。；“”【】±–—①②' #ASCII 标点符号，空格和数字
#    k=20[:k]
    texts=[]
    dictionary = corpora.Dictionary()
    txt = open('数据集.txt','w+',encoding='mbcs')
    dirname = 'C:\MyPy\sogou\Reduced'
    for dir in os.listdir(dirname):
        for file in os.listdir(os.path.join(dirname,dir)):
            lines = []
            try:
                for line in  open(os.path.join(dirname,dir,file),encoding='mbcs').readlines():  #
                    line = re.sub('\d+(\.)?\d*','',line)    #去掉数字
                    line = re.sub('[a-zA-Z]+','',line)     #去掉字母
                    line = re.sub('[\\s]*','',line)       #去掉换行符
                    text = list(jieba.cut(line))
                    text2 = [i for i in text if i not in empty+list(symbol)]
                    if text2 != []:
                        lines += text2
                        for word in text2:
                            txt.write(word+' ')
    #                texts.remove()
            except UnicodeDecodeError as e: #Exception所有异常
                print (e)
                continue
            txt.write('\n')
            texts.append(lines)
    txt.close()
    dictionary.add_documents(texts)
    dictionary.save('词典.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('corpus', corpus)
    return texts,corpus



class Mycorpus(object):
    def __iter__(self):
        self.k=1
        for dir in os.listdir('C:\MyPy\sogou\Reduced'):
            for file in os.listdir(os.path.join('C:\MyPy\sogou\Reduced',dir))[:self.k]:
                 for line in fenci(os.path.join('C:\MyPy\sogou\Reduced',dir,file)):
                     yield dictionary.doc2bow(line)


def model():
    texts,corpus = fenci()
#    corpus = [dictionary.doc2bow(text) for text in texts]
#    corpora.MmCorpus.serialize('corpus', corpus) #保存语料库矩阵在硬盘
    #corpus = corpora.MmCorpus('corpus.mm')读取语料库

    tfidf = models.TfidfModel(corpus) #建立Tf-Idf模型
    corpus_tfidf = tfidf[corpus]    #转换语料库
    corpora.MmCorpus.serialize('corpus_tfidf',corpus_tfidf)  #保存corpus_tfidf

    n = 20

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n) # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    corpora.MmCorpus.serialize('corpus_lsi',corpus_lsi)
    print (lsi.print_topics(n)) #what do these two latent dimensions stand for?

    lda = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=n)
    corpus_lda = lda[corpus_tfidf]
    corpora.MmCorpus.serialize('corpus_lda',corpus_lda)
    print (lda.print_topics(n))


    wv = Word2Vec(texts,size=100,window=5,min_count=5,workers=4)
    wv.save('词向量')

    return texts,wv
def recompute():    #重新计算，从硬盘读取模型数据
    global dictionary
    dictionary = corpora.Dictionary.load('词典.dict')

    global corpus
    corpus = corpora.MmCorpus('corpus')

    global corpus_tfidf
    corpus_tfidf = corpora.MmCorpus('corpus_tfidf')


    global corpus_lsi
    corpus_lsi = corpora.MmCorpus('corpus_lsi')

    global corpus_lda
    corpus_lda = corpora.MmCorpus('corpus_lda')

    global wv
    wv = Word2Vec.load('词向量')


def xiangsi(k,n):    #查询k语料库中第n篇文档的相似文档
    m = [corpus,corpus_tfidf,corpus_lsi,corpus_lda]
    index = similarities.SparseMatrixSimilarity(m[k])
    sims = index[m[k][n]]
    s = pandas.Series(sims)
    s.sort()
    result = [[dictionary[i] for i in np.array(m[k][j],dtype=int)[:,0
]] for j in  s.index[-10:]]
    return result


def duanyu():
    with open('数据集.txt',encoding='mbcs') as f:
        a=f.readlines()
        bigram = models.Phrases(a)
        print(bigram[a[0]])

