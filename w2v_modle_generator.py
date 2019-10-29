#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description:训练生成word2vec模型

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from para_setting import *
import os


def train_word2vec_model(dim_w2v):
    time_start = time.time()
    temp_path = os.path.join(out_put_path, 'article_afterseg.txt')
    sentences = LineSentence(temp_path)
    model = Word2Vec(sentences,size=dim_w2v, window=5, min_count=1, iter=100, workers=8)
    model.wv.save_word2vec_format(w2v_model_path, binary=True)
    time_end = time.time()
    a=(time_end - time_start) / 60
    print('Word2Vec模型训练完成，总共耗时：%f分钟'%(a))
    print('模型已保存到%s'%(w2v_model_path))