#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description: 主函数：训练word2vec、词嵌入、文档向量计算、分类器训练

import time
from processor import *
from para_setting import *
from w2v_modle_generator import train_word2vec_model
import numpy as np

time_start = time.time()#计时
print('开始第一步处理...')
num_keyword=30#设置关键词数量
#step1:提取前N个关键词keywords_list；前N个关键词权重weight_list；所有关键词all_keywords_list；所有分词结果all_words
keywords_list,weight_list,all_keywords_list,all_words=keywords_extract_Tfidf(info_data_read(text_path),num_keyword)
print('第一步已完成，已处理语料总数为%d,每条语料提取关键词数量为%d'%(len(weight_list),num_keyword))
print('----------------------------------------------------------------------------------------')

print('开始第二步处理....')
#setp2:训练word2vec模型并通过模型获取关键词词向量
train_word2vec_model(dim_w2v)#训练得到word2vec模型
words_vectors=Wordvector_W2V(all_words)#将每条语料中的每个分词结果都进行词嵌入
print('进行词嵌入的所有词已保存到%s目录下'%(word_going_to_be_vec_path))
outputs = open(word_going_to_be_vec_path, 'w', encoding='utf-8')
for i in range(len(all_words)):
    outputs.write(str(all_words[i])+'\n')
print('进行词嵌入的所有词向量已保存到%s目录下'%(word_vec_path))
outputs = open(word_vec_path, 'w', encoding='utf-8')
for i in range(len(words_vectors)):
    outputs.write(str(words_vectors[i])+'\n')
print('第二步已完成，已训练%s模型,%d条语料的所有词语已全部嵌入'%(w2v_model_path,len(weight_list)))
print('----------------------------------------------------------------------------------------')

print('开始第三步处理....')
#step3:将所有词向量相加求平均,获得文档向量
doc_vectors_list=[]
sum=0
for i in range(len(words_vectors)):
    for item in words_vectors[i]:
        sum += np.array(item)
    new_vector=sum / len(words_vectors[i])
    sum = 0
    doc_vectors_list.append(new_vector)
doc_vectors_list=np.array(doc_vectors_list)
print('其中每个文档向量维度维为：',len(doc_vectors_list[0]))
print('所有word2vec均值文档向量已保存到%s目录下'%(doc_vectors_path))
outputs = open(doc_vectors_path, 'w', encoding='utf-8')
for i in range(len(doc_vectors_list)):
    outputs.write(str(doc_vectors_list[i])+'\n')
print('第三步已完成，已计算%d条语料的文档向量'%(len(weight_list)))
print('----------------------------------------------------------------------------------------')

print('开始第四步处理....')
#setp4：训练svm分类器对doc_vectors_list中的所有文档向量中的数据划分训练集与测试集，并分类
svm=cls_model('SVM',doc_vectors_list)#支持向量机
svm.split_train_test_function(0.5)#将数据集分为训练集与测试集合
svm.train()#训练模型,可在path.py中设置SVM分类器核函数:'poly''rbf''linear''sigmoid'
svm.evaluate()#评价模型
svm.plot_roc()#绘制roc曲线

#以下为其他分类器代码
# lr=cls_model('LR',doc_vectors_list)#逻辑回归
# lr.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# lr.train()#训练模型
# lr.evaluate()#评价模型
# lr.plot_roc()#绘制roc曲线

# nb=cls_model('NB',doc_vectors_list)#朴素贝叶斯
# nb.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# nb.train()#训练模型
# nb.evaluate()#评价模型


# dt=cls_model('DT',doc_vectors_list)#决策树
# dt.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# dt.train()#训练模型
# dt.evaluate()#评价模型


# knn=cls_model('KNN',doc_vectors_list)#K近邻
# knn.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# knn.train()#训练模型
# knn.evaluate()#评价模型


# rf=cls_model('RF',doc_vectors_list)#随机森林
# rf.split_train_test_function(0.5)#将数据集分为训练集与测试集合
# rf.train()#训练模型
# rf.evaluate()#评价模型


print('第四步已完成')
print('----------------------------------------------------------------------------------------')

time_end = time.time()
a=(time_end - time_start) / 60
print('任务完成，总共耗时为：%f分钟'%(a))

