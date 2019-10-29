#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description: 文件路径与参数设置

#训练参数设置
dim_w2v=768#word2vec模型词向量维度
data_ration=0.5#设置训练集与测试集比例
kernel_func='poly'#svm核函数设置:'sigmoid''rbf''linear'

#训练文档路径
text_path ='data\\data.tsv'
#待分类文档路径
text_path2='data\\data2.tsv'
#分类器模型保存路径
model_save_path='output\\cls_model'
#w2v模型地址
w2v_model_path='output\\word2vec_model'
#添加的自定义词表
custom_dic_path='data\\custom_dic.txt'
#停用词表地址
stop_word_path='data\\stop_dic.txt'

#输出文件夹地址
out_put_path='output'
#被嵌入的词语保存地址
word_going_to_be_vec_path='output\\word_before_Eembedding.txt'
#词向量保存地址
word_vec_path='output\\word_vector.txt'
#文档向保存量地址
doc_vectors_path='output\\doc_vector.txt'

