#!/usr/bin/env python
# coding: utf-8
# @Author  : Mr.K
# @Software: PyCharm Community Edition
# @Description:辅助函数

import os
import gensim
import jieba
import numpy as np
from para_setting import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
jieba.load_userdict(custom_dic_path)

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def info_data_read(filepath):
    with open(filepath,mode='r',encoding='utf-8') as f:
        s=f.read()
        s1=s.split()
        lines=[]
        for i in s1:
            lines.append(i)
        return lines

def processor_tfidf(text,num):
    text_cut = " ".join(jieba.cut(text,cut_all=False))
    stopwords = stopwordslist(stop_word_path)
    outstr = ''
    for word in text_cut:
        if word not in stopwords:
            outstr += word.lower()
        else:
            outstr +=" "
    corpus_zh = [outstr]
    outstr_list=corpus_zh[0].split()
    tfidfvec=TfidfVectorizer()
    tf_zh=tfidfvec.fit_transform(corpus_zh)
    words_bag=tfidfvec.get_feature_names()
    tf_zh_array = tf_zh.toarray().flat
    result_dic = {}
    for i in range(len(tf_zh_array)):
        key = "%s"%(words_bag[i])
        result_dic[key] = tf_zh_array[i]
    result_dic= sorted(result_dic.items(), key=lambda d:d[1], reverse = True)
    keyword_list=[]
    tfidf_weight_list = []
    everywords_list=[]
    for i in range(len(result_dic)):
            everywords_list.append(result_dic[i][0])
    for i in range(len(result_dic)):
        if i< num:
            keyword_list.append(result_dic[i][0])
            tfidf_weight_list.append(result_dic[i][1])
        i+=1
    return keyword_list,tfidf_weight_list,outstr,everywords_list,outstr_list


def keywords_extract_Tfidf(strings,num):
    key_words=[]
    weights=[]
    all_key_word=[]
    all_word=[]
    temp_path=os.path.join(out_put_path,'article_afterseg.txt')
    outputs = open(temp_path, 'w', encoding='utf-8')
    for sentence in strings:
        a,b,c,d,e=processor_tfidf(sentence,num)
        outputs.write(c + '\n')
        all_word.append(e)
        key_words.append(a)
        weights.append(b)
        all_key_word.append(d)
    print("所有分词结果已保存到%s,可用于word2vec模型训练"%temp_path)
    return key_words,weights,all_key_word,all_word

def Wordvector_W2V(keywords_list):
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True, unicode_errors='ignore')
    wordslist=[]
    for each_ele in keywords_list:
        wordslist.append(model.wv[each_ele])
    return wordslist

def split_train_test_function(input_doc_vectors,ration):
    total_num=len(input_doc_vectors)
    train_num=int((total_num/2)*ration)
    train_x=[]
    train_y=[]
    text_x=[]
    text_y=[]
    print('处理语料总量为',len(input_doc_vectors))
    print('训练集数量为',train_num*2)
    for i in range(0,int(total_num/2)):
        if i<=train_num-1:
            train_x.append(input_doc_vectors[i])
            train_y.append(0)
        if i>train_num-1:
            text_x.append(input_doc_vectors[i])
            text_y.append(0)
    for i in range(int(total_num/2),total_num):
        if i <= train_num+(total_num/2)-1:
            train_x.append(input_doc_vectors[i])
            train_y.append(1)
        if i > train_num+(total_num/2)-1:
            text_x.append(input_doc_vectors[i])
            text_y.append(1)
    return train_x,train_y,text_x,text_y

def ROC_curve_plot(y_test, y_score):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    #plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

class cls_model(object):
    def __init__(self,flag,data):
        self.flag=flag
        self.data=data

    def split_train_test_function(self,ration):
        total_num = len(self.data)
        train_num = int((total_num / 2) * ration)
        # test_num=int(total_num-train_num)
        train_x = []
        train_y = []
        text_x = []
        text_y = []
        print('处理语料总量为', len(self.data))
        print('训练集数量为', train_num * 2)
        for i in range(0, int(total_num / 2)):
            if i <= train_num - 1:
                train_x.append(self.data[i])
                train_y.append(0)
            if i > train_num - 1:
                text_x.append(self.data[i])
                text_y.append(0)
        for i in range(int(total_num / 2), total_num):
            if i <= train_num + (total_num / 2) - 1:
                train_x.append(self.data[i])
                train_y.append(1)
            if i > train_num + (total_num / 2) - 1:
                text_x.append(self.data[i])
                text_y.append(1)
        self.x_train=train_x
        self.y_train=train_y
        self.x_test=text_x
        self.y_test=text_y

    def train(self):
        if self.flag=='SVM':
            classifier = OneVsRestClassifier(SVC(kernel=kernel_func, probability=True, C=1.0, random_state=0, gamma=0.2))
            classifier.fit(self.x_train, self.y_train)
            self.score = classifier.decision_function(self.x_test)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(classifier, model_save_path)
        if self.flag=='RF':
            clf = RandomForestClassifier()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag=='NB':
            clf = GaussianNB()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag == 'DT':
            clf = DecisionTreeClassifier()
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf,model_save_path)
        if self.flag == 'LR':
            clf = LogisticRegression()
            clf.fit(self.x_train, self.y_train)
            self.score = clf.decision_function(self.x_test)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)
        if self.flag == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(self.x_train, self.y_train)
            print('模型已保存到%s'%(model_save_path))
            joblib.dump(clf, model_save_path)

    def evaluate(self):
        clf=joblib.load(model_save_path)
        result_set = [clf.predict(self.x_test), self.y_test]
        temp_path1 = os.path.join(out_put_path, 'true_lable.txt')
        temp_path2 = os.path.join(out_put_path, 'predict_lable.txt')
        np.savetxt(temp_path1, result_set[1], fmt='%.4e')
        np.savetxt(temp_path2, result_set[0], fmt='%.4e')
        print('%s分类器评价指标如下：'%(self.flag))
        print('Accuracy:\t', accuracy_score(result_set[1], result_set[0]))
        print('Precision:\t', precision_score(result_set[1], result_set[0]))
        print('Recall:\t', recall_score(result_set[1], result_set[0]))
        print('f1 score:\t', f1_score(result_set[1], result_set[0]))

    def plot_roc(self):
        num_Y_test = len(self.y_test)
        Y_test = np.array(self.y_test)
        Y_test = Y_test.reshape(num_Y_test, 1)
        ROC_curve_plot(Y_test, self.score)

    def predict(self):
        classifier = joblib.load(model_save_path)
        y_score = classifier.decision_function(self.data)
        result_set = [classifier.predict(self.data), y_score]
        temp_path3 = os.path.join(out_put_path, 'predict_result.txt')
        temp_path4 = os.path.join(out_put_path, 'predict_prob.txt')
        np.savetxt(temp_path3, result_set[0], fmt='%.4e')
        np.savetxt(temp_path4, result_set[1], fmt='%.4e')
        print('预测结果已保存到目录%s下' % (temp_path3))
        print('预测结果概率已保存到目录%s下' % (temp_path4))

