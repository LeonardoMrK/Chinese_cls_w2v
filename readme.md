##### 概述：使用word2vec与常见分类器实现中文文本二分类

##### 特征提取方式：word2vec加权平均

##### 分类器：SVM，KNN，逻辑回归，随机森林，朴素贝叶斯，决策树

--------

##### 使用步骤：

1.将训练语料以及待分类语料整理为tsv格式，其中每条语料占一行，每行中间不可有空格、缩进等空白符号（具体格式请看data\data.tsv中的示例语料）

注意：训练语料不需要打标签，只需将正类放在文档前半部分，负类放在后半部分。比如示例语料中有1000条的训练语料集，其中有”汽车“”金融“两类文本各500条，只需将”汽车“类文本放在训练语料data.tsv的1-500行，将”金融“类文本放在501-1000行即可；待分类语料顺序不限

2.在para_setting.py中设置数据路径以及训练参数

3.运行main.py训练模型，并给出评价指标

4.运行predict.py对待分类文档进行分类作业

------

##### para_setting.py参数详解：

文档路径：

“text_path”训练文档路径，默认为'data\\data.tsv'
”text_path2“待分类文档路径，默认为='data\\data2.tsv'

训练参数：

”dim_w2v“为word2vec模型维度，即词向量维度；
“data_ration”训练集与测试集比例
“kernel_func”svm核函数类型，可设置为'poly''rbf''linear''sigmoid'

模型路径：

"model_save_path"为分类器模型保存路径,默认为'output\\cls_model'
“w2v_model_path”为word2vec模型路径，默认为'output\\word2vec_model'

停用词与自定义词路径：

“custom_dic_path”为分词自定义词表路径,默认为'data\\custom_dic.txt'(每一个自定义词单独占一行)
"stop_word_path"为停用词表路径，默认为'data\\stop_dic.txt'

输出文件路径：

”out_put_path“为输出文件夹路径，默认为'output'
“word_going_to_be_vec_path”为被嵌入的词语保存路径，默认为'output\\word_before_Eembedding.txt'
“word_vec_path”为词向量保存路径，默认为='output\\word_vector.txt'
“doc_vectors_path”为文档向量保存路径，默认为'output\\doc_vector.txt'

-----

##### 训练步骤：

（1）对text_path的文档进行分词处理（去除停用词，保留自定义词）

（2）使用上一步分词结果训练word2vec模型，将模型保存到w2v_model_path，并调用模型对词语进行词嵌入

（3）将每一条语料的词向量相加求平均，得到文档向量

（4）使用上一步的文档向量训练分类器，将模型保存到model_save_path，并进行评价

-----

##### 预测步骤：

（1）对text_path2的文档进行分词处理（去除停用词，保留自定义词）

（2）载入w2v_model_path中的word2vec模型进行词嵌入

（3）将每一条语料的词向量相加求平均，得到文档向量

（4）载入model_save_path中的分类器并进行预测