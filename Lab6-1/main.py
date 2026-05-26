import os
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC  # sklearn 机器学习库，svm支持向量机
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯分类器

# 绘制混淆矩阵
import pandas as pd
import matplotlib.pyplot as plt

# 计算评估指标
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# 定义构建字典函数
def create_word_dictionary(train_directory):
    email_files=[os.path.join(train_directory,file) for file in os.listdir(train_directory)]    # 获取文件夹下的所有文件完整文件名
    all_words=[]
    
    for email_file in email_files:
        with open(email_file) as f:
            for i ,line in enumerate(f):  # enumerate()函数用于将可遍历的数据对象(如列表、字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
                if i==2:    # 邮件正文只从第三行开始，所以这里是从第三行开始读取
                    words=line.split()  # split()默认以空格为分隔符，将字符串拆分成单词列表
                    all_words+=words
    
    word_dictionary=Counter(all_words)  # Counter用于计数可哈希对象,这里用于词频统计
    words_to_remove=list(word_dictionary.keys())    # 获取所有的单词，相当于变为set
    
    for word in words_to_remove:    # 删除非纯字母单词和长度为1的单词
        if not word.isalpha() or len(word)==1:  # isalpha() 方法检测字符串是否只由字母组成，这里用于去除标点符号和数字
            del word_dictionary[word]
            
    word_dictionary=word_dictionary.most_common(3000)
    return word_dictionary
    
# 定义特征提取函数
def extract_features(mail_directory,word_dictionary):
    files = [os.path.join(mail_directory,file) for file in os.listdir(mail_directory)]  # 获取文件夹下的所有文件完整文件名
    features_maxtrix=np.zeros((len(files),3000))
    doc_id=0
    
    for file in files:
        with open(file) as f:
            for i,line in enumerate(f):
                if i==2:
                    words=line.split()
                    for word in words:
                        word_id=0
                        for i,d in enumerate(word_dictionary):  # 这里暴力遍历列表，我觉得可以把列表改为字典，就不必每次都遍历了
                            if d[0]==word:
                                word_id=i
                                features_maxtrix[doc_id,word_id]=words.count(word)
                                # 而且这里其实还可以剪枝，找到之后就break
        doc_id+=1
    return features_maxtrix

# 可视化混淆矩阵
def show_confusion_matrix(model_name,matrix):
    model_df=pd.DataFrame(matrix,index=['ham','spam'],columns=['ham','spam'])
    cell_text=[[model_name,'ham','spam']]
    for row_label,row in zip(['ham','spam'],model_df.values):
        cell_text.append([row_label]+list(row))
    fig,ax=plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    tbl=ax.table(cellText=cell_text,loc='center',cellLoc='center')
    plt.show()
    
# 计算评估指标
def calculate_metrics(model_name,result,test_labels):
    accuracy=accuracy_score(test_labels,result)
    precision=precision_score(test_labels,result)
    recall=recall_score(test_labels,result)
    f1=f1_score(test_labels,result)
    print(f"{model_name}模型评估指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")

# 指定训练集目录并创建单词字典
train_directory=r"实践6-1\data\train-mails"
word_dictionary=create_word_dictionary(train_directory)

# 定义训练标签
train_labels=np.zeros(702)
train_labels[351:701]=1  # 前351个是0，后350个是1(垃圾邮件)

# 提取训练集特征
train_matrix=extract_features(train_directory,word_dictionary)

# 初始化svm
svm_model=LinearSVC()

# 初始化朴素贝叶斯分类器
naive_bayes_model=MultinomialNB()

# 训练模型
svm_model.fit(train_matrix,train_labels)
naive_bayes_model.fit(train_matrix,train_labels)

# 指定测试集
test_directory=r"实践6-1\data\test-mails"
test_matrix=extract_features(test_directory,word_dictionary)

# 定义测试集标签
test_labels=np.zeros(260)
test_labels[130:260]=1  # 前130个是0，后130个是1(垃圾邮件)

# 使用svm预测
svm_result=svm_model.predict(test_matrix)

# 使用朴素贝叶斯预测
naive_bayes_result=naive_bayes_model.predict(test_matrix)

# 打印混淆矩阵
svm_cm=confusion_matrix(test_labels,svm_result)
print(svm_cm)
naive_bayes_cm=confusion_matrix(test_labels,naive_bayes_result)
print(naive_bayes_cm)

# 可视化混淆矩阵
show_confusion_matrix('SVM',svm_cm)
show_confusion_matrix('Naive Bayes',naive_bayes_cm)

# 计算评估指标
calculate_metrics('SVM',svm_result,test_labels)
calculate_metrics('Naive Bayes',naive_bayes_result,test_labels)
