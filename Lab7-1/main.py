import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from keras.models import Sequential
from keras.layers import Dense,LSTM,Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from config import NUMBER_OF_SAMPLES,TRAIN_LEN

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_baseline():
    model=Sequential([Bidirectional(LSTM(64,activation='tanh',kernel_regularizer='l2')),
                     Dense(128,activation='relu',kernel_regularizer='l2'),
                     Dense(1,activation="sigmoid",kernel_regularizer='l2')])
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

def plot_metrics(history,metric,title,ylabel,save_as):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_"+metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(["Train","Validation"],loc="best")
    plt.savefig("./Lab7-1/"+save_as)
    plt.show()

# 数据加载
# dataset_attack=input("请输入攻击流量路径:")
# dataset_normal=input("请输入正常流量路径:")
print("="*30)
print("开始加载数据...\n")
dataset_attack="./Lab7-1/data/dataset_attack.csv"
dataset_normal="./Lab7-1/data/dataset_normal.csv"
data_attack=pd.read_csv(dataset_attack,nrows=NUMBER_OF_SAMPLES)
data_normal=pd.read_csv(dataset_normal,nrows=NUMBER_OF_SAMPLES)

# 设置列名
columns=['frame.len','frame.protocols','ip.hdr_len','ip.len','ip.flags.rb','ip.flags.df','ip.flags.mf',
         'ip.frag_offset','ip.ttl','ip.proto','ip.src','ip.dst','tcp.srcport','tcp.dstport','tcp.len',
         'tcp.ack','tcp.flags.res','tcp.flags.ns','tcp.flags.cwr','tcp.flags.ecn','tcp.flags.urg',
         'tcp.flags.ack','tcp.flags.push','tcp.flags.reset','tcp.flags.syn','tcp.flags.fin','tcp.window_size',
         'tcp.time_delta','class']
data_attack.columns=columns
data_normal.columns=columns

# 删除无关列
drop_columns=['ip.src','ip.dst','frame.protocols']
data_normal.drop(columns=drop_columns,inplace=True)
data_attack.drop(columns=drop_columns,inplace=True)

# 定义特征
features=['frame.len','ip.hdr_len','ip.len','ip.flags.rb','ip.flags.df','ip.flags.mf','ip.frag_offset',
         'ip.ttl','ip.proto','tcp.srcport','tcp.dstport','tcp.len','tcp.ack','tcp.flags.res','tcp.flags.ns',
         'tcp.flags.cwr','tcp.flags.ecn','tcp.flags.urg','tcp.flags.ack','tcp.flags.push','tcp.flags.reset',
         'tcp.flags.syn','tcp.flags.fin','tcp.window_size','tcp.time_delta']

# 提取特征 x  标签 y
x=np.concatenate((data_normal[features].values,data_attack[features].values))
# concatenate 用于拼接数组的函数，它能将多个结构一致的数组（维度相同、除拼接轴外的其他维度长度一致）合并成一个新数组
# 这里的作用是将 “正常流量数据” 和 “攻击流量数据” 按行拼接，形成完整的特征集和标签集
Y=np.concatenate((data_normal['class'].values,data_attack['class'].values))

# 标准化
scaler=StandardScaler()  # 初始化标准化器
# 标准化 将特征数据转换为 “均值为 0、标准差为 1” 的标准正态分布 X_std = (X - 均值) / 标准差
# 区别归一化：将数据压缩到 指定范围（默认 [0,1]），消除量纲 	X_norm = (X - 最小值) / (最大值 - 最小值)
# 针对 网络流量分类 任务，标准化能够确保所有特征在相同的尺度上，避免某些特征对模型训练的影响过大，如果用归一化的话，可能有极端值的影响
X=scaler.fit_transform(x)
# 将特征正式标准化
# fit 阶段：计算 x 中 25 个特征（如 frame.len、tcp.srcport）各自的均值和标准差。
# transform 阶段：用每个特征的均值和标准差，将该特征的所有数值转换为 “均值 0、标准差 1” 的标准化值，最终得到 X（与 x 形状相同，但数值已标准化）。

# 转换标签
Y=np.where(Y=="attack",0,1)

# 转变LSTM输入数据
# 1. 获取总样本数（X是二维数组，X.shape[0]是行数，即总时间点数量）
samples=X.shape[0]

# 2. 计算可构建的序列样本数
# TRAIN_LEN是序列长度（如50，表示每个样本包含50个连续时间点的特征）
input_len=samples-TRAIN_LEN

# 3. 构建三维序列样本（核心步骤）
# 用列表推导式生成每个序列：每个样本是X中连续TRAIN_LEN行的特征
I=np.array([X[i:i+TRAIN_LEN] for i in range(input_len)])

# 划分训练集测试集
X_train,X_test,Y_train,Y_test,train_indices,test_indices=train_test_split(I,Y[TRAIN_LEN:],range(len(Y[TRAIN_LEN:])),test_size=0.2,random_state=42)

# 创建模型
model=create_baseline()

# 训练模型
print("="*30)
print("\n开始训练模型...\n")
history=model.fit(X_train,Y_train,epochs=5,validation_split=0.2,verbose=1)

# 绘制正确率和损失曲线
plot_metrics(history,"accuracy","BRNN Model Accuracy","Accuracy","BRNN_Model_Accuracy.png")
plot_metrics(history,"loss","BRNN Model Loss","Loss","BRNN_Model_Loss.png")

# 模型预测
print("="*30)
print("\n开始预测...\n")
predictions=model.predict(X_test,verbose=1).flatten().round()

# 混淆矩阵
conf_matrix=confusion_matrix(Y_test,predictions)
conf_matrix_df=pd.DataFrame(conf_matrix,index=["Attack","Normal"],columns=["Attack","Normal"])
# 关键：添加热力图绘制代码
plt.figure(figsize=(8, 6))  # 设置图片大小
sns.heatmap(
    conf_matrix_df, 
    annot=True,          # 显示数值
    fmt="d",             # 数值格式为整数
    cmap="Blues",        # 颜色主题
    cbar=True            # 显示颜色条
)
# 设置标题和坐标轴标签
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")  # 行标签：实际类别
plt.xlabel("Predicted Label")  # 列标签：预测类别

plt.title("Confusion Matrix")
plt.savefig("./Lab7-1/BRNN_Confusion_Matrix.png",dpi=400)
plt.show()

# 模型保存
model.save("./Lab7-1/BRNN_Model.keras")

# 评估模型
print("="*30)
print("\n开始评估模型...\n")
scores=model.evaluate(X_test,Y_test,verbose=0)
recall=conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
print(f"模型准确率: {scores[1]:.4f}")
print(f"模型召回率: {recall:.4f}")

# 保存预测结果
predictions_df=pd.DataFrame({'Predicted':predictions,'Actual':Y_test})
predictions_df['Predicted']=predictions_df['Predicted'].map({0:'attack',1:'normal'})
predictions_df['Actual']=predictions_df['Actual'].map({0:'attack',1:'normal'})

# 合并测试集数据和预测结果
test_data_subset=pd.concat([data_normal,data_attack]).iloc[test_indices,:10]
result_df=pd.concat([test_data_subset.reset_index(drop=True),predictions_df.reset_index(drop=True)],axis=1)
result_df.to_csv("./Lab7-1/BRNN_Predictions.csv",index=False)
