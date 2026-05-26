# 训练提取到的特征值
import config
from config import n_estimatorsList
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from time import time
import matplotlib.pyplot as plt




def train(n_estimators=80):
    print("--------------Training model with {} estimators--------------".format(n_estimators))
    model=RandomForestRegressor(n_estimators=n_estimators)    
    model.fit(trainX,trainY)
    return model


def evaluate(model,testX):
    print("--------------Evaluating--------------")
    preds=model.predict(testX)
    rmse=np.sqrt(mean_squared_error(testY,preds))
    print("RMSE: {:.4f}".format(rmse))
    return rmse

def draw_table(timeList,rmseList):
    print("| {:^12} | {:^12} | {:^12} |".format("n_estimators","RMSE","Time"))
    print("| {:^12} | {:^12} | {:^12} |".format("------------","------------","------------"))
    for n_estimators,rmse,time in zip(n_estimatorsList,rmseList,timeList):
        print("| {:^12} | {:^12.4f} | {:^12.4f} |".format(n_estimators,rmse,time))
        
def draw_plot(timeList,rmseList):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(n_estimatorsList,timeList)
    plt.xlabel("n_estimators")
    plt.ylabel("Training time (seconds)")
    plt.subplot(1,2,2)
    plt.plot(n_estimatorsList,rmseList)
    plt.xlabel("n_estimators")
    plt.ylabel("RMSE")
    plt.show()

features=[]
targets=[]

for row in open(config.FEATURES_PATH):
    row=row.strip().split(",")
    row=[float(x) for x in row]
    target=row[0]
    pixels=row[1:]
    
    features.append(pixels)
    targets.append(target)

# 数据预处理，放入np会更高效
features=np.array(features,dtype="float")
targets=np.array(targets,dtype="float")


# 分割数据集
(trainX,testX,trainY,testY)=train_test_split(features,targets,test_size=0.25,random_state=42)


timeList=[]
rmseList=[]

for n_estimators in n_estimatorsList:
    # 训练模型
    start=time()
    model=train(n_estimators)
    end=time()
    print("Training time: {:.4f} seconds".format(end-start))
    timeList.append(end-start)
    # 评估模型
    rmse=evaluate(model,testX)
    rmseList.append(rmse)
    # 保存模型
    path=config.MODEL_PATH+"_"+str(n_estimators)+".pickle"
    f=open(path,"wb")
    f.write(pickle.dumps(model))    # pickle.dumps() 函数将模型对象序列化为字节流
    f.close()
    
model = pickle.loads(open(path, "rb").read())
    
draw_table(timeList,rmseList)
draw_plot(timeList,rmseList)