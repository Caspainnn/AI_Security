# 从图像中提取特征和目标值，写入csv
import config
import cv2
import os
import random
import csv
from config import TRAIN_PATH,CLEANED_PATH
import numpy as np


# 对图像进行模糊处理和阈值化
def blur_and_threshold(image,eps=1e-7):
    blur=cv2.medianBlur(image,5)
    foreground=image.astype("float")-blur
    foreground[foreground>0]=0
    minVal=np.min(foreground)
    maxVal=np.max(foreground)
    foreground=(foreground-minVal)/(maxVal-minVal+eps)
    return foreground


imagePaths = [
    (os.path.join(TRAIN_PATH, train_file), 
     os.path.join(CLEANED_PATH, clean_file))
    for train_file, clean_file in zip(os.listdir(TRAIN_PATH), os.listdir(CLEANED_PATH))
]

if __name__ == "__main__":
    # 使用两个嵌套的循环遍历图像中的每个像素
    with open("./Lab5-1/features.csv", "w", newline="") as f:
        writer=csv.writer(f)   # 使用csv.writer，自动处理逗号等特殊字符
        
        # 遍历每个图像
        for (i,(trainPath,cleanedPath)) in enumerate(imagePaths):
            # 读取图像
            trainImage=cv2.imread(trainPath)
            cleanImage=cv2.imread(cleanedPath)
            # 转换为灰度图像
            trainImage=cv2.cvtColor(trainImage,cv2.COLOR_BGR2GRAY)
            cleanImage=cv2.cvtColor(cleanImage,cv2.COLOR_BGR2GRAY)
            # 对图像进行边界复制  作用：避免卷积 / 滤波时的边缘信息丢失
            trainImage=cv2.copyMakeBorder(trainImage,2,2,2,2,cv2.BORDER_REPLICATE) # 给图像上下左右各增加 2 个像素的边界
            cleanImage=cv2.copyMakeBorder(cleanImage,2,2,2,2,cv2.BORDER_REPLICATE)
            
            trainImage=blur_and_threshold(trainImage)   # 模糊处理
            if i==1:
                print("---trainImage.shape:",trainImage.shape)
            cleanImage=cleanImage.astype("float") / 255.0   # 归一化
            
            # 遍历每个像素
            for y in range(0,trainImage.shape[0]):
                for x in range(0,trainImage.shape[1]):
                    trainROI=trainImage[y:y+5,x:x+5]    # 每个像素的 5x5 区域被视为一个特征样本
                    cleanROI=cleanImage[y:y+5,x:x+5]
                    (rH,rW)=trainROI.shape[:2]
                    if rW!=5 or rH!=5:
                        continue
                    
                    features=trainROI.flatten()
                    target=cleanROI[2,2]
                    
                    if random.random()<config.SAMPLE_PROB:
                        # 构造行：[目标值] + [特征列表]（纯数值列表）
                        features_list = features.tolist()  # 把(25,)的numpy数组转为长度25的Python列表
                        row = [target] + features_list
                        # 使用csv.writer写入
                        writer.writerow(row)
                        
                        # features=[str(x) for x in features]
                        # row=[str(target)]+features
                        # row=",".join(row)
                        # csv.write("{}\n".format(row))

    print("Finished building features.csv")
            
        