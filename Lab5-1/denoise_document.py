# 去噪文档
import argparse
import config
import pickle
import random
import cv2
from imutils import paths
from build_features import blur_and_threshold
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QPushButton, QLabel, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


ap=argparse.ArgumentParser()    # 初始化命令行参数
ap.add_argument(    # 添加一个必须的命令行参数 --testing，用于指定测试图像所在的目录路径
    "-t",
    "--testing",
    required=True,
    help="path to directory of testing images",
)
ap.add_argument(    # 用于指定处理的测试图像数量，默认为10
    "-s",
    "--sample",     
    type=int,
    default=10,
    help="sample size for testing images",
)


args=vars(ap.parse_args())  # 解析命令行参数，并存储为字典
# 之后在命令行输入 python denoise_document.py -t<testing_images_directory> [-s <sample_size>] 即可运行





model=pickle.loads(open(config.MODEL_PATH+".pickle","rb").read())    # 加载训练好的模型

imagePaths=list(paths.list_images(args["testing"]))    # 获取测试图像目录下的所有图像路径
random.shuffle(imagePaths)    # 随机打乱图像路径列表
imagePaths=imagePaths[:args["sample"]]    # 取前 args["sample"] 个图像路径

# 加载好图像之后即可进行预处理
i=0
for imagePath in imagePaths:
    print("Processing {}".format(imagePath))
    image=cv2.imread(imagePath)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    # 转换为灰度图像
    orig=image.copy()
    
    image=cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_REPLICATE)
    image=blur_and_threshold(image)
    
    # 提取图像特征
    roiFeatures=[]
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            roi=image[y:y+5,x:x+5]
            (rH,rW)=roi.shape[:2]
            if rW!=5 or rH!=5:
                continue
            
            features=roi.flatten()
            roiFeatures.append(features)
    
    pixels=model.predict(roiFeatures)
    pixels=pixels.reshape(orig.shape)
    output=(pixels*255).astype("uint8")
    
    cv2.imwrite("./Lab5-1/output/Original_"+str(i)+".jpg",orig)
    cv2.imwrite("./Lab5-1/output/Denoised_"+str(i)+".jpg",output)
    print("  Output saved as Original_"+str(i)+".jpg and Denoised_"+str(i)+".jpg")
    i+=1
    
print("Done!")