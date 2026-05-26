# 配置文件
import os
BASE_PATH="./Lab5-1/denoising-dirty-documents"
TRAIN_PATH=BASE_PATH+"/train"  # sep 表示添加路径分隔符
CLEANED_PATH=BASE_PATH+"/train_cleaned"
FEATURES_PATH="./Lab5-1/features.csv"
SAMPLE_PROB=0.02    # 采样概率
MODEL_PATH="./Lab5-1/donoiser"   # 训练后去噪模型的保存路径
n_estimatorsList=[5,10,30,50,70,90]


