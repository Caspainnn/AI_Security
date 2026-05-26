import os
import numpy as np
import cv2
import random
from model import MiniVGG
from imutils import paths  # 方便获取图像文件的路径列表

from sklearn.metrics import classification_report,confusion_matrix
# from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import Adam


from tensorflow.python.keras.utils.np_utils import to_categorical  # 用于将标签转换为one-hot编码
# from keras_preprocessing.image import img_to_array,ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split    # 用于划分训练集和测试集
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 禁用GPU

img_height=128
img_width=128
EPOCHS=10
num_classes=2
INIT_LR=1e-3    # 初始学习率
BS=32         # 批量大小

data=[]
labels=[]
# imagePaths=sorted(list(paths.list_images(r"实践3-2\NUAA\train")))
# random.seed(42)
# random.shuffle(imagePaths)   # 随机打乱图像路径列表

# # 加载图像
# for imagePath in imagePaths:
#     image=cv2.imread(imagePath)
#     image=cv2.resize(image,(img_height,img_width))
#     image=img_to_array(image)
    
#     data.append(image)
#     label=imagePath.split(os.path.sep)[-2]
#     label=1 if label=="fake" else 0
#     labels.append(label)

# 修改为如下：
with open(r"Lab3-2\NUAA\train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    random.seed(42)
    random.shuffle(lines)  # 随机打乱

    for line in lines:
        path, label = line.strip().split(',')
        # 路径兼容
        path = path.replace('/', os.sep)
        image = cv2.imread(path)
        if image is None:
            print(f"无法读取图片: {path}")
            continue
        image = cv2.resize(image, (img_height, img_width))
        image = img_to_array(image)
        data.append(image)
        labels.append(int(label))

# 图像预处理
data=np.array(data,dtype="float")/255.0
np.save('Lab3-2/data.npy',data)    # .npy是使用 NumPy 保存的二进制文件，里面存储的是所有图像数据的数组
labels=np.array(labels)
np.save('Lab3-2/labels.npy',labels)

# 重新加载数据
data=np.load('Lab3-2/data.npy')
labels=np.load('Lab3-2/labels.npy')

# 划分训练集和测试集
(trainX,testX,trainY,testY)=train_test_split(
    data,labels,test_size=0.25,random_state=42)
channels=trainX.shape[3]

# 将标签转换为one-hot编码
trainY=to_categorical(trainY,num_classes)
testY=to_categorical(testY,num_classes)

# 数据增强器 aug
aug=ImageDataGenerator(rotation_range=30,   # 旋转范围+-30度
    width_shift_range=0.1,height_shift_range=0.1,   # 水平和垂直平移范围 10%
    shear_range=0.2,zoom_range=0.2, # 随机错切变换 随机缩放图片
    horizontal_flip=True,fill_mode="nearest")  # 水平翻转 当图片变换后出现空白像素时，用最近的像素值填充

# 模型构建
print("\n=====================\n")
print("Compiling model...")
print("\n=====================\n")
model=MiniVGG(width=img_width,height=img_height,
            depth=channels,classes=num_classes)
# opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

# 模型训练
print("\n=====================\n")
print("Training network...")
print("\n=====================\n")
H=model.fit(aug.flow(trainX,trainY,batch_size=BS),
    validation_data=(testX,testY),
    epochs=EPOCHS,verbose=1)
label_name=["real","fake"]

# 模型评估
print("\n=====================\n")
print("[INFO] evaluating network...")
print("\n=====================\n")
predictions=model.predict(testX,batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1)))
cm=confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1))
total=sum(sum(cm))

model.save('Lab3-2/model.h5')
print("finish")
