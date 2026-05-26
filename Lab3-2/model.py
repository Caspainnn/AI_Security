# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D,MaxPooling2D
# from keras.layers import BatchNormalization
# from keras.layers.core import Activation,Dropout,Flatten,Dense
# from keras import backend as K 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K


# 定义miniVGG模型
def MiniVGG(width,height,depth,classes):    # 接收图像的宽高深，类别
    model=Sequential()
    inputS=(height,width,depth)
    chanDim=-1  # 用于指定通道维度的索引

    # 判断图像数据格式是否为channels_first。若是，调整输入形状和通道维度
    if (K.image_data_format()=="channels_first"):
        inputS=(depth,height,width)
        chanDim=1
        
    # 添加卷积层
    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputS))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    # 添加最大池化层
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    # 添加最大池化层
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    
    # 添加激活函数层、BatchNomalization层和Dropout层来处理全连接层的输出
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    
    return model
    