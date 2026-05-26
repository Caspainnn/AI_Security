# 工具文件
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array
# from train import model
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 禁用GPU 因为tf版本和电脑的不匹配




def predictperson():
    model = load_model('Lab3-2/model.h5', compile=False)
    video_capture=cv2.VideoCapture(0)  # 0表示默认开启摄像头
    while True:
        if cv2.waitKey(1) & 0xFF==ord('b'): # 判断是否按下B键，如果是，则结束
            break
        ret,frame=video_capture.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # 定义 opencv 人脸检测分类器
        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades +
                                        'haarcascade_frontalface_default.xml')
        faces=faceCascade.detectMultiScale(gray,
                                scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
        cv2.rectangle(frame,
                    (190,125), # 矩形框左上角坐标
                    (400,400), # 矩形框右下角坐标
                    (255,0,0), # 矩形框颜色
                    2) # 矩形框粗细
        
        # 使用putText函数在图像帧左上角绘制文字
        cv2.putText(frame,"Please put your face in the blue box",
                    (10,10),    # 文字坐标
                    cv2.FONT_HERSHEY_SIMPLEX,   # 字体
                    0.5,    # 字体大小
                    (0,0,255),   # 文字颜色
                    2)   # 文字粗细
        faces_inside_box=0  # 记录在框内检测到的人脸数量
        
        # 记录人脸数量
        for (x,y,w,h) in faces:
            if (x<400 and x>190 and y<400 and y>125 and (x+w)<400  and (x+w)>190
                and (y+h)<400 and (y+h)>125):
                faces_inside_box+=1 # 框内人脸数量加1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # 使用for循环，依次绘制检测到的人脸
        if faces_inside_box==1: # 如果框内检测到一张人脸
            # 确认是否在框内
            if (x<400 and x>190 and y<400 and y>125 and (x+w)<400  and (x+w)>190
                and (y+h)<400 and (y+h)>125):
                image=cv2.resize(frame,(128,128))
            
                # 图像数据转float并归一化
                image=image.astype("float")/255.0
                image=img_to_array(image)
                
                # 添加一个维度，变成(1,128,128,3)
                image=np.expand_dims(image,axis=0)
                (real,fake)=model.predict(image)[0]
            
            if fake>real:
                label="fake"
            else:
                label="real"

            # 在图像上绘制标签
            cv2.putText(frame,label,
                        (10,30),    # 文字坐标
                        cv2.FONT_HERSHEY_SIMPLEX,   # 字体
                        0.7,    # 字体大小
                        (0,0,255),   # 文字颜色
                        2)   # 文字粗细
            

        else:    # 如果人脸不在蓝色区域，则显示提示文字
            cv2.putText(frame,"Put your face in the blue box, now!",
                        (100,420),
                        cv2.FONT_HERSHEY_SIMPLEX,   # 字体
                        0.5,    # 字体大小
                        (0,0,255),   # 文字颜色
                        2)   # 文字粗细
        cv2.imshow("Frame",frame)   # 显示处理最后的图像帧
    
if __name__=="__main__":
    predictperson()