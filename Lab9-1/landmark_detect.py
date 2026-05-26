# 特征点标记
import cv2
import dlib
import numpy as np

# 模型路径和图像缩放系数
PREDICTOR_PATH = "Lab9-1\data\shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1

# 初始化检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

# 获取人脸
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces("检测到多个人脸")
    if len(rects) == 0:
        raise NoFaces("没有检测到任何人脸")
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# 标记图像上的特征点
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for point in landmarks:
        pos = (point[0, 0], point[0, 1])
        cv2.circle(im, pos, radius=3, color=(0, 255, 255), thickness=-1)
    return im

# 读取图像并获取特征点
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, dsize=(im.shape[1] * SCALE_FACTOR,
                                 im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s

def main():
    image_path = input("输入图片的路径:")
    try:
        im, landmarks = read_im_and_landmarks(image_path)
        im_with_landmarks = annotate_landmarks(im, landmarks)
        output_path = r'Lab9-1\data\output\output_landmarked.jpg'
        cv2.imwrite(output_path, im_with_landmarks)
        print(f"特征点标记完成，结果保存为 {output_path}")
    except (TooManyFaces, NoFaces) as e:
        print(e)

if __name__ == '__main__':
    main()