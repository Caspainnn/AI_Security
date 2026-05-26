import sys
import cv2
import numpy as np
import pickle
import config  # 假设你有这个配置文件
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QPushButton, QLabel, QHBoxLayout, QVBoxLayout,QSizePolicy,
                            QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from build_features import blur_and_threshold  # 假设你有这个特征构建文件


# 后台处理线程，避免UI卡顿
class DenoiseThread(QThread):
    # 定义信号
    processing_finished = pyqtSignal(np.ndarray)  # 传递处理后的图像
    processing_error = pyqtSignal(str)           # 传递错误信息

    def __init__(self, image, model):
        super().__init__()
        self.image = image  # 原始灰度图像
        self.model = model  # 加载的模型

    def run(self):
        try:
            # 复制原始图像用于处理
            processed_image = self.image.copy()
            
            # 添加边框，确保能提取5x5的ROI
            processed_image = cv2.copyMakeBorder(
                processed_image, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
            
            # 预处理图像
            processed_image = blur_and_threshold(processed_image)
            
            # 提取图像特征
            roi_features = []
            for y in range(0, processed_image.shape[0]):
                for x in range(0, processed_image.shape[1]):
                    roi = processed_image[y:y+5, x:x+5]
                    (rH, rW) = roi.shape[:2]
                    if rW != 5 or rH != 5:
                        continue
                    
                    features = roi.flatten()
                    roi_features.append(features)
            
            # 如果没有提取到特征，抛出错误
            if not roi_features:
                self.processing_error.emit("无法提取图像特征，请尝试其他图片")
                return
            
            # 模型预测
            pixels = self.model.predict(roi_features)
            
            # 重塑为原始图像形状
            pixels = pixels.reshape(self.image.shape)
            
            # 转换为uint8类型
            output = (pixels * 255).astype("uint8")
            
            # 发送处理完成信号
            self.processing_finished.emit(output)
            
        except Exception as e:
            self.processing_error.emit(f"处理出错: {str(e)}")

class DenoiseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None  # 降噪模型
        self.original_image = None  # 原始图像
        self.init_ui()
        self.load_model()  # 加载降噪模型

    def init_ui(self):
        # 窗口基础设置
        self.setWindowTitle("图片降噪工具")
        self.resize(1000, 600)
        self.setMinimumSize(800, 500)

        # 中心容器与整体垂直布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        overall_layout = QVBoxLayout(central_widget)
        overall_layout.setSpacing(20)
        overall_layout.setContentsMargins(30, 30, 30, 30)

        # ---------------------- 1. 置顶按钮区 ----------------------
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setSpacing(15)

        self.select_btn = QPushButton("选择照片")
        self.denoise_btn = QPushButton("降噪")
        self.clear_btn = QPushButton("清空照片")
        
        # 初始禁用降噪按钮（未选择图片时）
        self.denoise_btn.setEnabled(False)

        # 绑定按钮事件
        self.select_btn.clicked.connect(self.select_image)
        self.denoise_btn.clicked.connect(self.process_denoise)
        self.clear_btn.clicked.connect(self.clear_images)

        btn_size = (120, 40)
        self.select_btn.setFixedSize(*btn_size)
        self.denoise_btn.setFixedSize(*btn_size)
        self.clear_btn.setFixedSize(*btn_size)

        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.denoise_btn)
        btn_layout.addWidget(self.clear_btn)
        overall_layout.addWidget(btn_widget)

        # ---------------------- 2. 图片显示区 ----------------------
        img_widget = QWidget()
        img_layout = QHBoxLayout(img_widget)
        img_layout.setSpacing(30)

        # 原始图片标签
        self.orig_img_label = QLabel("原始图片")
        self.orig_img_label.setAlignment(Qt.AlignCenter)
        self.orig_img_label.setStyleSheet("border: 1px solid #cccccc;")
        self.orig_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_img_label.setScaledContents(True)

        # 降噪后图片标签
        self.denoised_img_label = QLabel("降噪后图片")
        self.denoised_img_label.setAlignment(Qt.AlignCenter)
        self.denoised_img_label.setStyleSheet("border: 1px solid #cccccc;")
        self.denoised_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.denoised_img_label.setScaledContents(True)

        img_layout.addWidget(self.orig_img_label)
        img_layout.addWidget(self.denoised_img_label)
        overall_layout.addWidget(img_widget, stretch=1)

    def load_model(self):
        """加载降噪模型"""
        try:
            model_path = config.MODEL_PATH + ".pickle"
            self.model = pickle.loads(open(model_path, "rb").read())
            print(f"模型加载成功，预期特征数量: {self.model.n_features_in_}")
        except Exception as e:
            QMessageBox.warning(self, "模型加载失败", 
                              f"无法加载降噪模型: {str(e)}\n程序仍可运行，但降噪功能不可用")
            print(f"模型加载失败: {str(e)}")

    def select_image(self):
        """选择图片并显示在原始图片区域"""
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "./",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )

        if img_path:
            # 读取图片并转换为灰度图
            self.original_image = cv2.imread(img_path)
            if self.original_image is None:
                QMessageBox.warning(self, "错误", "无法读取选中的图片")
                return
                
            # 保存原始BGR图像用于显示，同时准备灰度图用于处理
            self.original_bgr = self.original_image.copy()
            self.original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # 显示原始图片
            self.display_image(self.original_bgr, self.orig_img_label)
            
            # 启用降噪按钮
            self.denoise_btn.setEnabled(True)
            
            # 清空右侧降噪图片区域
            self.denoised_img_label.setText("降噪后图片")
            self.denoised_img_label.setPixmap(QPixmap())

    def process_denoise(self):
        """处理降噪"""
        if self.original_gray is None or self.model is None:
            return
            
        # 禁用按钮防止重复操作
        self.select_btn.setEnabled(False)
        self.denoise_btn.setEnabled(False)
        
        # 显示处理中提示
        self.denoised_img_label.setText("处理中，请稍候...")
        
        # 创建并启动后台处理线程
        self.denoise_thread = DenoiseThread(self.original_gray, self.model)
        self.denoise_thread.processing_finished.connect(self.on_denoise_finished)
        self.denoise_thread.processing_error.connect(self.on_denoise_error)
        self.denoise_thread.start()

    def on_denoise_finished(self, denoised_image):
        """降噪完成回调"""
        # 显示降噪后的图片
        self.display_image(denoised_image, self.denoised_img_label, is_gray=True)
        
        # 恢复按钮状态
        self.select_btn.setEnabled(True)
        self.denoise_btn.setEnabled(True)

    def on_denoise_error(self, error_msg):
        """降噪错误回调"""
        # 显示错误信息
        QMessageBox.warning(self, "处理错误", error_msg)
        self.denoised_img_label.setText("降噪后图片")
        
        # 恢复按钮状态
        self.select_btn.setEnabled(True)
        self.denoise_btn.setEnabled(True)

    def clear_images(self):
        """清空图片显示"""
        self.orig_img_label.setText("原始图片")
        self.orig_img_label.setPixmap(QPixmap())
        self.denoised_img_label.setText("降噪后图片")
        self.denoised_img_label.setPixmap(QPixmap())
        self.original_image = None
        self.original_gray = None
        self.denoised_img = None
        self.denoise_btn.setEnabled(False)

    def display_image(self, image, label, is_gray=False):
        """在标签中显示图像"""
        try:
            # 如果是灰度图，转换为RGB格式
            if is_gray:
                # 转换为BGR格式以便显示
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img = image.copy()
                
            # 转换图像格式为QImage
            height, width, channel = img.shape
            bytes_per_line = channel * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            # 显示图像
            label.setPixmap(QPixmap.fromImage(q_img))
            label.setText("")  # 清空文字
        except Exception as e:
            label.setText("无法显示图像")
            print(f"图像显示错误: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DenoiseWindow()
    window.show()
    sys.exit(app.exec_())
    