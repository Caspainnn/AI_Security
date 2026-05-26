import torch
import torch.nn as nn
import common

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()
        self.layer1=self.create_conv_block(1,64)    # 定义单个卷积块，输入通道为1，输出通道为64，以此类推
        self.layer2=self.create_conv_block(64,128)
        self.layer3=self.create_conv_block(128,256)
        self.layer4=self.create_conv_block(256,512)
        
        self.layer6=nn.Sequential(nn.Flatten(),     # 展平层，多维特征转一维
                                  nn.Linear(15360,4096),    # 全连接层
                                  nn.Dropout(0.2),
                                  nn.ReLU(),
                                  nn.Linear(4096,common.captcha_size * len(common.captcha_array))   # 全连接层，输出通道为验证码长度*验证码字符集大小
                                  )
    
    # 定义每个卷积块
    def create_conv_block(self,in_channels,out_channels):
        return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                            )
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer6(x)
        return x
    
    
