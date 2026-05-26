import torch
import torch.nn as nn



# 下面两个函数用于计算卷积层和池化层操作后的输出尺寸
def size_conv(size,kernel,stride=1,padding=0):
    out=int(((size-kernel+2*padding)/stride)+1)
    return out

def size_max_pool(size,kernel,stride=None,padding=0):
    if stride == None:
        stride=kernel
    out=int(((size-kernel+2*padding)/stride)+1)
    return out




# 下面两个函数用来根据输入图像的尺寸，计算 CIFAR和MNIST 数据集上CNN模型最后一个卷积层输出的特征尺寸，为全连接层提供特征数量
def calc_feat_linear_cifar(size):
    feat=size_conv(size,3,1,1)
    feat=size_max_pool(feat,2,2)
    feat=size_conv(feat,3,1,1)
    out=size_max_pool(feat,2,2)
    return out
def calc_feat_linear_mnist(size):
    feat=size_conv(size,5,1)
    feat=size_max_pool(feat,2,2)
    feat=size_conv(feat,5,1)
    out=size_max_pool(feat,2,2)
    return out


# 权重初始化
def init_params(m):
    if isinstance(m,nn.Conv2d): # 卷积层使用kaiming 初始化
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        # Kaiming 初始化是为带 ReLU（或类似非线性）的层设计的，可以保持前向传播时信号的方差比较稳定，避免梯度消失/爆炸
        # mode='fan_out'：控制计算 scale 时用的“fan”（通道数）度量
        # nonlinearity='relu'：告诉初始化器激活函数是 ReLU，从而选择合适的 scale
        
        # 如果卷积层含偏置项 (bias)，将其初始化为 0 因为你没必要一开始就“瞎给”它一个随机偏移，反而可能扰乱训练
        if m.bias is not None:  
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.BatchNorm2d):  # 批量归一化层使用常量初始化
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
        # BatchNorm 通常含有两个可学习参数：weight（gamma，缩放系数）和 bias（beta，偏置项）
        # 将 weight 初始化为 1（即初始时不改变标准差），将 bias 初始化为 0（即不平移均值）
        # 这是常见且合理的做法，能让一开始 BatchNorm 的行为接近「恒等映射」，利于训练稳定
    elif isinstance(m,nn.Linear):   # 全连接层使用 Xavier 初始化
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)
        # Xavier（Glorot）正态初始化 初始化线性层的权重（.weight）。
        # Xavier 是为 sigmoid/tanh 这类对称激活设计的，但也常被用于线性层或在网络中以经验方式使用

# 使用 TargetNet 和 ShadowNet 针对 CIFAR 数据集设计自定义CNN模型
# 目标模型 TargetNet
class TargetNet(nn.Module):
    def __init__(self,input_dim,hidden_layers,size,out_classes):
        # hidden_layers 是一个列表，包含了每个卷积层的输出通道数
        # 比如本模型输入的为 target_filters=[128,256,256]（即hidden_layers接收到的）
        # 表示第一层卷积输出通道数为 128，第二层卷积输出通道数为 256，第三层卷积输出通道数为 256
        super(TargetNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_dim,out_channels=hidden_layers[0],kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0],out_channels=hidden_layers[1],kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        features=calc_feat_linear_cifar(size)
        # 定义全连接层（是由多个线性层组成的，这里就有两个线性层）
        # 这里的hidden_layers[2]是其中的隐藏层的特征数量
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2*hidden_layers[1]),hidden_layers[2]),
            # 相当于展平后，传给第一个线性层的特征数量为 (features**2*hidden_layers[1])
            # 这里的 features 是卷积层输出的特征尺寸，hidden_layers[1] 是卷积层的输出通道数
            # 就相当于输出了一堆纸片，每个纸片的面积是 features**2，纸片的数量是 hidden_layers[1]
            # 一个纸片的单位面积就代表一个特征
            nn.ReLU(inplace=True),
            # inplace=True：直接在输入上操作，不分配额外内存，节省显存
            # inplace=False: 会创建一个新的张量来存储输出，占用额外的内存
            # 一般建议：只有在确定不会影响梯度计算时再使用 inplace=True
            nn.Linear(hidden_layers[2],out_classes)
        )
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.classifier(out)
        return out

# 白盒影子模型 ShadowNet
class ShadowNet(nn.Module):
    def __init__(self,input_dim,hidden_layers,size,out_classes):
        super(ShadowNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_dim,out_channels=hidden_layers[0],kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0],out_channels=hidden_layers[1],kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        features=calc_feat_linear_cifar(size)
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2*hidden_layers[1]),hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2],out_classes)
        )
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.classifier(out)
        return out
        
        
# 
class MNISTNet(nn.Module):
    def __init__(self,input_dim,n_hidden,out_classes=10,size=28):
        super(MNISTNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_dim,out_channels=n_hidden,kernel_size=5),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=n_hidden,out_channels=n_hidden*2,kernel_size=5),
            nn.BatchNorm2d(n_hidden*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        features=calc_feat_linear_mnist(size)
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2*n_hidden*2),n_hidden*2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden*2,out_classes)
        )
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.classifier(out)
        return out
    
    
# 多层感知机MLP 攻击模型
class AttackMLP(nn.Module):
    def __init__(self,input_size,hidden_size=64,out_classes=2):
        super(AttackMLP,self).__init__()
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,out_classes)
        )
    
    def forward(self,x):
        out=self.classifier(x)
        return out
    
