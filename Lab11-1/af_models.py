import torch.nn as nn
import torch

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 攻击模型
class AttackModel(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.classifier = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 目标模型
class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        y = self.output(x)
        return y, x