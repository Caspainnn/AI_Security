import pandas as pd
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import PIL

from af_train import training, test, test_class
from af_models import TargetModel, AttackModel
from af_datasets import UTKFace, AttackData

# 数据划分
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
PATH = 'Lab11-1/data/'
TEST_SPLIT = 0.2
ATTACK_SPLIT = 0.5

samples = pd.read_pickle('Lab11-1/data/UTKFaceDF.pkl')
np.random.seed(SEED)
dataset_size = len(samples)
indices = list(range(dataset_size))
split = int(np.floor(TEST_SPLIT * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

attack_split = int(np.floor(ATTACK_SPLIT * len(train_indices)))
np.random.shuffle(train_indices)
attack_indices, train_indices = train_indices[attack_split:], train_indices[:attack_split]

train_samples = samples.iloc[train_indices]
test_samples = samples.iloc[test_indices]
attack_samples = samples.iloc[attack_indices]

# 设备、超参、预处理
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_LEARNING_RATE = 0.001
TARGET_BATCH_SIZE = 128
ATTACK_LEARNING_RATE = 0.001
ATTACK_BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.Resize([50, 50]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 数据加载器
target_train_loader = DataLoader(
    UTKFace(train_samples, 'gender', transform), batch_size=TARGET_BATCH_SIZE
)
target_test_loader = DataLoader(
    UTKFace(test_samples, 'gender', transform), batch_size=TARGET_BATCH_SIZE
)

# 先创建模型，再传入数据集
target_model_attack = TargetModel().to(DEVICE)

attack_train_loader = DataLoader(
    AttackData(attack_samples, target_model_attack, transform), batch_size=ATTACK_BATCH_SIZE
)
attack_test_loader = DataLoader(
    AttackData(test_samples, target_model_attack, transform), batch_size=ATTACK_BATCH_SIZE
)

# 预训练模型攻击
def perform_pretrained_dummy():
    target_model_path = 'Lab11-1/models/target_model_' + str(30) + '.pth'
    attack_model_path = 'Lab11-1/models/attack_model_' + str(50) + '.pth'

    target_model = TargetModel().to('cpu')
    print('Loading Target Model...')
    target_model.load_state_dict(torch.load(target_model_path, map_location='cpu'))
    print('Testing Target Model...')
    test(target_test_loader, target_model, True)
    print()

    attack_model = AttackModel(64).to('cpu')
    print('Loading Attack Model...')
    attack_model.load_state_dict(torch.load(attack_model_path, map_location='cpu'))
    print('Testing Attack Model...')
    test(attack_test_loader, attack_model, False)
    test_class(attack_test_loader, attack_model, False)

# 训练目标+攻击模型
def perform_train_dummy(target_epochs, attack_epochs):
    target_model = TargetModel().to(DEVICE)
    target_criterion = nn.CrossEntropyLoss()
    target_optimizer = optim.Adam(target_model.parameters(), lr=TARGET_LEARNING_RATE)
    target_path = f'Lab11-1/models/target_model_{target_epochs}.pth'
    training(target_epochs, target_train_loader, target_optimizer, target_criterion,
             target_model, target_path, True)

    attack_model = AttackModel(64).to(DEVICE)
    attack_criterion = nn.CrossEntropyLoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=ATTACK_LEARNING_RATE)
    attack_path = f'Lab11-1/models/attack_model_{attack_epochs}.pth'
    training(attack_epochs, attack_train_loader, attack_optimizer, attack_criterion,
             attack_model, attack_path, False)

    perform_pretrained_dummy()

# 自定义目标模型攻击
def perform_supply_target(class_file, state_path, dimension, attack_epochs):
    try:
        module = __import__(class_file, globals(), locals(), ['TargetModel'])
    except ImportError:
        print('Target model class could not be imported...')
        return
    TargetModel = vars(module)['TargetModel']

    target_model = TargetModel().to('cpu')
    print('Loading Target Model...')
    target_model.load_state_dict(torch.load(state_path, map_location='cpu'))
    print('Testing Target Model...')
    test(target_test_loader, target_model, True)
    print()

    attack_model_path = f'Lab11-1/models/custom_attack_model_{attack_epochs}.pth'
    attack_model = AttackModel(dimension).to(DEVICE)
    attack_criterion = nn.CrossEntropyLoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=ATTACK_LEARNING_RATE)

    print('Training Attack Model for ' + str(attack_epochs) + ' epochs...')
    training(attack_epochs, attack_train_loader, attack_optimizer, attack_criterion,
             attack_model, attack_model_path, False)

    attack_model.to('cpu')
    print('Loading Attack Model...')
    attack_model.load_state_dict(torch.load(attack_model_path, map_location='cpu'))
    print('Testing Attack Model...')
    test(attack_test_loader, attack_model, False)
    test_class(attack_test_loader, attack_model, False)