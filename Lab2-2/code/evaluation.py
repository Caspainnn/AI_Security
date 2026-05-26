from __future__ import print_function
import argparse
import os
import gc
import sys
import xlwt
import random

# ==================== 彻底不导入 numpy ====================
# ==================== FIX 1: 自己定义 zero_gradients ====================
import torch
def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()
torch.autograd.gradcheck.zero_gradients = zero_gradients

# ==================== FIX 2: 攻击库 ====================
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp

from net import Net_s, Net_m, Net_l
SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--adv', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--manualSeed', type=int)
parser.add_argument('--target', action='store_true')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# ==========================================
# 【终极绝杀】完全不用numpy，纯PIL→Tensor，无任何类型冲突
# ==========================================
class FinalToTensor:
    def __call__(self, pic):
        # 完全不碰 numpy！
        tensor = torch.empty(1, 28, 28, dtype=torch.float32)
        for i in range(28):
            for j in range(28):
                tensor[0, i, j] = pic.getpixel((j, i)) / 255.0
        return tensor

testset = torchvision.datasets.MNIST(
    root='../dataset/', train=False, download=True,
    transform=FinalToTensor()
)

data_list = list(range(10000))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1,
    sampler=sp.SubsetRandomSampler(data_list),
    num_workers=0
)

device = torch.device("cuda:0" if opt.cuda else "cpu")

def test_adver(net, tar_net, attack, target):
    net.eval()
    tar_net.eval()

    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.25, nb_iter=120, eps_iter=0.02, clip_min=0.0, clip_max=1.0, targeted=opt.target)
    elif attack == 'PGD':
        adversary = PGDAttack(
            net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.25, nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=opt.target)
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.26, targeted=opt.target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net, num_classes=10, learning_rate=0.45,
            binary_search_steps=10, max_iterations=12, targeted=opt.target)

    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' % (100. * correct_netD.float() / total))

    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0

    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = tar_net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        if target:
            labels = torch.randint(0, 9, (1,), device=device)
            if predicted != labels:
                adv_inputs_ori = adversary.perturb(inputs, labels)
                L2_distance = torch.norm(adv_inputs_ori - inputs).item()
                total_L2_distance += L2_distance
                with torch.no_grad():
                    outputs = tar_net(adv_inputs_ori)
                    _, predicted = torch.max(outputs.data, 1)
                    total += 1
                    correct += (predicted == labels).sum()
        else:
            if predicted == labels:
                adv_inputs_ori = adversary.perturb(inputs, labels)
                L2_distance = torch.norm(adv_inputs_ori - inputs).item()
                total_L2_distance += L2_distance
                with torch.no_grad():
                    outputs = tar_net(adv_inputs_ori)
                    _, predicted = torch.max(outputs.data, 1)
                    total += 1
                    correct += (predicted == labels).sum()

    if target:
        print('Attack success rate: %.2f %%' % (100. * correct.float() / total))
    else:
        print('Attack success rate: %.2f %%' % (100.0 - 100. * correct.float() / total))
    print('l2 distance:  %.4f ' % (total_L2_distance / total))

# ==================== 模型加载（你原版路径不动） ====================
target_net = Net_m().to(device)
state_dict = torch.load(r'E:\self-cultivation\Python\AI_Security\Lab2-2\code\pretrained\net_m.pth', map_location=device)
target_net.load_state_dict(state_dict)
target_net.eval()

if opt.mode == 'black':
    attack_net = Net_l().to(device)
    state_dict = torch.load(r'E:\self-cultivation\Python\AI_Security\Lab2-2\code\pretrained\net_l.pth', map_location=device)
    attack_net.load_state_dict(state_dict)
elif opt.mode == 'white':
    attack_net = target_net
elif opt.mode == 'dast':
    attack_net = Net_l().to(device)
    attack_net = nn.DataParallel(attack_net)
    state_dict = torch.load(r'E:\self-cultivation\Python\AI_Security\Lab2-2\code\saved_model\netD_epoch_best.pth', map_location=device)
    attack_net.load_state_dict(state_dict)

test_adver(attack_net, target_net, opt.adv, opt.target)