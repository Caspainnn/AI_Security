# 把前面的 model + train 模块串起来、完成 成员推理攻击（MIA） 的主流程

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import model
from model import init_params as w_init
from train import train_model,train_attack_model,prepare_attack_data
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score
import argparse
import os

# 环境设置+数据预处理
np.random.seed(1234)
need_earlystop=False

# 模型超参数设置
target_filters=[128,256,256]
shadow_filters=[64,128,128]
n_fc=[256,128]
num_classes=10
num_epochs=50
batch_size=128
learning_rate=0.0001
lr_decay=0.96
reg=1e-4
shadow_split=0.6
n_validation=1000
num_workers=2
n_hidden_mnist=32

# 攻击模型超参数设置
NUM_EPOCHS=50
BATCH_SIZE=64
LR_ATTACK=0.001
REG=1e-6
LR_DECAY=0.95
n_hidden=512
out_classes=2

# 解析命令行参数
def get_cmd_arguments():
    parser=argparse.ArgumentParser(prog='Membership Inference Attack')
    # 创建一个解析器 parser，用于解析命令行参数。prog 是程序名（显示在帮助信息里）
    parser.add_argument('--dataset',default='CIFAR10',type=str,choices=['CIFAR10','MNIST'],
                        help='Which dataset to use (CIFARiO or MNIST)')
    # 添加一个可选参数 --dataset。指定使用哪个数据集
    parser.add_argument('--dataPath',default='./Lab10-1/data',type=str,
                        help='Path to store dataset')
    parser.add_argument('--model_path',default='./Lab10-1/model',type=str,
                        help='Path to save or load model checkpoints')
    parser.add_argument('--trainTargetModel',action='store_true',
                        help='Train target model,if false then load an already trained model')
    parser.add_argument('--trainShadowModel',action='store_true',
                        help='Train shadow model,if false then load an already trained model')
    # 决定是否训练目标模型（True）还是加载已训练模型（False）。action='store_true'用来创建 布尔开关（flag）
    # 作用：这个参数只需要存在与否，不用额外传值，存在就 True，不存在就 False
    # 比如 命令行里 写了 --trainTargetModel → 对应变量 args.trainTargetModel 就是 True
    # 如果命令行里 没有写 --trainTargetModel → 对应变量就是 False
    parser.add_argument('--need_augment',action='store_true',
                        help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--need_topk',action='store_true',
                        help='Flag to enable custom model params initialization')
    parser.add_argument('--verbose',action='store_true',
                        help='Add Verbosity')
    return parser.parse_args()


# 获取数据转换方法 根据是否需要数据增强来选择不同的转换方式
def get_data_transforms(dataset,augm=False):
    if dataset=='CIFAR10':
        normalize=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        test_transforms=transforms.Compose([transforms.ToTensor(),normalize])
    
        if augm:
            train_transforms=transforms.Compose([transforms.RandomRotation(5),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:
            train_transforms=transforms.Compose([transforms.ToTensor(),normalize])
    else:
        test_transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,),(0.3081,))])
        if augm:
            train_transforms=torchvision.transforms.Compose([torchvision.transforms.RandomRotation(5),
                                                             torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize((0.1307,),(0.3081,))])
        else:
            train_transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize((0.1307,),(0.3081,))])
            
    return train_transforms,test_transforms


# 将数据集分为4个部分，分别用于目标模型和影子模型的训练和测试
def split_dataset(train_dataset):
    total_size=len(train_dataset)
    split1=total_size//4
    split2=split1*2
    split3=split1*3
    
    indices=list(range(total_size))
    np.random.shuffle(indices)
    
    s_train_idx=indices[:split1]
    s_test_idx=indices[split1:split2]
    t_train_idx=indices[split2:split3]
    t_test_idx=indices[split3:]
    
    return s_train_idx,s_test_idx,t_train_idx,t_test_idx


# 获取数据加载器，用于目标模型和影子模型的训练和测试
def get_data_loader(dataset,data_dir,batch,shadow_split=0.5,augm_required=False,num_workers=1):
    error_msg="[!] shadow_split should be in the range [0,1]."
    assert ((shadow_split>=0) and (shadow_split<=1)),error_msg
    
    train_transforms,test_transforms=get_data_transforms(dataset,augm_required)
    
    if dataset=='CIFAR10':
        train_set=torchvision.datasets.CIFAR10(root=data_dir,train=True,download=True,transform=train_transforms)
        test_set=torchvision.datasets.CIFAR10(root=data_dir,train=False,transform=test_transforms)
        s_train_idx,s_out_idx,t_train_idx,t_out_idx=split_dataset(train_set)
    else:
        train_set=torchvision.datasets.MNIST(root=data_dir,train=True,download=True,transform=train_transforms)
        test_set=torchvision.datasets.MNIST(root=data_dir,train=False,transform=test_transforms)
        s_train_idx,s_out_idx,t_train_idx,t_out_idx=split_dataset(train_set)
        
    s_train_sampler=SubsetRandomSampler(s_train_idx)
    s_out_sampler=SubsetRandomSampler(s_out_idx)
    t_train_sampler=SubsetRandomSampler(t_train_idx)
    t_out_sampler=SubsetRandomSampler(t_out_idx)
    
    if dataset=='CIFAR10':
        target_val_idx=t_out_idx[:n_validation]
        shadow_val_idx=s_out_idx[:n_validation]
        t_val_sampler=SubsetRandomSampler(target_val_idx)
        s_val_sampler=SubsetRandomSampler(shadow_val_idx)
    else:
        target_val_idx=t_out_idx[:n_validation]
        shadow_val_idx=s_out_idx[:n_validation]
        t_val_sampler=SubsetRandomSampler(target_val_idx)
        s_val_sampler=SubsetRandomSampler(shadow_val_idx)
        
    if dataset=='CIFAR10':
        t_train_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=t_train_sampler,
                                                   num_workers=num_workers)
        t_out_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=t_out_sampler,
                                                  num_workers=num_workers)
        t_val_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=t_val_sampler,
                                                  num_workers=num_workers)
        
        s_train_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=s_train_sampler,
                                                   num_workers=num_workers)
        s_out_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=s_out_sampler,
                                                  num_workers=num_workers)
        s_val_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=s_val_sampler,
                                                  num_workers=num_workers)
    else:
        t_train_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=t_train_sampler,
                                                   num_workers=num_workers)
        t_out_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=t_out_sampler,
                                                  num_workers=num_workers)
        t_val_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=t_val_sampler,
                                                  num_workers=num_workers)
        
        s_train_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=s_train_sampler,
                                                   num_workers=num_workers)
        s_out_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=s_out_sampler,
                                                  num_workers=num_workers)
        s_val_loader=torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=batch,
                                                  sampler=s_val_sampler,
                                                  num_workers=num_workers)
        
    print(f'Total Test samples in {dataset} dataset:{len(test_set)}')
    print(f'Total Train samples in {dataset} dataset:{len(train_set)}')
    print(f'Number of Target train samples:{len(t_train_sampler)}')
    print(f'Number of Target valid samples:{len(t_val_sampler)}')
    print(f'Number of Target test samples:{len(t_out_sampler)}')
    print(f'Number of Shadow train samples:{len(s_train_sampler)}')
    print(f'Number of Shadow valid samples:{len(s_val_sampler)}')
    print(f'Number of Shadow test samples:{len(s_out_sampler)}')
    
    return t_train_loader,t_val_loader,t_out_loader,s_train_loader,s_val_loader,s_out_loader


# 使用攻击模型进行推断 评估其性能
def attack_inference(model,test_X,test_Y,device):
    print('----Attack Model Testing----')
    targetnames=['Non-Member','Member']
    pred_y=[]
    true_y=[]
    X=torch.cat(test_X)
    Y=torch.cat(test_Y)
    inferdataset=TensorDataset(X,Y)
    dataloader=torch.utils.data.DataLoader(dataset=inferdataset,
                                           batch_size=50,
                                           shuffle=False,
                                           num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for i,(inputs,labels) in enumerate(dataloader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            pred_y.append(predicted.cpu())
            true_y.append(labels.cpu())
            
    attack_acc=correct/total
    print('Attack Test Accuracy is : {:.2f}%'.format(100*attack_acc))
    
    true_y=torch.cat(true_y).numpy()
    pred_y=torch.cat(pred_y).numpy()
    print('----Detailed Results----')
    print(classification_report(true_y,pred_y,target_names=targetnames))
    

# 创建攻击模型并执行训练和推断
def create_attack(dataset,dataPath,modelPath,trainTargetModel,trainShadowModel,need_augm,need_topk,param_init,verbose):
    # dataset=dataset
    # need_augm=need_augm
    # verbose=verbose
    top_k=need_topk
    
    if dataset=='CIFAR10':
        img_size=32
        input_dim=3
    else:
        img_size=28
        input_dim=1
    
    datasetDir=os.path.join(dataPath,dataset)
    modelDir=os.path.join(modelPath,dataset)
    
    if not os.path.exists(datasetDir):
        try:
            os.makedirs(datasetDir)
        except OSError:
            pass
    
    if not os.path.exists(modelDir):
        try:
            os.makedirs(modelDir)
        except OSError:
            pass
        
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ',device)
    
    t_train_loader,t_val_loader,t_test_loader,\
    s_train_loader,s_val_loader,s_test_loader=get_data_loader(dataset,datasetDir,batch_size,
                                                              shadow_split,need_augm,num_workers)
    
    if (trainTargetModel):
        if dataset=='CIFAR10':
            target_model=model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
        else:
            target_model=model.MNISTNet(input_dim,n_hidden_mnist,num_classes).to(device)
            
        if (param_init):
            target_model.apply(w_init)
        
        if verbose:
            print('----Target Model Architecture----')
            print(target_model)
            print('----Model Learnable Params----')
            for name,param in target_model.named_parameters():
                if param.requires_grad==True:
                    print("\t",name)
        
        loss=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(target_model.parameters(),lr=learning_rate,weight_decay=reg)
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay)
        
        targetX,targetY=train_model(target_model,t_train_loader,t_val_loader,t_test_loader,
                                    loss,optimizer,lr_scheduler,device,modelDir,verbose,
                                    num_epochs,top_k,need_earlystop,is_target=True)
    else:
        target_file=os.path.join(modelDir,'best_target_model.ckpt')
        print('Use Target model at the path ---> [{}]'.format(modelDir))
        if dataset=='CIFAR10':
            target_model=model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
        else:
            target_model=model.MNISTNet(input_dim,n_hidden_mnist,num_classes).to(device)
            
        target_model.load_state_dict(torch.load(target_file,map_location=torch.device('cpu')))
        print('----Peparing Attack Training data----')
        t_trainX,t_trainY=prepare_attack_data(target_model,t_train_loader,device,top_k)
        t_testX,t_testY=prepare_attack_data(target_model,t_test_loader,device,top_k,test_dataset=True)
        targetX=t_trainX+t_testX
        targetY=t_trainY+t_testY
        
        
    if (trainShadowModel):
        if dataset=='CIFAR10':
            shadow_model=model.ShadowNet(input_dim,shadow_filters,img_size,num_classes).to(device)
        else:
            n_shadow_hidden=16
            shadow_model=model.MNISTNet(input_dim,n_shadow_hidden,num_classes).to(device)
            
        if (param_init):
            shadow_model.apply(w_init)
        
        if verbose:
            print('----Shadow Model Architecture----')
            print(shadow_model)
            print('----Model Learnable Params----')
            for name,param in shadow_model.named_parameters():
                if param.requires_grad==True:
                    print("\t",name)
        
        shadow_loss=nn.CrossEntropyLoss()
        shadow_optimizer=torch.optim.Adam(shadow_model.parameters(),lr=learning_rate,weight_decay=reg)
        shadow_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(shadow_optimizer,gamma=lr_decay)
        
        shadowX,shadowY=train_model(shadow_model,s_train_loader,s_val_loader,s_test_loader,
                                    shadow_loss,shadow_optimizer,shadow_lr_scheduler,device,modelDir,verbose,
                                    num_epochs,top_k,need_earlystop,is_target=False)
    else:
        shadow_file=os.path.join(modelDir,'best_shadow_model.ckpt')
        print('Use Shadow model at the path ---> [{}]'.format(modelDir))
        if dataset=='CIFAR10':
            shadow_model=model.ShadowNet(input_dim,shadow_filters,img_size,num_classes).to(device)
        else:
            n_shadow_hidden=16
            shadow_model=model.MNISTNet(input_dim,n_shadow_hidden,num_classes).to(device)
            
        shadow_model.load_state_dict(torch.load(shadow_file,map_location=torch.device('cpu')))
        print('----Peparing Attack Training data----')
        s_trainX,s_trainY=prepare_attack_data(shadow_model,s_train_loader,device,top_k)
        s_testX,s_testY=prepare_attack_data(shadow_model,s_test_loader,device,top_k,test_dataset=True)
        shadowX=s_trainX+s_testX
        shadowY=s_trainY+s_testY
        
    input_size=shadowX[0].size(1)
    print('Input Featrue dim for Attack MOdel: [{}]'.format(input_size))
    
    attack_model=model.AttackMLP(input_size,n_hidden,out_classes).to(device)
    
    if (param_init):
        attack_model.apply(w_init)
        
    attack_loss=nn.CrossEntropyLoss()
    attack_optimizer=torch.optim.Adam(attack_model.parameters(),lr=LR_ATTACK,weight_decay=REG)
    attack_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(attack_optimizer,gamma=LR_DECAY)
    
    attackdataset=(shadowX,shadowY)
    
    attack_valacc=train_attack_model(attack_model,attackdataset,attack_loss,
                                     attack_optimizer,attack_lr_scheduler,device,modelDir,
                                     NUM_EPOCHS,BATCH_SIZE,num_workers,verbose)
    
    print('Validation Accuracy for the Best Attack Model is: {:.2f}%'.format(100*attack_valacc))
    
    attack_path=os.path.join(modelDir,'best_attack_model.ckpt')
    attack_model.load_state_dict(torch.load(attack_path,map_location=torch.device('cpu')))
    
    attack_inference(attack_model,targetX,targetY,device)
