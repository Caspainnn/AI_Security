'''
统一管理目标模型（TargetNet）、影子模型（ShadowNet）、攻击模型（AttackMLP）的训练与验证过程”
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from torch.utils.data import TensorDataset
import copy
import model



# 为攻击模型准备数据  后验概率+成员标签
# 攻击模型训练的目标：
# 学会根据后验分布判断一个样本是不是成员。
def prepare_attack_data(model,iterator,device,top_k=False,test_dataset=False):
    attack_X=[]
    attack_Y=[]
    
    model.eval()
    with torch.no_grad():   
        # no_grad：表示这个代码块里的操作不用记录梯度，也不参与反向传播。用于预测阶段，不更新模型参数。减少开销
        # 在手搓预测的时候，model.eval()和torch.no_grad()一般要一起用
        for (inputs,_) in iterator:
            inputs=inputs.to(device)
            outputs=model(inputs)   # 目标模型or影子模型的预测结果
            
            posteriors=F.softmax(outputs,dim=1) # 转换为后验概率（即每个类别的概率）
            
            if top_k:   # 只保留模型输出中前三个最大概率的类别
                topk_probs,_=torch.topk(posteriors,3,dim=1)
                attack_X.append(topk_probs.cpu())
            else:   # 保留完整的概率向量（一般攻击模型更喜欢用完整概率）
                attack_X.append(posteriors.cpu())
                
            if test_dataset:    # test_dataset=True 表示当前 batch 来自测试集（非成员样本）→ 标签 = 0
                attack_Y.append(torch.zeros(posteriors.size(0),dtype=torch.long))
            else:   # 反之说明来自训练集，为1
                attack_Y.append(torch.ones(posteriors.size(0),dtype=torch.long))
                
        return attack_X,attack_Y
            

# 每个 epoch 内执行一次完整训练循环
def train_per_epoch(model,iterator,criterion,optimizer,device,bce_loss=False):
    '''
    典型的 PyTorch 训练流程：
    1 前向传播得到预测；
    2 计算损失；
    3 清空梯度；
    4 反向传播；
    5 更新参数。
    '''
    epoch_loss=0
    epoch_acc=0
    correct=0
    total=0
    
    model.train()
    for _,(features,target) in enumerate(iterator):
        features=features.to(device)
        target=target.to(device)
        
        outputs=model(features)
        
        if bce_loss:    # 处理 二分类任务（Binary Classification）的情况
            # criterion就是损失函数的统称，这个情况下是BCEWithLogitsLoss
            # BCEWithLogitsLoss 要求：[batch, 1]，标签：[batch,1]，当前的标签是[batch]，需要进一步转换才行
            loss=criterion(outputs,target.unsqueeze(1)) # unsqueeze(1) 是为了让标签的 shape 匹配模型输出
            # 举个例子：
            '''
            outputs = torch.tensor([[0.7], [0.2], [0.9]])  # shape = [3, 1]
            target = torch.tensor([1, 0, 1])               # shape = [3]
            target.unsqueeze(1)--> 
                torch.tensor([[1], [0], [1]])              # shape = [3, 1]
            '''
        else:   
            # 处理多分类任务（Multi-class Classification）的情况 
            # 使用 CrossEntropyLoss 损失函数，要求：[batch, num_classes]，标签：[batch]
            loss=criterion(outputs,target)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        total+=target.size(0)
        correct+=(predicted==target).sum().item()
        
    epoch_acc=correct/total
    epoch_loss=epoch_loss/total
    
    return epoch_loss,epoch_acc
        
        
        
def val_per_epoch(model,iterator,criterion,device,bce_loss=False):
    epoch_loss=0
    epoch_acc=0
    correct=0
    total=0
    
    model.eval()
    with torch.no_grad():
        for _,(features,target) in enumerate(iterator):
            features=features.to(device)
            target=target.to(device)
            
            outputs=model(features)
            
            # 计算损失
            if bce_loss:
                loss=criterion(outputs,target.unsqueeze(1))
            else:
                loss=criterion(outputs,target)
                
            epoch_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
            
    epoch_acc=correct/total    
    epoch_loss=epoch_loss/total
    
    return epoch_loss,epoch_acc

# 训练攻击模型
def train_attack_model(model,dataset,criterion,optimizer,lr_scheduler,
                       device,model_path='./Lab10-1/Membership-Inference/best_models',epochs=10,
                       b_size=20,num_workers=1,verbose=False,earlystopping=True):
    n_validation=1000
    best_valacc=0
    stop_count=0
    patience=10
    
    path=os.path.join(model_path,'./best_attack_model.ckpt')
    
    train_loss_hist=[]
    valid_loss_hist=[]
    val_acc_hist=[]
    
    train_X,train_Y=dataset
    # dataset 实际上就是 (attack_X, attack_Y) 这个二元组，所以可以这样拆
    t_X=torch.cat(train_X)  # 隐藏了dim=0，把多个 batch 按 dim=0 拼接起来
    # torch.cat 的意思是 concatenate（拼接），它把多个 tensor 按指定维度拼接成一个大的 tensor
    '''
    举个例子：
    # 每个 batch 有 2 个样本，每个样本有 3 个类别的预测概率
    train_X = [
        torch.tensor([[0.1, 0.7, 0.2],
                    [0.3, 0.4, 0.3]]),   # batch 1, shape [2,3]
        torch.tensor([[0.2, 0.2, 0.6],
                    [0.5, 0.1, 0.4]])    # batch 2, shape [2,3]
    ]
    train_Y = [
        torch.tensor([1, 0]),   # batch 1
        torch.tensor([1, 1])    # batch 2
    ]
    
    经过cat后：
    t_X = tensor([[0.1, 0.7, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.2, 0.6],
                [0.5, 0.1, 0.4]])
    t_X.shape  # [4, 3]，总共 4 个样本，每个样本 3 个类别预测

    t_Y = tensor([1, 0, 1, 1])
    t_Y.shape  # [4]，总共 4 个标签
    '''
    t_Y=torch.cat(train_Y)
    
    attackdataset=TensorDataset(t_X,t_Y)
    
    n_train_samples=len(attackdataset)-n_validation
    train_data,val_data=torch.utils.data.random_split(attackdataset,[n_train_samples,n_validation])
    train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=b_size,
                                             shuffle=True,      # 每个 epoch 开始时，随机打乱数据顺序
                                             num_workers=num_workers)   # 多线程加载数据，加速训练。=4表示4个线程
    
    val_loader=torch.utils.data.DataLoader(dataset=val_data,
                                           batch_size=b_size,
                                           shuffle=False,     # 验证集和测试集不需要打乱顺序
                                           num_workers=num_workers)
    
    for i in range(epochs):
        train_loss,train_acc=train_per_epoch(model,train_loader,criterion,optimizer,device)
        valid_loss,valid_acc=val_per_epoch(model,val_loader,criterion,device)
        
        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)
        
        lr_scheduler.step() # 表示在训练过程中 动态调整学习率
        
        if earlystopping:
        # 当验证准确率提升时保存模型并重置 stop_count，否则增加；当 stop_count >= patience 时提前退出循环
            if best_valacc<=valid_acc:
                best_valacc=valid_acc
                best_model=copy.deepcopy(model.state_dict())
                torch.save(best_model,path)
                stop_count=0
            else:
                stop_count+=1
                if stop_count>=patience:
                    break
        else:
            best_valacc=valid_acc
            best_model=copy.deepcopy(model.state_dict())    
            # 保存参数，如可学习参数（weights、biases）和某些层的缓冲区（buffers）。
            torch.save(best_model,path)    
            
    return best_valacc


def train_model(model,train_loader,val_loader,test_loader,loss,optimizer,scheduler,device,model_path,
                verbose=False,num_epochs=50,top_k=False,earlystopping=False,is_target=False):
    best_valacc=0
    patience=5
    stop_count=0
    train_loss_hist=[]
    valid_loss_hist=[]
    val_acc_hist=[]
    
    target_path=os.path.join(model_path,'best_target_model.ckpt')
    shadow_path=os.path.join(model_path,'best_shadow_model.ckpt')
    
    for epoch in range(num_epochs):
        train_loss,train_acc=train_per_epoch(model,train_loader,loss,optimizer,device)
        valid_loss,valid_acc=val_per_epoch(model,val_loader,loss,device)
        
        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)
        
        scheduler.step()
        
        if earlystopping:
            if best_valacc<=valid_acc:
                best_valacc=valid_acc
                best_model=copy.deepcopy(model.state_dict())
                if is_target:
                    torch.save(best_model,target_path)
                else:
                    torch.save(best_model,shadow_path)
                stop_count=0
            else:
                stop_count+=1
                if stop_count>=patience:
                    break
        else:
            best_valacc=valid_acc
            best_model=copy.deepcopy(model.state_dict())
            if is_target:
                torch.save(best_model,target_path)
            else:
                torch.save(best_model,shadow_path)
            
    if is_target:
        model.load_state_dict(torch.load(target_path))
    else:
        model.load_state_dict(torch.load(shadow_path))
        
    attack_X,attack_Y=prepare_attack_data(model,train_loader,device,top_k)
    
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for inputs,labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            test_outputs=model(inputs)
            _,predicted=torch.max(test_outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            
            probs_test=F.softmax(test_outputs,dim=1)
            if top_k:
                topk_t_probs,_=torch.topk(probs_test,3,dim=1)
                attack_X.append(topk_t_probs.cpu())
            else:
                attack_X.append(probs_test.cpu())
            attack_Y.append(torch.zeros(probs_test.size(0),dtype=torch.long))
            
    return attack_X,attack_Y