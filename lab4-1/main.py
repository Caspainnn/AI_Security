import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



# 定义被攻击的白盒神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()   # super是父类nn.Module的构造函数，是一种常见的初始化方式
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)  # 全连接层
        self.fc2=nn.Linear(50,10)
        
    # 定义向前传播函数
    def forward(self,x):    # 向前传播函数是用来定义网络中的各层是如何连接的
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x=x.view(-1,320)    # 实现数据x从二维通过flatten转一维，-1表示自动计算得到的第一个维度大小
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)   # training=self.training表示是否在训练模式下进行dropout操作
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)       # dim=1

        
    
# FGSM
def fgsm_attack(image,epsilon,data_grad):
    sign_data_grad=data_grad.sign()
    perurbed_image=image+epsilon*sign_data_grad
    perturbed_image=torch.clamp(perurbed_image,0,1)
    return perturbed_image


# PGD攻击
def pgd_attack(model,image,label,epsilon,iters=4):
    # 原图像
    ori_image=image.data
    # 使用梯度信息生成对抗样本
    alpha=epsilon/iters
    
    for i in range(iters):
        image.requires_grad=True
        output=model(image)
        loss=F.nll_loss(output,label)
        model.zero_grad()
        loss.backward()
        # 图像+梯度得到对抗样本
        data_grad=image.grad
        adv_images=image+alpha*data_grad.sign()
        # 限制扰动范围
        eta=torch.clamp(adv_images-ori_image,min=-epsilon,max=epsilon)
        # 进行下一轮对抗样本生成 破坏之前的计算图
        image=torch.clamp(ori_image+eta,min=0,max=1).detach()
    return image

    
def test(model,device,test_loader,epsilon,attack_type="fgsm"):
    correct=0
    adv_examples=[]
    
    for data,target in test_loader:
        data,target=data.to(device),target.to(device)
        data.requires_grad=True
        output=model(data)
        init_pred=output.max(1,keepdim=True)[1]
        
        if init_pred.item()!=target.item():
            continue
        
        loss=F.nll_loss(output,target)
        model.zero_grad()
        loss.backward()
        data_grad=data.grad
        
        if attack_type=="fgsm":
            perturbed_data=fgsm_attack(data,epsilon,data_grad)
        elif attack_type=="pgd":
            perturbed_data=pgd_attack(model,data,target,epsilon)
        else:
            raise ValueError("attack_type must be 'fgsm' or 'pgd'")
        
        
        output=model(perturbed_data)
        final_pred=output.max(1,keepdim=True)[1]
        if final_pred.item()==target.item():
            correct+=1
            if (epsilon==0) and (len(adv_examples)<5):
                adv_ex=perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(),final_pred.item(),adv_ex))
        else:
            if len(adv_examples)<5:
                adv_ex=perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(),final_pred.item(),adv_ex))
                
    final_acc=correct/float(len(test_loader))
    print("Epsilon:{}\tTest Accuracy={}/{}={}".format(epsilon,correct,len(test_loader),final_acc))
    return final_acc,adv_examples


def show_picture(exss,accs):
    # 绘制折线图
    plt.figure(figsize=(5,5))
    plt.plot(exss,accs,marker="o")
    plt.yticks(np.arange(0,1.1,step=0.1))
    plt.xticks(np.arange(0,0.31,step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    filename='./lab4-1/FGSM.png'
    plt.savefig(filename)
    plt.show()


def show_picture_FGSMvsPGD(exss,accs_FGSM,accs_PGD):
    # 绘制折线图
    plt.figure(figsize=(5,5))
    plt.plot(exss,accs_FGSM,marker="o",label="FGSM")
    plt.plot(exss,accs_PGD,marker="o",label="PGD")
    plt.yticks(np.arange(0,1.1,step=0.1))
    plt.xticks(np.arange(0,0.31,step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    filename='./lab4-1/FGSMvsPGD.png'
    plt.savefig(filename)
    plt.show()


if __name__=="__main__":
    # 模型和数据库导入
    test_loader=torch.utils.data.DataLoader(datasets.MNIST(root="./lab4-1/data",train=False,download=True,
                                            transform=transforms.Compose([transforms.ToTensor()])),
                                            batch_size=1,shuffle=True)
    print("CUDA Available: ",torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Net().to(device)
    pretrained_model="./lab4-1/lenet_mnist_model.pth"
    model.load_state_dict(torch.load(pretrained_model,map_location=device))
    model.eval()
    
    
    # 遍历数据集
    epsilons=[0,0.05,0.1,0.15,0.2,0.25,0.3] 
    fgsm_accs=[]
    pgd_accs=[]
    fgsm_exss=[]
    pgd_exss=[]
    print("-----------FGSM-----------")
    for eps in epsilons:
        facc,fexs=test(model,device,test_loader,eps,"fgsm")
        fgsm_accs.append(facc)
        fgsm_exss.append(fexs)
        
    print("-----------PGD-----------")
    for eps in epsilons:
        pacc,pexs=test(model,device,test_loader,eps,"pgd")
        pgd_accs.append(pacc)
        pgd_exss.append(pexs)
    
    show_picture_FGSMvsPGD(epsilons,fgsm_accs,pgd_accs)
        
    
        