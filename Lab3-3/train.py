import torch
import torch.nn as nn
from model import mymodel
import datasets as my_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train():
    train_data=my_datasets.mydatasets("./Lab3-3/datasets/train/")
    train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
    m=mymodel().to(device)
    
    loss_fn=nn.MultiLabelSoftMarginLoss().to(device)
    optimizer=torch.optim.Adam(m.parameters(),lr=0.001)
    
    epochs=10
    for epoch in range(epochs):
        for step,(imgs,targets) in tqdm(enumerate(train_dataloader)):
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=m(imgs)
            loss=loss_fn(outputs,targets)
            optimizer.zero_grad()   # 清除之前的梯度
            
            loss.backward()
            optimizer.step()
            
            if step%100==0:
                print(f"Epoch [{epoch+1}/{10}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
        torch.save(m,"model.pth")
        
if __name__=="__main__":
    train()