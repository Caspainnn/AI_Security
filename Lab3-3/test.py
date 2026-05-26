import torch
from model import mymodel
import datasets as my_datasets
from torch.utils.data import DataLoader
import common
from tqdm import tqdm
from text2vec import vectotext

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_pred():
    m=torch.load("model.pth",map_location="cpu",weights_only=False).to(device)
    m.eval()
    test_data=my_datasets.mydatasets("./Lab3-3/datasets/test/")
    test_dataloader=DataLoader(test_data,batch_size=1,shuffle=False)
    test_length=len(test_data)
    correct=0
    label_length=common.captcha_array.__len__()
    
    for imgs,labels in tqdm(test_dataloader,desc="Testing"):
        imgs=imgs.to(device)
        labels=labels.to(device).view(-1,label_length)
        labels_text=vectotext(labels)
        
        predict_outputs=m(imgs).view(-1,label_length)
        predict_labels=vectotext(predict_outputs)
        if predict_labels==labels_text:
            correct+=1
            print(f"✅  正确值：{labels_text}，预测值：{predict_labels}")
        else:
            print(f"❌  正确值：{labels_text}，预测值：{predict_labels}")
    accuracy=100*correct/test_length
    print(f"测试集准确率：{accuracy:.2f}%")
    
if __name__=="__main__":
    test_pred()
