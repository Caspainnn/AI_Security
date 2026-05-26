import paddle
import os
import numpy as np
import matplotlib.pyplot as plt
from model import RNN
from RumorDataset import train_loader, test_loader
from pre_data import load_vocab
from config import model_file,data_root_path

vocab=load_vocab(os.path.join(data_root_path,'dict.txt'))
    

def draw_process(title,color,iters,data,label) :
    plt.title(title,fontsize=24)
    plt.xlabel("iter", fontsize= 20)
    plt.ylabel(label,fontsize=20)
    plt.plot(iters, data,color =color,label= label)
    plt.legend()
    plt.grid()
    plt.show()
    
def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate= 0.002,parameters = model.parameters())
    
    steps =0
    Iters,total_loss, total_acc =[],[],[]
    #训练3轮
    for epoch in range(3):
        for batch_id,data in enumerate(train_loader):
            steps +=1
            sent = data[0]
            label = data[1]
            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label.unsqueeze(1))    
            # paddle.metric.accuracy() 内部期望输入格式是： label.shape = [batch_size, 1]
            # 所以需要用 unsqueeze(1) 增加一个维度
            if batch_id % 50 == 0:
                Iters.append(steps)
                total_loss.append(loss.numpy())
                total_acc.append(acc.numpy())
                print("epoch: {}, batch_id: {}, loss: {}, acc: {}"\
                    .format(epoch, batch_id, loss.numpy(), acc.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()
        # 每个epoch后对模型进行评估
        model.eval()
        accuracies =[]
        losses =[]
        
        for batch_id, data in enumerate(test_loader):
            sent= data[0]
            label=data[1]
            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits,label.unsqueeze(1))
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
            
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))
        model.train()
    paddle.save(model.state_dict(), model_file)
    draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
    draw_process("trainning acc","blue",Iters,total_acc,"trainning acc")
    
if __name__ == '__main__':
    model=RNN()
    train(model)
