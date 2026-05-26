# 测试模型
import paddle
from model import RNN
from RumorDataset import test_loader
from data_caculate import ids_to_str,load_vocab,data_root_path
import numpy as np
import os

vocab=load_vocab(os.path.join(data_root_path,'dict.txt'))
model_state_dict = paddle.load('model_final.pdparams')
model = RNN()
model.set_state_dict(model_state_dict)
model.eval()
label_map = {0: '谣言', 1: '非谣言'}

samples=[]
predictions =[]
accuracies =[]
losses = []

for batch_id, data in enumerate(test_loader) :
    sent = data[0]
    label =data[1]
    logits = model(sent)
    for idx, probs in enumerate(logits) :
        # 映射分类label
        label_idx = np.argmax(probs)
        labels = label_map[label_idx]
        label_idx = np.argmax(probs)
        predictions.append(labels)
        samples.append(sent[idx].numpy())
    
    acc = paddle.metric.accuracy(logits, label.unsqueeze(1))
    loss = paddle.nn.functional.cross_entropy(logits,label)

    accuracies.append(acc.numpy())
    losses.append(loss.numpy())
avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))
print("数据：{}\n\n是否谣言：{}".format(ids_to_str(samples[0],vocab), predictions[0]))