import paddle
import io
import sys
import numpy as np
import os
from pre_data import load_vocab
import config as cfg

data_root_path=cfg.data_root_path
vocab=load_vocab(os.path.join(data_root_path,'dict.txt'))

class RumorDataset(paddle.io.Dataset):
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.all_data=[]
        
        with io.open(self.data_dir,'r',encoding='utf-8') as f:
            for line in f:
                cols=line.strip().split('\t')
                if len(cols)!=2:
                    if len(cols)==1:    # 跳过空行
                        continue
                    else:   # 格式错误行
                        sys.stderr.write("[NOTICE] Error Format Line!\n"+line)
                        continue
                label=int(cols[1])
                wids=cols[0].split(',')
                if len(wids)>=150:
                    wids=np.array(wids[:150]).astype('int64')
                else:
                    wids=np.concatenate([wids,[vocab["<pad>"]]*(150-len(wids))]).astype('int64')
                    label=np.array(label).astype('int64')
                self.all_data.append((wids,label))
                
    def __getitem__(self,index):
        data,label=self.all_data[index]
        return data,label
    
    def __len__(self):
        return len(self.all_data)
    
    
batch_size=32
train_dataset =RumorDataset(os.path.join(data_root_path, 'train_list.txt'))
test_dataset = RumorDataset(os.path.join(data_root_path, 'eval_list.txt'))
train_loader = paddle. io. DataLoader(train_dataset, places = paddle. CPUPlace(), return_list=True,
                                      batch_size=batch_size, shuffle=True)
test_loader = paddle. io. DataLoader(test_dataset, places = paddle. CPUPlace(), return_list=True,
                                      batch_size=batch_size, shuffle=True)

# 输出训练数据
print('============= train dataset =============')
for data, label in train_dataset:
    print(data)
    print(np.array(data).shape)
    print(label)
    break

# 输出验证数据
print('========= test dataset =============')
for data, label in test_dataset:
    print(data)
    print(np.array(data).shape)
    print(label)
    break



