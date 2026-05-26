import os
import config as cfg
import random
import sys


train_data,test_data,all_data_file=cfg.train_data,cfg.test_data,cfg.all_data_file
dict_file,data_list_path=cfg.dict_file,cfg.data_list_path


# 将训练集和数据集合并为一个文件
def create_all_data():
    all_data_list=[]
    with open(train_data,'r',encoding='utf-8') as f_train,\
    open(test_data,'r',encoding='utf-8') as f_test:        
        for line in f_train:
            all_data_list.append(line)
        for line in f_test:
            all_data_list.append(line)
            
    random.shuffle(all_data_list)
    
    # 先清空 all_data.txt 文件
    with open(all_data_file,'w',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
        
    # 再写入 all_data.txt 文件    
    with open(all_data_file,'w',encoding='utf-8') as f_all:
        for line in all_data_list:
            f_all.write(line)
            
# 读取 all_data.txt 文件，汇总成谣言、非谣言、不确定 三个列表
def read_all_data():
    
    # 谣言-标签0，非谣言-标签1
    rumor_label="0"
    non_rumor_label="1"
    uncertain_label="2"
    
    rumor_num=0
    non_rumor_num=0
    uncertain_num=0
    
    with open(all_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            label, _ = line.strip().split('\t', 1)
            if label == rumor_label:    # 谣言
                rumor_num+=1
            elif label == non_rumor_label:    # 非谣言
                non_rumor_num+=1
            elif label == uncertain_label:    # 不确定
                uncertain_num+=1
                
    print("[INFO] 读取 all_data.txt 完成！")
    print(f"[INFO] 样本总数: {rumor_num + non_rumor_num + uncertain_num}")
    print(f"[INFO] 谣言样本数: {rumor_num}")
    print(f"[INFO] 非谣言样本数: {non_rumor_num}")
    print(f"[INFO] 不确定样本数: {uncertain_num}")

# 基于 all_data.txt 创建字符级词典
def create_dict():
    
    with open(dict_file, 'w', encoding='utf-8') as f:
        f.seek(0)   # 将文件指针移动到文件的起始位置（0 字节处）
        f.truncate()    # 删除从当前文件指针位置到文件末尾的所有内容
        
    dict_set=set()
    
    # 读取全部数据，构建词典集合
    with open(all_data_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            content=line.split('\t')[-1].replace('\n','')
            for s in content:
                dict_set.add(s)
                
    # 把集合转换成字典，一个字对应一个数字
    dict_list=[]
    i=0
    for s in dict_set:
        dict_list.append([s,i])
        i+=1

    # 添加未知字符和填充字符
    dict_txt=dict(dict_list)
    end_dict={"<unk>":i}
    dict_txt.update(end_dict)
    end_dict={"<pad>":i+1}
    dict_txt.update(end_dict)
    
    # 把这些字典保存在本地
    with open(dict_file,'w',encoding='utf-8') as f:
        f.write(str(dict_txt))
        
    print(f"[INFO] 词典生成完成，已保存到 {dict_file}！")

            
# 创建序列化表示的数据
def create_data_list():
    # 先清空
    with open(data_list_path+'train_list.txt','w',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
    with open(data_list_path+'eval_list.txt','w',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
        
    # 读取全部数据
    with open(all_data_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        
    with open(dict_file,'r',encoding='utf-8') as f:
        dict_txt=eval(f.readlines()[0])

    i=0
    maxlen=0
    with open(data_list_path+'eval_list.txt','a',encoding='utf-8') as f_eval,\
        open(data_list_path+'train_list.txt','a',encoding='utf-8') as f_train:
        for line in lines:
            words=line.split('\t')[-1].replace('\n','')
            maxlen=max(maxlen,len(words))
            label=line.split('\t')[0]
            labs=""
            # 每8个抽取一个数据用于验证
            if i%8==0:
                for s in words:
                    lab=str(dict_txt[s])
                    labs=labs+lab+','
                labs=labs[:-1]
                labs=labs+"\t"+label+'\n'
                f_eval.write(labs)
            else:
                for s in words:
                    lab=str(dict_txt[s])
                    labs=labs+lab+','
                labs=labs[:-1]
                labs=labs+"\t"+label+'\n'
                f_train.write(labs)
            i+=1
    print("数据序列化完成！")
    print("样本最大长度："+str(maxlen))
    
    
# 读取字典文件
def load_vocab(file_path):
    fr=open(file_path,'r',encoding='utf-8')
    vocab=eval(fr.read())   # 读取的str转换为字典
    fr.close()
    return vocab
    
    
# 把数字id序列转换为字符串序列
def ids_to_str(ids,vocab):
    words=[]
    for k in ids:
        w=list(vocab.keys())[list(vocab.values()).index(int(k))]
        words.append(w if isinstance(w,str) else w.decode('ASCII'))
    return ''.join(words)


# 打印前k条数据
def print_data(k,vocab):  
    with open(os.path.join(data_list_path,'train_list.txt'),'r',encoding='utf-8') as fin:
        i=0
        for line in fin:
            i+=1
            cols=line.strip().split('\t')
            if len(cols)!=2:
                sys.stderr.write("[NOTICE] Error Format Line!")
                continue
            label=int(cols[1])
            wids=cols[0].split(',')
            print(str(i)+":")
            print("sentence list id is:",wids)
            print("sentence is:",ids_to_str(wids,vocab))
            print("sentence label id is:",label)
            print("=========================")
            
            if i>=k:
                break


if __name__=="__main__":
    create_all_data()
    read_all_data()
    create_dict()
    create_data_list()
    vocab=load_vocab(dict_file)
    print_data(1,vocab)