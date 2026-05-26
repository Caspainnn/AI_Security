import os
import random
import json
import sys

lab_path="./Lab6-4-Blue/"

src_path="data/data20519/Rumor_Dataset.zip"
# target_path="/home/aistudio/data/Chineses_Rumor_Dataset-master"
target_path=lab_path+"data/Chinese_Rumor_Dataset-master"
# data_root_path="/home/aistudio/data/"
data_root_path=lab_path+"data/"
file_path=os.path.join(data_root_path,'train_list.txt')
# 谣言数据文件路径
rumor_class_dirs=os.listdir(target_path+"/CED_Dataset/rumor-repost/")
# 非谣言
non_rumor_class_dirs=os.listdir(target_path+"/CED_Dataset/non-rumor-repost/")
# 原始微博数据文件路径
original_microblog=target_path+"/CED_Dataset/original-microblog/"
data_path=os.path.join(data_root_path,'all_data.txt')
dict_path=os.path.join(data_root_path,'dict.txt')

data_list_path=lab_path+"data/"
all_data_path=data_list_path+"all_data.txt"



vocab={}



def read_data():
    
    # 解压数据集文件
    # if(not os.path.isdir(target_path)):
    #     z=zipfile.ZipFile(src_path,'r')
    #     z.extractall(path=target_path)
    # z.close()

    # # 谣言数据文件路径
    # rumor_class_dirs=os.listdir(target_path+"/CED_Dataset/rumor-repost/")
    # # 非谣言
    # non_rumor_class_dirs=os.listdir(target_path+"/Chineses_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
    # # 原始微博数据文件路径
    # original_microblog=target_path+"/Chineses_Rumor_Dataset-master/CED_Dataset/original-microblog/"

    # 谣言-标签0，非谣言-标签1
    rumor_label="0"
    non_rumor_label="1"

    # 分别统计谣言数据与非谣言数据的总和
    rumor_num=0
    non_rumor_num=0

    all_rumor_list=[]
    all_non_rumor_list=[]

    # 解析谣言数据
    for rumor_class_dir in rumor_class_dirs:
        if rumor_class_dir != '._.DS_Store' and rumor_class_dir != '.DS_Store':
            with open(original_microblog+rumor_class_dir,'r',encoding='utf-8') as f:
                rumor_content=f.read()
            rumor_dict=json.loads(rumor_content)
            all_rumor_list.append(rumor_label+"\t"+rumor_dict["text"]+"\n")
            rumor_num+=1
            
    # 解析非谣言数据
    for non_rumor_class_dir in non_rumor_class_dirs:
        if non_rumor_class_dir != '._.DS_Store' and non_rumor_class_dir != '.DS_Store':
            with open(original_microblog+non_rumor_class_dir,'r',encoding='utf-8') as f:
                non_rumor_content=f.read()
            non_rumor_dict=json.loads(non_rumor_content)
            all_non_rumor_list.append(non_rumor_label+"\t"+non_rumor_dict["text"]+"\n")
            non_rumor_num+=1
            
    print("谣言数据总数："+str(rumor_num))
    print("非谣言数据总数："+str(non_rumor_num))


    # =========== 合并谣言数据与非谣言数据 =========== 
    all_data_list=all_rumor_list+all_non_rumor_list
    random.shuffle(all_data_list)
    
    # 在生成 all_data.txt 之前，首先将其清空
    with open(all_data_path,'w',encoding='utf-8') as f:
        f.seek(0)
        f.truncate()
        
    with open(all_data_path,'a',encoding='utf-8') as f:
        for data in all_data_list:
            f.write(data)   # 生成 all_data.txt
            
    print("合并完成，已生成 all_data.txt")
   
   
# 构建数据字典 dict.txt 
def create_dict(data_path,dict_path):
    with open(dict_path,'w') as f:
        f.seek(0)
        f.truncate()
        
    dict_set=set()
    
    # 读取全部数据
    with open(data_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        
    # 把数据生成一个元组
    for line in lines:
        content=line.split('\t')[-1].replace('\n','')
        for s in content:
            dict_set.add(s)
        
    # 把元组转换成字典，一个字对应一个数字
    dict_list=[]
    i=0
    for s in dict_set:
        dict_list.append([s,i])
        i+=1
    
    # 添加未知字符
    dict_txt=dict(dict_list)
    end_dict={"<unk>":i}
    dict_txt.update(end_dict)
    end_dict={"<pad>":i+1}
    dict_txt.update(end_dict)
    
    # 把这些字典保存在本地
    with open(dict_path,'w',encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成完成！")    


# 创建序列化表示的数据
def create_data_list(data_list_path):
    # 在生成数据之前，先把 eval_list.txt 和 train_list.txt 清空
    with open(os.path.join(data_list_path,'eval_list.txt'),'w') as f:
        f.seek(0)
        f.truncate()
    
    with open(os.path.join(data_list_path,'train_list.txt'),'w') as f:
        f.seek(0)
        f.truncate()
        
    with open(os.path.join(data_list_path,'dict.txt'),'r',encoding='utf-8') as f:
        dict_txt=eval(f.readlines()[0])
        
    with open(os.path.join(data_list_path,'all_data.txt'),'r',encoding='utf-8') as f:
        lines=f.readlines()
        
    i=0
    maxlen=0
    
    with open(os.path.join(data_list_path,'eval_list.txt'),'a',encoding='utf-8') as f_eval,\
        open(os.path.join(data_list_path,'train_list.txt'),'a',encoding='utf-8') as f_train:
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

# 对构建的数据集进行打印
def load_vocab(file_path):
    fr=open(file_path,'r',encoding='utf-8')
    vocab=eval(fr.read())   # 读取的str转换为字典
    fr.close()
    return vocab
    
    
# 把数字id序列转换为字符串序列
def ids_to_str(ids,vocab=vocab):
    words=[]
    for k in ids:
        w=list(vocab.keys())[list(vocab.values()).index(int(k))]
        words.append(w if isinstance(w,str) else w.decode('ASCII'))
    return ''.join(words)

def print_data(k):  # 打印前k条数据
    with open(file_path,'r',encoding='utf-8') as fin:
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
            print("sentence is:",ids_to_str(wids))
            print("sentence label id is:",label)
            print("=========================")
            
            if i>=k:
                break



if __name__ == '__main__':
    # ========== 数据准备 ==========
    read_data()
    # 创建数据字典
    create_dict(data_path,dict_path)
    # 创建数据列表
    create_data_list(data_root_path)
    # 加载数据字典
    vocab = load_vocab(dict_path)
    # 打印前两条数据
    print_data(2)   

    # ========== 定义数据集和加载器 ==========
    # vocab=load_vocab(os.path.join(data_root_path,'dict.txt'))
    
