import common
import torch


def text2vec(text):
    # 使用预计算的索引映射以提高性能
    index_map={char:idx for idx,char in enumerate(common.captcha_array)}
    vectors=torch.zeros((common.captcha_size,len(common.captcha_array)))
    for i,char in enumerate(text):
        if char in index_map:
            vectors[i,index_map[char]]=1
        else:
            raise ValueError(f"character '{char}' not found in captcha array")
    return vectors

def vectotext(vec):
    # 使用列表推导和join方法来构建最终字符串
    indices=torch.argmax(vec,dim=1)
    return ''.join(common.captcha_array[idx] for idx in indices)

if __name__=="__main__":
    vec=text2vec("1112")
    print(vec)
    print(vec.shape)
    print(vectotext(vec))