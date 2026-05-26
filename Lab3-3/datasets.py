import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from text2vec import text2vec
import common


# 数据预处理
class mydatasets(Dataset):
    def __init__(self,root_dir):
        super(mydatasets,self).__init__()
        self.list_image_path=[os.path.join(root_dir,image_name) for image_name in os.listdir(root_dir)]
        self.transforms=transforms.Compose([
            transforms.Resize((60,160)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
        
    def __getitem__(self, index):
        image_path=self.list_image_path[index]
        
        img_ = Image.open(image_path)
        image_name=image_path.split("\\")[-1]
        img_tesor=self.transforms(img_)
        
        img_label=image_name.split("_")[0]
        img_label=img_label[-4:]   # 获取后四个字符
        img_label=text2vec(img_label)
        img_label=img_label.view(1,-1)[0]
        return img_tesor,img_label
    
    def __len__(self):
        return self.list_image_path.__len__()


if __name__=="__main__":
    d=mydatasets("./Lab3-3/datasets/train/")
    img,label=d[0]
    # 创建 TensorBoard 的日志写入器SummaryWriter，用于记录和可视化实验数据，日志会保存在"logs"目录下
    writer=SummaryWriter("./Lab3-3/logs")    
    writer.add_image("img",img,1)
    """
        将获取的图像数据img写入 TensorBoard 日志，参数含义：
            "img"：图像在 TensorBoard 中的标签名称
            img：要显示的图像数据
            1：全局步数（用于区分不同步骤的图像）
    """
    print(img.shape)
    writer.close # 关闭日志写入器，释放资源。