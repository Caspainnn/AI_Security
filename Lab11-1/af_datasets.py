from torch.utils.data import Dataset
import torch
import PIL

# 设置随机种子和设备
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UTKFace 数据集
class UTKFace(Dataset):
    def __init__(self, samples, label, transform=None):
        self.samples = samples
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        if self.transform:
            image = self.transform(image)
        if self.label == 'gender':
            label = int(self.samples.iloc[idx, 1])
            sample = {'image': image, 'gender': label}
        elif self.label == 'race':
            label = int(self.samples.iloc[idx, 2])
            sample = {'image': image, 'race': label}
        return sample

# AttackData 数据集
class AttackData(Dataset):
    def __init__(self, samples, target_model, transform=None):
        self.samples = samples
        self.target_model = target_model
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        if self.transform:
            image = self.transform(image)

        # ✅ 修复点：把图片放到和模型一样的设备上
        image = image.unsqueeze(0).to(next(self.target_model.parameters()).device)
        
        _, z = self.target_model(image)
        label = int(self.samples.iloc[idx, 2])
        sample = {'z': z.detach().squeeze(0).cpu(), 'race': label}
        return sample