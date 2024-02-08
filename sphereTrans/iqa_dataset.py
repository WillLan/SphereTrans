import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms

class IQADataset(Dataset):
    def __init__(self, data_dir, mos_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.mos = []
        with open(mos_dir) as f:                        # mos txt文件
            lines = f.readlines()
            for line in lines:
                idx_mos = line.strip().split(" ")       # 使用 line.strip().split(" ") 进行处理，会得到一个列表[idx, mos]
                self.mos.append(idx_mos)
        print(len(self.mos))

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, str(self.mos[index][0])+'.jpg')
        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img= self.transform(img)
        # print(img.shape)
        mos = self.mos[index][2]
        y_label = torch.FloatTensor(np.array(float(mos)))

        return img, y_label

    def __len__(self):
        return len(self.mos)

if __name__ == '__main__':
    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = IQADataset(data_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/fov',
                               mos_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/train_mos.txt',
                               transform=transformations)
    #print(len(train_dataset))
    test_dataset = IQADataset(data_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/fov',
                               mos_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/test_mos.txt',
                               transform=transformations)
