import cv2
import numpy as np
import random
import glob
import torch
from torch.utils import data
from torchvision import transforms
import os
from imageio.v2 import imread

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_valid': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_valid': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_valid': ['area_1', 'area_3', 'area_6'],
}

class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""
    NUM_CLASSES = 13
    ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']

    def __init__(self, root, fold, depth=True, hw=(512, 1024), mask_black=True, flip=True, rotate=True, color_augmentation=True, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []
        for dname in __FOLD__[fold]:
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))  # glob.glob 函数来匹配指定目录下所有以 .png 扩展名结尾的文件
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        print(len(self.rgb_paths), len(self.sem_paths), len(self.dep_paths))    # 训练集和测试集大小
        assert len(self.rgb_paths)
        assert len(self.rgb_paths) == len(self.sem_paths)
        assert len(self.rgb_paths) == len(self.dep_paths)

        self.is_training = is_training
        self.h = hw[0]
        self.w = hw[1]
        self.flip = flip
        self.rotate = rotate
        self.color_augmentation = color_augmentation
        self.color_aug = transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))


        self.max_depth_meters = 10.0
        self.to_tensor = transforms.ToTensor()    # [0,255]->[0,1], H*W*C->C*H*W
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        inputs = {}
        """
        使用函数cv2.imread(filepath,flags)读入一幅图片,函数默认读取的是一幅彩色图片
        filepath：要读入图片的完整路径
        flags：读入图片的标志 ，{cv2.IMREAD_COLOR，cv2.IMREAD_GRAYSCALE，cv2.IMREAD_UNCHANGED}
        cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
        cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
        cv2.IMREAD_UNCHANGED：读入完整图片，包括alpha通道，可用-1作为实参替代
        cv2.IMREAD_UNCHANGED：读入完整图片，包括alpha通道，可用-1作为实参替代
        PS：alpha通道，又称A通道，是一个8位的灰度通道，该通道用256级灰度来记录图像中的透明度复信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明
        """
        rgb_name = self.rgb_paths[idx]
        #print(rgb_name)
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_name = self.dep_paths[idx]
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = gt_depth.astype(np.float32)/512
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1
        # print(gt_depth.shape)

        sem_name = self.sem_paths[idx]
        sem = imread(sem_name)
        # print(sem.shape)
        H = rgb.shape[0]
        W = rgb.shape[1]
        if (H, W) != self.hw:
            rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)  # 双线性插值
            gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
            sem = cv2.resize(sem, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # 数据增强
        if self.is_training and self.rotate:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
            sem = np.roll(sem, roll_idx, 1)
        if self.is_training and self.flip and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
            sem = cv2.flip(sem,1)
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))

        mask = torch.ones([self.h, self.w])
        mask[0:int(self.h * 0.15), :] = 0
        mask[self.h - int(self.h * 0.15):self.h, :] = 0


        rgb = self.to_tensor(rgb.copy())     # H*W*C->C*H*W
        sem = torch.LongTensor(sem) - 1    # 这里标签值不能用transforms.totensor, 因为会被归一化

        if self.mask_black:
            sem[rgb.sum(0) == 0] = -1

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(rgb)
        inputs['sem'] = sem
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))   # 扩展出一个c维度
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        inputs["val_mask"] = inputs["val_mask"] * mask

        inputs["name"] = os.path.split(self.rgb_paths[idx])[1].ljust(200)

        return inputs