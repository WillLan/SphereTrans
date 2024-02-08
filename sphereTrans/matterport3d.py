from __future__ import print_function
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip())
    return rgb_depth_list


class Matterport3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, list_file, depth=True, hw=(512, 1024), mask_black=True, flip=True, rotate=True, color_augmentation=True, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black

        self.rgb_depth_list = read_list(list_file)

        print(len(self.rgb_depth_list))  # 训练集和测试集大小

        self.is_training = is_training
        self.h = hw[0]
        self.w = hw[1]
        self.flip = flip
        self.rotate = rotate
        self.color_augmentation = color_augmentation
        self.color_aug = transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                                                hue=(-0.1, 0.1))

        self.max_depth_meters = 10.0
        self.to_tensor = transforms.ToTensor()  # [0,255]->[0,1], H*W*C->C*H*W
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.rgb_depth_list)

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
        rgb_name = self.rgb_depth_list[idx]
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # print(rgb_name)

        depth_name = self.rgb_depth_list[idx].replace('emission', 'depth').replace('.png', '.exr')
        gt_depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth.astype(np.float32)
        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1
        # print(gt_depth)

        H = rgb.shape[0]
        W = rgb.shape[1]
        if (H, W) != self.hw:
            rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)  # 双线性插值
            gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.is_training and self.rotate:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        if self.is_training and self.flip and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))

        mask = torch.ones([512, 1024])
        mask[0:int(512 * 0.15), :] = 0
        mask[512 - int(512 * 0.15):512, :] = 0


        rgb = self.to_tensor(rgb.copy())  # H*W*C->C*H*W
        # if self.mask_black:
        #     sem[rgb.sum(0) == 0] = -1

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(rgb)
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))  # 扩展出一个c维度
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        inputs["val_mask"] = inputs["val_mask"] * mask

        inputs["name"] = os.path.split(self.rgb_depth_list[idx])[1].ljust(200)

        return inputs


