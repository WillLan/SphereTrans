from __future__ import print_function
import os
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
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


# Matterport3D/10_85cef4a4c3c244479c56e56d9a723ad21_color_0_Left_Down_180.0.png
def recover_filename(file_name):

    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]    # 获取旋转角度（180）
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)    # Matterport3D/10_85cef4a4c3c244479c56e56d9a723ad21_color_0_Left_Down_0.0.png


class ThreeD60(data.Dataset):
    """The 3D60 Dataset"""

    def __init__(self, root, list_file, depth=True, hw=(256, 512), mask_black=True, flip=True, rotate=True, color_augmentation=True, is_training=False):
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.root = root
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

        rgb_name, _ = recover_filename(self.rgb_depth_list[idx][0])
        rgb_name = os.path.join(self.root, rgb_name)
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_name, _ = recover_filename(self.rgb_depth_list[idx][1])
        depth_name = os.path.join(self.root, depth_name)
        gt_depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth.astype(np.float32)
        gt_depth[gt_depth>self.max_depth_meters] = self.max_depth_meters + 1

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

        mask = torch.ones([H, W])
        mask[0:int(H * 0.15), :] = 0
        mask[H - int(H * 0.15):H, :] = 0

        rgb = self.to_tensor(rgb.copy())  # H*W*C->C*H*W

        # if self.mask_black:
        #     sem[rgb.sum(0) == 0] = -1

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))  # 扩展出一个c维度
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        inputs["val_mask"] = inputs["val_mask"] * mask

        # inputs["name"] = os.path.split(self.rgb_depth_list[idx])[1].ljust(200)

        return inputs