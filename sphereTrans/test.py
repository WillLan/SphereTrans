from __future__ import absolute_import, division, print_function
import os
import argparse
from trainer import Trainer

parser = argparse.ArgumentParser(description="360 Degree Panorama Segmentation Training")

# dataset
parser.add_argument("--data_path", default="/media/lby/lby_8t/dataset/s2d3d_resize", type=str, help="path to the dataset stanford2d3d.")
parser.add_argument("--dataset", default="stanford2d3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to train on.")
parser.add_argument("--train_fold", default='1_train')
parser.add_argument("--test_fold", default='1_valid')
parser.add_argument("--depth", default=False)

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--device", default='cuda:0', help="availacd ble gpus")

# model settings
parser.add_argument("--model_name", type=str, default="test", help="folder to save the model in") # v4: sphereVit+attention v5:only attention

parser.add_argument("--hw", type=int, default=(512,1024), help="input image height and width")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_epochs", type=int, default=120, help="number of epochs")
parser.add_argument("--imagenet_pretrained", action="store_true", help="if set, use imagenet pretrained parameters")

# loading and logging settings
parser.add_argument("--load_weights_dir", default = '/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/logs/Resnet101+sphereTrans_decoder(stage=4, cosLR)/models/weights_28', type=str, help="folder of model to load")
parser.add_argument("--log_dir", type=str, default="./logs", help="log directory")
parser.add_argument("--log_frequency", type=int, default=1, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=25, help="number of epochs between each save")
parser.add_argument("--vis_dir", default='./result', help="path to save visualization results")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", default=True, help="if set, do not use color augmentation")
parser.add_argument("--flip", default=True,
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--rotate", default=True,
                    help="if set, do not use yaw rotation augmentation")

# ablation settings
parser.add_argument("--net", type=str, default="sphereTrans", choices=["UniFuse", "Equi"], help="model to use")

args = parser.parse_args()


def main():
    trainer = Trainer(args)
    iou = trainer.validate()


if __name__ == "__main__":
    main()




