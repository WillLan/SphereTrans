from __future__ import absolute_import, division, print_function
import os
import argparse

from trainer_depth import Trainer
#from test import Tester

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# data settings
parser.add_argument("--dataset", default="matterport3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset to train on.")
parser.add_argument("--data_path", default="/media/lby/lby_8t/dataset/s2d3d_resize", type=str, help="path to the dataset.")
parser.add_argument("--hw", type=int, default=(512, 1024), help="input image height and width")
parser.add_argument("--train_fold", default='1_train')
parser.add_argument("--test_fold", default='1_valid')

parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=str, default='cuda:1', help="available gpus")

# model settings
parser.add_argument("--model_name", type=str, default="new_mt3d", help="folder to save the model in")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--start_epoch", type=int, default=0, help="start_epoch")

# loading and logging settings
parser.add_argument("--load_weights_dir", default='', type=str, help="folder of model to load")#, default='./tmp_abl_offset/panodepth/models/weights_49'
parser.add_argument("--log_dir", type=str, default='./3d60_logs', help="log directory")
parser.add_argument("--log_frequency", type=int, default=1, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=100, help="number of epochs between each save")

# data augmentation settings
parser.add_argument("--flip", default=True,
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--rotate", default=True,
                    help="if set, do not use yaw rotation augmentation")


args = parser.parse_args()


def main():
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
