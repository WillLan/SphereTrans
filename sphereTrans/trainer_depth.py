from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from metrics import Evaluator
from losses import BerhuLoss, Gradient_Net
from network.model_depth import SphereTrans
from stanford2d3d import Stanford2D3D
from matterport3d import Matterport3D
from td60 import ThreeD60
#from network.model_dpt import SphereTrans

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device(self.settings.gpu_devices)
        # self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.settings.dataset == "stanford2d3d":
            train_dataset = Stanford2D3D(self.settings.data_path, depth=False, fold=self.settings.train_fold,
                                     hw=self.settings.hw, mask_black=True, flip=self.settings.flip, rotate=True, is_training=True)
            val_dataset = Stanford2D3D(self.settings.data_path, depth=False, fold=self.settings.test_fold,
                                       hw=self.settings.hw, mask_black=True, flip=False, rotate=False,
                                       color_augmentation=False, is_training=False)
        elif self.settings.dataset == 'matterport3d':
            train_dataset = Matterport3D(list_file="/media/lby/lby_8t/dataset/pano3D/M3D_high/train.txt", depth=False,
                                         hw=self.settings.hw, mask_black=True, flip=self.settings.flip, rotate=True,
                                         is_training=True)
            val_dataset = Matterport3D(list_file="/media/lby/lby_8t/dataset/pano3D/M3D_high/test.txt", depth=False,
                                       hw=self.settings.hw, mask_black=True, flip=False, rotate=False,
                                       color_augmentation=False, is_training=False)
        elif self.settings.dataset == '3d60':
            train_dataset = ThreeD60(root="/media/lby/lby_8t/dataset/3d60", list_file="/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/datasets/3d60_train.txt", depth=False,
                                         hw=self.settings.hw, mask_black=True, flip=self.settings.flip, rotate=True,
                                         is_training=True)
            val_dataset = ThreeD60(root="/media/lby/lby_8t/dataset/3d60", list_file="/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/datasets/3d60_test.txt", depth=False,
                                         hw=self.settings.hw, mask_black=True, flip=self.settings.flip, rotate=True,
                                         is_training=False)
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)


        self.model = SphereTrans()
        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-7, T_max=20)

        if self.settings.load_weights_dir != '':
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            if not os.path.exists(os.path.join(self.log_path, mode)):
                os.makedirs(os.path.join(self.log_path, mode))
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = self.settings.start_epoch
        self.step = 0
        self.start_time = time.time()
        best_rmse = 10
        for self.epoch in range(self.settings.start_epoch+1, self.settings.num_epochs):
            self.train_one_epoch()
            rmse = self.validate()
            if rmse < best_rmse:
                best_rmse = rmse
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            self.step += 1
        self.scheduler.step()
        writer = self.writers['train']
        print(f'EP[{self.epoch}/{self.settings.num_epochs}] train:  ' + f'loss: {losses["loss"]}')

        writer.add_scalar('train-loss', losses["loss"], self.epoch)

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb", "name"]:
                inputs[key] = ipt.to(self.device)

        losses = {}
        equi_inputs = inputs["normalized_rgb"]  # * inputs["val_mask"]


        outputs = self.model(equi_inputs)
        gt = inputs["gt_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        gradient = Gradient_Net()
        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(outputs["pred_depth"] )
        # dmap = get_dmap(self.settings.batch_size)
        losses["loss"] = self.compute_loss(inputs["gt_depth"].float() * inputs["val_mask"], outputs["pred_depth"]) + \
                         self.compute_loss(G_x, p_x) + \
                         self.compute_loss(G_y, p_y)
        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]
                # mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth)

        self.evaluator.print()

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)

        return losses["err/rms"]

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.epoch)

        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.epoch)
            # writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data / inputs["gt_depth"][j].data.max(), self.epoch)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data / outputs["pred_depth"][j].data.max(), self.epoch)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        print("model saved at epoch{}".format(self.epoch))
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        output = {"output_proj.proj.0.weight", "output_proj.proj.0.bias", "output_proj.proj.2.weight", "output_proj.proj.2.bias", "output_proj.proj.2.running_mean", "output_proj.proj.2.running_var", "output_proj.proj.2.num_batches_tracked", "output_proj.proj.3.weight", "output_proj.proj.3.bias"}
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in output}

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

