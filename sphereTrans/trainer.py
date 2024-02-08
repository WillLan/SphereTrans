from __future__ import absolute_import, division, print_function
import os

import cv2
import numpy as np
import time
import json
import tqdm
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from collections import Counter
from imageio import imwrite
import datasets
from network.model_3 import SphereTrans as SphereTrans
from metrics import compute_depth_metrics, Evaluator
from stanford2d3d import Stanford2D3D
# from losses import BerhuLoss
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        self.epoch = 0
        self.device = torch.device(self.settings.device)
        # self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # checking the input height and width are multiples of 32
        assert self.settings.hw[0] % 32 == 0, "input height must be a multiple of 32"
        assert self.settings.hw[1] % 32 == 0, "input width must be a multiple of 32"

        # data
        # datasets_dict = {"3d60": datasets.ThreeD60,
        #                  "panosuncg": datasets.PanoSunCG,
        #                  "stanford2d3d": datasets.Stanford2D3D,
        #                  "matterport3d": datasets.Matterport3D}
        # self.dataset = datasets_dict[self.settings.dataset]

        train_dataset = Stanford2D3D(self.settings.data_path, depth=self.settings.depth, fold=self.settings.train_fold, hw=self.settings.hw, mask_black=True, flip=self.settings.flip, rotate=True, is_training=True)
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True, num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        val_dataset = Stanford2D3D(self.settings.data_path, depth=self.settings.depth, fold=self.settings.test_fold, hw=self.settings.hw, mask_black=True, flip=False, rotate=False, is_training=False)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False, num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)

        # network
        self.model = SphereTrans(device=self.device)
        self.model.to(self.device)
        # optimizer
        self.parameters_to_train = list(self.model.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-7, T_max=20)

        if self.settings.load_weights_dir != '':
            print(1)
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

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
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        best_miou = 0
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            miou = self.validate()
            if miou > best_miou:
                best_miou = miou
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        epoch_losses = Counter()  # 用于统计可迭代对象中每个元素出现的次数，并返回一个字典

        for batch_idx, inputs in enumerate(pbar):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)
            outputs = self.model(inputs['rgb'])
            self.optimizer.zero_grad()
            # print(outputs.shape)
            losses = self.compute_losses(inputs["sem"], outputs)
            losses["total"].backward()
            self.optimizer.step()

            bs = self.settings.batch_size
            epoch_losses['N'] += bs
            for k, v in losses.items():
                if torch.is_tensor(v):
                    epoch_losses[k] += bs * v.item()
                else:
                    epoch_losses[k] += bs * v

        self.scheduler.step()

        # Statistic over the epoch
        n = epoch_losses.pop('N')
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / n

        # log & tensorboard
        writer = self.writers['train']
        print(f'EP[{self.epoch}/{self.settings.num_epochs}] train:  ' + ' \ '.join([f'{k} {v:.3f}' for k, v in losses.items()]))

        writer.add_scalar('train-loss', epoch_losses['total'], self.epoch)

    # def process_batch(self, inputs):
    #     for key, ipt in inputs.items():
    #         if key not in ["rgb", "cube_rgb"]:
    #             inputs[key] = ipt.to(self.device)
    #
    #     losses = {}
    #
    #     equi_inputs = inputs["normalized_rgb"]
    #
    #
    #     outputs = self.model(equi_inputs)
    #
    #     gt = inputs["gt_depth"] * inputs["val_mask"]
    #     pred = outputs["pred_depth"] * inputs["val_mask"]
    #     outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
    #
    #     losses["loss"] = self.compute_loss(inputs["gt_depth"].float() * inputs["val_mask"], outputs["pred_depth"])
    #
    #     return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()
        epoch_losses = Counter()

        pbar = tqdm.tqdm(self.val_loader)
        # pbar.set_description("Validating Epoch_{}".format(self.epoch))
        cm = 0
        cmap = (plt.get_cmap('gist_rainbow')(np.arange(13) / 13)[..., :3] * 255).astype(np.uint8)  # 这段代码的作用是根据 num_classes 的数量，从 'gist_rainbow' 颜色映射中获取一组不同的颜色，并将它们表示为整数的 RGB 值。这些颜色通常用于将不同类别或标签在可视化中以不同的颜色进行区分。
        vis_dir = self.settings.vis_dir
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                # 数据转换到cuda
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(self.device)
                outputs = self.model(inputs['rgb'])
                losses = self.compute_losses(inputs["sem"], outputs)

                for k, v in losses.items():
                    if torch.is_tensor(v):
                        epoch_losses[k] += float(v.item()) / len(self.val_loader)
                    else:
                        epoch_losses[k] += v / len(self.val_loader)

                # 计算miou 和 macc
                sem = inputs['sem']
                mask = (sem >= 0)
                gt = sem[mask]
                pred = outputs.argmax(1)[mask]
                assert gt.min() >= 0 and gt.max() < 13 and outputs.shape[1] == 13
                cm += np.bincount((gt * 13 + pred).cpu().numpy(), minlength=13 ** 2)

                #vis
                vis_sem = cmap[outputs[0].argmax(0).cpu().numpy()]
                vis_sem = vis_sem.astype(np.uint8)
                imwrite(os.path.join(vis_dir, inputs['name'][0].strip() + '.png'), vis_sem)

        print('  Summarize  '.center(50, '='))
        cm = cm.reshape(13, 13)
        id2class = np.array( ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table','wall', 'window'])
        valid_mask = (cm.sum(1) != 0)
        cm = cm[valid_mask][:, valid_mask]
        id2class = id2class[valid_mask]
        inter = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - inter
        ious = inter / union
        accs = inter / cm.sum(1)  # 按行求和为每一个类别真实出现的像素个数
        #print(f'EP[{self.epoch}/{self.settings.num_epochs}] valid:  ' + ' \ '.join([f'{k} {v:.3f}' for k, v in epoch_losses.items()]))
        for name, iou, acc in zip(id2class, ious, accs):
            print(f'{name:20s}:    iou {iou * 100:5.2f}    /    acc {acc * 100:5.2f}')
        print(f'{"Overall":20s}:    iou {ious.mean() * 100:5.2f}    /    acc {accs.mean() * 100:5.2f}')

        writer = self.writers['val']
        writer.add_image("rgb", inputs["rgb"][0].data, self.epoch)


        label = cmap[inputs['sem'][0].cpu().numpy()]
        writer.add_image("semantic_label", label, self.epoch, dataformats='HWC')

        pred_sem = cmap[outputs[0].argmax(0).cpu().numpy()]
        writer.add_image('pred_visualize', pred_sem, dataformats='HWC', global_step=self.epoch)

        writer.add_scalar('valid-loss', epoch_losses['total'], self.epoch)
        writer.add_scalar('valid-miou', ious.mean(), self.epoch)
        writer.add_scalar('valid-macc', accs.mean(), self.epoch)
        return ious.mean()

    # 加入dice-loss
    def compute_losses(self, inputs, outputs):

        losses = {}
        mask = (inputs >= 0)
        gt = inputs[mask]
        outputs = outputs.permute(0, 2, 3, 1)[mask]
        ce = F.cross_entropy(outputs.float(), gt.long(), reduction='none')    # 注意此处的数据类型
        ce = ce[~torch.isinf(ce) & ~torch.isnan(ce)]
        losses['total'] = ce.mean()
        losses['acc'] = (outputs.argmax(1) == gt).float().mean()
        return losses

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
        pretrained_dict = torch.load(path, map_location=torch.device('cuda:0'))
        for k in model_dict.keys():
            #print(k)
            if k not in pretrained_dict.keys():
                print(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
