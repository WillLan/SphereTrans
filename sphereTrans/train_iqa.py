import os, argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from iqa_dataset import IQADataset
from network.model_iqa import SphereTrans
from scipy import stats
from scipy.optimize import curve_fit
import tqdm

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="No reference 360 degree image quality assessment.")
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default='cuda:1', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate.',
                        default=1e-4, type=float)

    parser.add_argument('--model_name', default='replace_linear', type=str)
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
                        default='ODI_IQA_crop', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=8, type=int)

    parser.add_argument('--save_path', help='Path of model to save',
                        default='./iqa_logs', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    print(args.model_name)
    print(gpu)

    num_epochs = args.num_epochs
    batch_size = args.batch_size

    lr = args.lr

    # 保存日志地址
    log_folder = os.path.join(args.save_path, args.model_name, args.database)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    writers = {}
    for mode in ["train", "val"]:
        if not os.path.exists(os.path.join(log_folder, mode)):
            os.makedirs(os.path.join(log_folder, mode))
        writers[mode] = SummaryWriter(os.path.join(log_folder, mode))

    # load the network
    model = SphereTrans(device=gpu)

    train_transformations = transforms.Compose([transforms.RandomCrop((256,512)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transformations = transforms.Compose([transforms.RandomCrop((256, 512)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = IQADataset(data_dir='/media/lby/lby_8t/dataset/QA/ODI_IQA/image_resize',
                               mos_dir='/media/lby/lby_8t/dataset/QA/ODI_IQA/train_mos_dmos.txt',
                               transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=8)

    test_dataset = IQADataset(data_dir='/media/lby/lby_8t/dataset/QA/ODI_IQA/image_resize',
                              mos_dir='/media/lby/lby_8t/dataset/QA/ODI_IQA/test_mos_dmos.txt',
                              transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model.to(gpu)
    params_conv1 = sum(p.numel() for p in model.conv1.parameters())
    print("Conv1 层的参数量：", params_conv1)
    criterion = nn.MSELoss()

    # regression loss coefficient
    parameter = list(model.parameters())
    #optimizer = torch.optim.RMSprop(parameter, lr=lr, alpha=0.9)
    optimizer = torch.optim.Adam(parameter, lr=lr, weight_decay=5e-4)

    print("Ready to train network")

    best_val_criterion = -1  # SROCC min
    best_val = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        pbar = tqdm.tqdm(train_loader)
        pbar.set_description("Train Epoch_{}".format(epoch + 1))
        for i, (img, mos) in enumerate(pbar):
            img = img.to(gpu)

            mos = mos.to(gpu)
            # print(mos.shape)

            # Forward pass
            mos_predict = model(img)
            mos_predict = mos_predict.squeeze(-1)
            # print(mos_predict.shape)
            # print(mos_predict)

            # MSE loss
            loss = criterion(mos_predict, mos)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()
            optimizer.step()
            # if (i + 1) % 100 == 0:
            #     session_end_time = time.time()
            #     avg_loss_epoch = sum(batch_losses_each_disp) / 100
            #     print('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f CostTime: %.4f'
            #           % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, avg_loss_epoch,
            #              session_end_time - session_start_time))
            #     session_start_time = time.time()
            #     batch_losses_each_disp = []

        avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)    # 每批次的平均损失值，通过将损失总和除以总批次数得到。
        writers['train'].add_scalar('train-loss', avg_loss, epoch+1)     # tensorboard
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(test_dataset)])
            y_output = np.zeros([len(test_dataset)])
            pbar = tqdm.tqdm(test_loader)
            pbar.set_description("Val Epoch_{}".format(epoch+1))
            for i, (img,  mos) in enumerate(pbar):
                img = img.to(gpu)

                mos = mos.to(gpu)
                # print('label',mos.item())
                label[i] = mos.item()

                mos_predict = model(img)
                #print(mos_predict.shape)
                #print('pre',mos_predict.item())
                y_output[i] = mos_predict.item()


            y_output_logistic = fit_function(label, y_output)
            # print(y_output)
            # print(label)
            # print(y_output)
            val_PLCC = stats.pearsonr(y_output_logistic, label)[0]
            val_SRCC = stats.spearmanr(y_output, label)[0]
            val_KRCC = stats.kendalltau(y_output_logistic, label)[0]
            val_RMSE = np.sqrt(((y_output_logistic - label) ** 2).mean())

            print('Epoch {} completed. SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1,
                                                                                                          val_SRCC,
                                                                                                          val_KRCC,
                                                                                                          val_PLCC,
                                                                                                          val_RMSE))
            writers['val'].add_scalar('SRCC', avg_loss, epoch+1)
            writers['val'].add_scalar('KRCC', avg_loss, epoch+1)
            writers['val'].add_scalar('PLCC', avg_loss, epoch+1)
            writers['val'].add_scalar('RMSE', avg_loss, epoch+1)

            # save the best model
            if val_SRCC > best_val_criterion:
                print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                best_val_criterion = val_SRCC
                best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]
                print('Saving model...')

                save_folder = os.path.join(log_folder, "models", "weights_{}".format(epoch+1))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = os.path.join(save_folder, "{}.pth".format("model"))
                torch.save(model.state_dict(), save_path)

                save_path = os.path.join(save_folder, "{}.pth".format("optim"))
                torch.save(optimizer.state_dict(), save_path)

    print('Training completed.')
    print('The best training result SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_val[0], best_val[1], best_val[2], best_val[3]))








