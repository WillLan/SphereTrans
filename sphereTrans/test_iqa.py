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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default='cuda:0', type=str)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate.',
                        default=1e-4, type=float)

    parser.add_argument('--model_name', default='bottleneck_depth=2', type=str)
    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
                        default='OIQA', type=str)
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



    # load the network
    model = SphereTrans(device=gpu)
    transformations = transforms.Compose([transforms.Resize((512,1024)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    test_dataset = IQADataset(data_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/image_resize',
                              mos_dir='/media/lby/lby_8t/dataset/QA/other_IQA360_datasets/OIQA/test_mos.txt',
                              transform=transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model.to(gpu)
    model.load_state_dict(torch.load('/media/lby/lby_8t/pano_seg/sphereTrans/sphereTrans/iqa_logs/bottleneck_depth=2/OIQA/models/weights_35/model.pth'))
     # do validation after each epoch
    with torch.no_grad():
        model.eval()
        label = np.zeros([len(test_dataset)])
        y_output = np.zeros([len(test_dataset)])
        pbar = tqdm.tqdm(test_loader)
        for i, (img,  mos) in enumerate(pbar):
            img = img.to(gpu)
            mos = mos.to(gpu)

            label[i] = mos.item()
            mos_predict = model(img)

            y_output[i] = mos_predict.item()
        y_output_logistic = fit_function(label, y_output)

        val_PLCC = stats.pearsonr(y_output_logistic, label)[0]
        val_SRCC = stats.spearmanr(y_output, label)[0]
        val_KRCC = stats.kendalltau(y_output_logistic, label)[0]
        val_RMSE = np.sqrt(((y_output_logistic - label) ** 2).mean())

        print('completed. SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(val_SRCC,
                                                                                                        val_KRCC,
                                                                                                        val_PLCC,
                                                                                                        val_RMSE))

        with open("./predict_mos.txt", 'w') as f:
            for i in range(len(y_output)):
                f.write(str(label[i]) + '  '+ str(y_output[i]) + '\n')



    plt.scatter(label, y_output,c='b')
    plt.xlabel('truth')
    plt.ylabel('predict')

    plt.show()



