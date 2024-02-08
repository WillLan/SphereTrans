import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from numpy import sin, cos, tan, pi, arcsin, arctan
from network.sphereNet import gen_grid_coordinates
import math
import os


def genSpherePoint(h, w, kh, kw, stride=1):
    coordinates = gen_grid_coordinates(h, w, stride) # (1, h*3, w*3, 2)

    # (1, H, W, Kh*Kw, 2)
    coordinates = coordinates.reshape(1, h, kh, w, kw, -1)
    coordinates = coordinates.transpose(0, 1, 3, 2, 4, 5).reshape(1, h, w, kh*kw, -1)
    with torch.no_grad():
      coordinates = torch.FloatTensor(coordinates)
      coordinates.requires_grad = False

    return coordinates

def cal_pos_radian(h, w, h_idx, w_idx):
    phi = -(h_idx / h * pi - pi / 2)
    theta = w_idx / w * 2 * pi - pi
    position = np.array([phi, theta])

    return position

def great_circle_distance(lon1, lat1, lon2, lat2): # 弧度

    # 计算经度和纬度之间的差值
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    # 使用haversine公式计算great circle distance
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 1  #
    distance = c * r

    return distance # 输出为弧度

def EuclideanDistance(x1, y1, x2, y2):
    distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return distance


import matplotlib.pyplot as plt

def genEuDistance(h, w, ph, pw):
    # (H/ph, W/pw, 2)
    embedding_pos = np.array([[np.array([j, i]) for j in range(int(pw/2), w, pw)] for i in range(int(ph/2), h, ph)])
    #print(embedding_pos.shape)
    h1, w1, xy= embedding_pos.shape

    embedding_pos_flat = embedding_pos.reshape(h1*w1, xy)
    # (H/ph*W/pw, H/ph*W/pw)
    havDis =np.array([[EuclideanDistance(embedding_pos_flat[o][1], embedding_pos_flat[o][0],
                                             embedding_pos_flat[m][1], embedding_pos_flat[m][0])
                       for m in range(0, h1*w1, 1)] for o in range(0, h1*w1, 1)])


    #print(havDis.max())
    embedding_pos = (embedding_pos - embedding_pos.min()) / (embedding_pos.max() - embedding_pos.min())
    havDis = 1- (havDis - havDis.min()) / (havDis.max() - havDis.min())


    #return embedding_pos.cuda(), havDis.cuda()

    plt.matshow(havDis, cmap=plt.cm.jet)
    plt.show()
    print(havDis, havDis.shape)




def genHaversinDistance(h, w, ph, pw):
    # (H/ph, W/pw, 2)
    embedding_pos = np.array([[cal_pos_radian(h, w, i, j) for j in range(int(pw/2), w, pw)] for i in range(int(ph/2), h, ph)])
    #print(embedding_pos.shape)
    h1, w1, xy= embedding_pos.shape

    embedding_pos_flat = embedding_pos.reshape(h1*w1, xy)
    # (H/ph*W/pw, H/ph*W/pw)
    havDis =np.array([[great_circle_distance(embedding_pos_flat[o][1], embedding_pos_flat[o][0],
                                             embedding_pos_flat[m][1], embedding_pos_flat[m][0])
                       for m in range(0, h1*w1, 1)] for o in range(0, h1*w1, 1)])

    havDis[havDis == 0] = 1e-16
    #print(havDis.max())
    embedding_pos = (embedding_pos - embedding_pos.min()) / (embedding_pos.max() - embedding_pos.min())
    havDis = 1- (havDis - havDis.min()) / (havDis.max() - havDis.min())
    with torch.no_grad():
      embedding_pos = torch.FloatTensor(embedding_pos)
      embedding_pos.requires_grad = False
      havDis = torch.FloatTensor(havDis)
      havDis.requires_grad = False
    #print(embedding_pos, havDis.shape)


    return embedding_pos.cuda(), havDis.cuda()

    # plt.matshow(havDis, cmap=plt.cm.jet)
    # plt.show()
    # print(havDis, havDis.shape)
if __name__ == '__main__':
    import math


    # def great_circle_distance(lon1, lat1, lon2, lat2):
    #     # 将经纬度转化为弧度
    #     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    #     #print(lon1, lat1, lon2, lat2)
    #
    #     # 计算经度和纬度之间的差值
    #     delta_lon = lon2 - lon1
    #     delta_lat = lat2 - lat1
    #
    #     # 使用haversine公式计算great circle distance
    #     a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    #     c = 2 * math.asin(math.sqrt(a))
    #     r = 1  #
    #     distance = c * r
    #
    #     return distance # 输出为弧度
    #
    # print(great_circle_distance(180, 0, -90, 0)) # 3.14

    #print(genHaversinDistance(256, 512, 32, 32))
    genEuDistance(256, 512, 32, 32)