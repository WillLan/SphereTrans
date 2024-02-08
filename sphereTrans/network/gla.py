from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pylab import  *
import scipy.misc
from torchvision import models, transforms
from models.sphereVit import AttentionBlock

def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):  # model的第一个Sequential()是有多层，所以遍�?
            x = layer(x)  # torch.Size([1, 64, 55, 55])生成�?4个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(
        feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大�?
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    #upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    #feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道�?
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，�?4个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        #plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        #scipy.misc.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1])
    plt.show()

def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid.cuda()


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):
    new_point = ref_point.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point


class SphereSelfAttention(nn.Module):
    def __init__(self, depth_atten, h,
                 d_model,
                 k,
                 last_feat_height,
                 last_feat_width,
                 dropout=0.1,
                ):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(SphereSelfAttention, self).__init__()
        #assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_head = int(d_model / h) # head的维度 一直是32
        self.h = h # head数量

        self.input_proj = nn.Linear(d_model, d_model) # (B, H, W, C)
        self.point_proj = nn.Linear(2, d_model) # (B, H, W, C)
        self.attend = AttentionBlock(dim=d_model, depth=depth_atten, heads=h, dim_head=self.d_head, dropout = 0.)


    def forward(self, input, sphere_point):
        #B, H, W, C = input.shape
        # B, H, W, Kh*Kw, 2
        B, H, W, KhKw, xy_dim = sphere_point.shape
        input = input.view(B, H, W, -1)  # (B, H, W, C)
        input = self.input_proj(input)  # (B, H, W, C)
        input = input.permute(0, 3, 1, 2)  # (B, C, H, W)
        #print(sphere_point.shape)

        # 对于图像中的每一个像素值，获得其周围的九个像素值。
        sph_pt_feat = sphere_point.view(B, H, W, 3, 3, xy_dim)
        sph_pt_feat = sph_pt_feat.permute(0, 1, 3, 2, 4, 5).contiguous() # B, H, 3, W, 3, 2
        sph_pt_feat = sph_pt_feat.view(B, 3*H, 3*W, -1) # (B, 3H, 3W, 2)

        # 按照球面坐标从ERP图像中sample值，然后与球面坐标的embedding相加, mode='bilinear': 这是空间插值的模式，指定了在采样点之间如何进行插值。在这里，使用的是双线性插值，它会考虑采样点周围的四个像素的权重，以生成插值结果。
        input = nn.functional.grid_sample(input, sph_pt_feat, mode='bilinear') # (B, C, 3H, 3W)
        #input = input.permute(0, 3, 1, 2).contiguous()  # (B, C, 3H, 3W)
        #print(input.shape)
        sph_pt_feat_emd = self.point_proj(sph_pt_feat) # (B, 3H, 3W, 2) -> (B, 3H, 3W, C)
        #print(sph_pt_feat_emd.shape)
        sph_pt_feat_emd = sph_pt_feat_emd.permute(0, 3, 1, 2).contiguous()  # (B, C, 3H, 3W)
        input_pe = input + sph_pt_feat_emd # input+PE  (B, C, 3H, 3W)
        # 输入到球面attention中进行计算
        output_att =self.attend(input_pe, sph_pt_feat_emd) # (B, C, H, W)

        return output_att.permute(0, 2, 3, 1).contiguous().view(B, H*W, -1)  # (B, H*W, C)




# class PanoSelfAttention(nn.Module):
#     def __init__(self, h,
#                  d_model,
#                  k,
#                  last_feat_height,
#                  last_feat_width,
#                  scales=1,
#                  dropout=0.1,
#                  need_attn=False):
#         """
#         :param h: number of self attention head
#         :param d_model: dimension of model
#         :param dropout:
#         :param k: number of keys
#         """
#         super(PanoSelfAttention, self).__init__()
#         #assert h == 8  # currently header is fixed 8 in paper
#         assert d_model % h == 0
#         # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
#         self.d_k = int(d_model / h) # head的维度 一直是32
#         self.h = h
#
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#
#         self.scales_hw = []
#         for i in range(scales):
#             self.scales_hw.append([last_feat_height * 2 ** i,
#                                    last_feat_width * 2 ** i])
#
#         self.dropout = None
#         if self.dropout:
#             self.dropout = nn.Dropout(p=dropout)
#
#         self.k = k # 9   ----  sampling number
#         self.scales = scales
#         self.last_feat_height = last_feat_height
#         self.last_feat_width = last_feat_width
#
#         self.offset_dims = 2 * self.h * self.k * self.scales # 2*9
#         self.A_dims = self.h * self.k * self.scales # 9
#
#         # 2MLK for offsets MLK for A_mlqk
#         self.offset_proj = nn.Linear(d_model, self.offset_dims)
#         self.A_proj = nn.Linear(d_model, self.A_dims)
#
#         self.wm_proj = nn.Linear(d_model, d_model)
#         self.need_attn = need_attn
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.constant_(self.offset_proj.weight, 0.0)
#         torch.nn.init.constant_(self.A_proj.weight, 0.0)
#
#         torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))
#
#         def init_xy(bias, x, y):
#             torch.nn.init.constant_(bias[:, 0], float(x))
#             torch.nn.init.constant_(bias[:, 1], float(y))
#
#         # caution: offset layout will be  M, L, K, 2
#         bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)
#
#         # init_xy(bias[0], x=-self.k, y=-self.k)
#         # init_xy(bias[1], x=-self.k, y=0)
#         # init_xy(bias[2], x=-self.k, y=self.k)
#         # init_xy(bias[3], x=0, y=-self.k)
#         # init_xy(bias[4], x=0, y=self.k)
#         # init_xy(bias[5], x=self.k, y=-self.k)
#         # init_xy(bias[6], x=self.k, y=0)
#         # init_xy(bias[7], x=self.k, y=self.k)
#
#     def forward(self,
#                 query: torch.Tensor,
#                 keys: List[torch.Tensor],
#                 ref_point: torch.Tensor,
#                 ):
#         """
#         :param key_masks:
#         :param query_mask:
#         :param query: B, H, W, C
#         :param keys: List[B, H, W, C]
#         :param ref_point: B, H, W, 2
#         :return:
#         """
#
#         assert len(keys) == self.scales
#
#         nbatches, query_height, query_width, _ = query.shape
#
#         # B, H, W, C
#         query = self.q_proj(query)
#
#         # B, H, W, 2MLK
#         offset = self.offset_proj(query) # B, H, W, 2*h*k
#         # B, H, W, M, 2LK
#         offset = offset.view(nbatches, query_height, query_width, self.h, -1) # B, H, W, h, 2*k
#
#         # B, H, W, MLK
#         A = self.A_proj(query) # B, H, W, h*k
#
#         # B, H, W, M, LK
#         A = A.view(nbatches, query_height, query_width, self.h, -1) # B, H, W, h, k
#         A = F.softmax(A, dim=-1) # B, H, W, h, 9    score
#
#         offset = offset.view(nbatches, query_height, query_width, self.h, self.scales, self.k, 2)
#         offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
#         # B*M, L, K, H, W, 2   # B*h, 1, 9, H, W, 2
#         offset = offset.view(nbatches * self.h, self.scales, self.k, query_height, query_width, 2)
#
#         A = A.permute(0, 3, 1, 2, 4).contiguous()
#         # B*M, H*W, LK    # B*h, H*W, 9
#         A = A.view(nbatches * self.h, query_height * query_width, -1)
#
#         scale_features = []
#         for l in range(self.scales):
#             feat_map = keys[l] # B, H, W, C
#             _, h, w, _ = feat_map.shape
#             #  1, H, W, Kh*Kw, 2
#             reversed_ref_point = ref_point #restore_scale(height=h, width=w, ref_point=ref_point)
#             #  B*h, H, W, K, 2
#             reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1, 1, 1)
#             # B, H, W, h, h_k
#             scale_feature = self.k_proj(feat_map).view(nbatches, h, w, self.h, self.d_k)
#
#             # B, h, h_k, H, W
#             scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
#             # B*h, h_k, H, W
#             scale_feature = scale_feature.view(-1, self.d_k, h, w)
#             k_features = []
#             #show_feature_map(scale_feature)
#             for k in range(self.k):
#                 # Bh, H, W, 2
#                 points = reversed_ref_point[:, :, :, k, :] + offset[:, l, k, :, :, :]#+ equi_offset[:, l, :, :, k, :] + offset[:, l, k, :, :, :]
#                 vgrid_x = 2.0 * points[:, :, :, 1] / max(w - 1, 1) - 1.0
#                 vgrid_y = 2.0 * points[:, :, :, 0] / max(h - 1, 1) - 1.0
#                 vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3) # Bh, H, W, 2
#                 #print(points)
#
#                 # B*h, h_k, H, W
#                 feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
#                 #show_feature_map(feat)
#
#
#
#                 k_features.append(feat)
#
#             # B*h, k, h_k, H, W
#             k_features = torch.stack(k_features, dim=1)
#             scale_features.append(k_features)
#
#         # B*h, L, K, h_k, H, W
#         scale_features = torch.stack(scale_features, dim=1)
#
#         # B*h, H*W, h_k, LK
#         scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
#         scale_features = scale_features.view(nbatches * self.h, query_height * query_width, self.d_k, -1)
#
#         # B*h, H*W, h_k                       # B*h, H*W, h_k, 9 x B*h, H*W, 9
#         feat = torch.einsum('nlds, nls -> nld', scale_features, A)
#
#         # B*h, H*W, h_k -> B, h, H, W, h_k
#         feat = feat.view(nbatches, self.h, query_height, query_width, self.d_k)
#         # B, h, H, W, h_k -> B, H, W, h, h_k
#         feat = feat.permute(0, 2, 3, 1, 4).contiguous()
#         # B, H, W, h, h_k -> B, H, W, C
#         feat = feat.view(nbatches, query_height, query_width, self.d_k * self.h)
#
#         feat = self.wm_proj(feat)
#         if self.dropout:
#             feat = self.dropout(feat)
#
#         return feat