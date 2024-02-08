"""
## PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation
## Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, Yao Zhao
## https://arxiv.org/abs/2203.09283
## The code is reproducted based on uformer:https://github.com/ZhendongWang6/Uformer
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from timm.models.resnet import _cfg
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from network.gla import *

from network.SampleSpherePoint import genSpherePoint, genHaversinDistance


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels):
        super(StripPooling, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        # self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.ac = nn.Sigmoid()
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        out = self.conv3(x1 + x2)
        out_att = self.ac(out)
        return out_att


#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


#########################################
########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., flag=0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())
        # self.hw = StripPooling(hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = H

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)
        # bs,hidden_dim,32x32
        # att = self.hw(x)

        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)

        # x = x * att

        # x = self.active(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),

        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),    # 反卷积操作
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=0),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    # https://blog.csdn.net/jorg_zhao/article/details/105295686 pad
    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width # 从w, h维度依次，左右1,1,上下1，1
        x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C (512, 256)
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # for 1024*512
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


#########################################
########### SphereTransformer #############
class SphereTransBlock(nn.Module):
    def __init__(self, depth_atten, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False, ref_point=None, flag=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point  # generate_ref_points(self.input_resolution[1], self.input_resolution[0])

        self.norm1 = norm_layer(dim)

        self.dattn = SphereSelfAttention(depth_atten, num_heads, dim, k=9,
                                         last_feat_height=self.input_resolution[0],
                                         last_feat_width=self.input_resolution[1], dropout=0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, flag=flag)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        x = self.dattn(x, self.ref_point.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


########### Basic layer of Uformer ################
class BasicTransformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth_atten, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False, ref_point=None, flag=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            SphereTransBlock(depth_atten=depth_atten, dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                             se_layer=se_layer, ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


########### Uformer ################
class SphereTrans(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths_atten=[1, 1, 1, 1, 1, 1, 1, 1, 1], depths=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2], num_classes=13,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, device='cuda:1', **kwargs):
        super(SphereTrans, self).__init__()
        self.num_enc_layers = len(depths) // 2  # 四层encoder
        self.num_dec_layers = len(depths) // 2  # 四层decoder
        self.embed_dim = embed_dim  # 初始embedding的维度 32
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.ref_point256x512 = genSpherePoint(256, 512, 3, 3).to(device)  # torch.load("network6/Equioffset256x512.pth")
        self.ref_point128x256 = genSpherePoint(128, 256, 3, 3).to(device) # torch.load("network6/Equioffset128x256.pth")
        self.ref_point64x128 = genSpherePoint(64, 128, 3, 3).to(device) # torch.load("network6/Equioffset64x128.pth")
        self.ref_point32x64 = genSpherePoint(32, 64, 3, 3).to(device)  ##torch.load("network6/Equioffset32x64.pth")
        self.ref_point16x32 = genSpherePoint(16, 32, 3, 3).to(device)  # torch.load("network6/Equioffset16x32.pth")

        # self.sphere_postion256x512 = genHaversinDistance(256, 512, 32, 32)
        # self.sphere_postion128x256 = genHaversinDistance(128, 256, 16, 16)
        # self.sphere_postion64x128 = genHaversinDistance(64, 128, 8, 8)
        # self.sphere_postion32x64 = genHaversinDistance(32, 64, 4, 4)
        # self.sphere_postion16x32 = genHaversinDistance(16, 32, 2, 2)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth  # drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，不经过主干，而直接由分支进行恒等映射。
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        # self.pre_block = PreprocBlock(in_channels=3, out_channels=64, kernel_size_lst=[[3, 9], [5, 11], [5, 7], [7, 7]])

        # 经过input_proj后为(B, H*W, C) HW为256, 512
        # self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.GELU)  # stride = 2 for 1024*512
        # 预训练特征提取
        config = _cfg(url='', file='/media/lby/lby_8t/backbone/resnet/resnet101.pth')
        self.feature_extract = timm.create_model('resnet101', pretrained=True, features_only=True, pretrained_cfg=config)
        self.con1 = nn.Conv2d(2048, embed_dim, kernel_size=1, stride=1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=embed_dim, kernel_size=2, stride=2)
        )

        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=num_classes, kernel_size=3, stride=1,          # 输出层
                                      input_resolution=(img_size, img_size * 2))
        #self.sphereVit = SphereViT(patch_size=32, dim=1024, depth=8, heads=8, channels=32, dim_head=64)
        # Encoder

        self.encoderlayer_0 = BasicTransformerLayer(dim=embed_dim,
                                                    output_dim=embed_dim,
                                                    input_resolution=(img_size, img_size * 2),
                                                    depth_atten=depths_atten[0],
                                                    depth=depths[0],
                                                    num_heads=num_heads[0],
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point256x512,
                                                    flag=0)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2, input_resolution=(img_size, img_size * 2))
        self.encoderlayer_1 = BasicTransformerLayer(dim=embed_dim * 2,
                                                    output_dim=embed_dim * 2,
                                                    input_resolution=(img_size // 2, img_size * 2 // 2),
                                                    depth_atten=depths_atten[1],
                                                    depth=depths[1],
                                                    num_heads=num_heads[1],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point128x256,
                                                    flag=0)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.encoderlayer_2 = BasicTransformerLayer(dim=embed_dim * 4,
                                                    output_dim=embed_dim * 4,
                                                    input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                    depth_atten=depths_atten[2],
                                                    depth=depths[2],
                                                    num_heads=num_heads[2],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point64x128,
                                                    flag=0)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8,
                                     input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.encoderlayer_3 = BasicTransformerLayer(dim=embed_dim * 8,
                                                    output_dim=embed_dim * 8,
                                                    input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                    depth_atten=depths_atten[3],
                                                    depth=depths[3],
                                                    num_heads=num_heads[3],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point32x64,
                                                    flag=0)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16,
                                     input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))

        # Bottleneck
        self.conv = BasicTransformerLayer(dim=embed_dim * 8,
                                          output_dim=embed_dim * 8,
                                          input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                          depth_atten=depths_atten[4],
                                          depth=depths[4],
                                          num_heads=num_heads[4],

                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=conv_dpr,
                                          norm_layer=norm_layer,
                                          use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer,
                                          ref_point=self.ref_point32x64, flag=0)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = BasicTransformerLayer(dim=embed_dim * 16,
                                                    output_dim=embed_dim * 16,
                                                    input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                    depth_atten=depths_atten[5],
                                                    depth=depths[5],
                                                    num_heads=num_heads[5],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[:depths[5]],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point32x64,
                                                    flag=1)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 4,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = BasicTransformerLayer(dim=embed_dim * 8,
                                                    output_dim=embed_dim * 8,
                                                    input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                    depth_atten=depths_atten[6],
                                                    depth=depths[6],
                                                    num_heads=num_heads[6],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point64x128,
                                                    flag=1)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = BasicTransformerLayer(dim=embed_dim * 4,
                                                    output_dim=embed_dim * 4,
                                                    input_resolution=(img_size // 2, img_size * 2 // 2),
                                                    depth_atten=depths_atten[7],
                                                    depth=depths[7],
                                                    num_heads=num_heads[7],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point128x256,
                                                    flag=1)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = BasicTransformerLayer(dim=embed_dim * 2,
                                                    output_dim=embed_dim * 2,
                                                    input_resolution=(img_size, img_size * 2),
                                                    depth_atten=depths_atten[8],
                                                    depth=depths[8],
                                                    num_heads=num_heads[8],

                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer, ref_point=self.ref_point256x512,
                                                    flag=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp}"

    def forward(self, x):
        # Input Projection
        # y = self.pre_block(x)

        # y = self.input_proj(x)

        # y = self.sphereVit(y, self.sphere_postion256x512)
        # y = self.pos_drop(y) # torch.Size([1, 32, 256, 512])
        # y = y.permute(0, 2, 3, 1).contiguous()
        # B, H, W, C = y.shape
        # y = y.view(B, -1, C)
        #print(y.shape)
        #print('Starting running######################')
        # Encoder
        y = self.feature_extract(x)[4]   # 预计训练提取
        # y = self.con1(y)
        # y = F.interpolate(y, size=[256, 512], mode="bilinear")
        y = self.upsample(y)
        y = y.flatten(2).transpose(1, 2).contiguous()

        conv0 = self.encoderlayer_0(y)  # (256, 512 )
        pool0 = self.dowsample_0(conv0)  # (128, 256)
        conv1 = self.encoderlayer_1(pool0)  # (128, 256)
        pool1 = self.dowsample_1(conv1)  # (64, 128)
        conv2 = self.encoderlayer_2(pool1)  # (64, 128)
        pool2 = self.dowsample_2(conv2)  # (32, 64)
        # conv3 = self.encoderlayer_3(pool2)  # (32, 64)
        # pool3 = self.dowsample_3(conv3)  # (16, 32)

        # Bottleneck
        conv4 = self.conv(pool2)  # (16, 32)
        #
        # # Decoder
        # up0 = self.upsample_0(conv4)  # (32, 64)
        # deconv0 = torch.cat([up0, conv3], -1)  # (32, 64)
        # deconv0 = self.decoderlayer_0(deconv0)  # (32, 64)

        up1 = self.upsample_1(conv4)  # (64, 128)
        deconv1 = torch.cat([up1, conv2], -1)  # (64, 128)
        deconv1 = self.decoderlayer_1(deconv1)  # (64, 128)

        up2 = self.upsample_2(deconv1)  # (128, 256)
        deconv2 = torch.cat([up2, conv1], -1)  # (128, 256)
        deconv2 = self.decoderlayer_2(deconv2)  # (128, 256)

        up3 = self.upsample_3(deconv2)  # (256, 512 )
        deconv3 = torch.cat([up3, conv0], -1)  # (256, 512 )
        deconv3 = self.decoderlayer_3(deconv3)  # (256, 512)

        # Output Projection

        y = self.output_proj(deconv3)  # (512, 1024)

        return y


if __name__ == '__main__':
    a = genSpherePoint(128, 256, 3, 3).cuda()
    print(a.shape)

