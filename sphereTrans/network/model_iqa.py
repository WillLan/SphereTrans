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
from models.sphereVit import SphereViT
from network.SampleSpherePoint import genSpherePoint, genHaversinDistance

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


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )


    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')  # for 1024*512
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
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
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths_atten=[1, 1, 1, 1, 1, 1, 1, 1, 1], depths=[1, 1, 1, 1, 2, 1, 1, 1, 1],
                 num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, device='cuda:0', **kwargs):
        super(SphereTrans, self).__init__()
        self.num_enc_layers = len(depths) // 2  # 四层encoder
        self.num_dec_layers = len(depths) // 2  # 四层decoder
        self.embed_dim = embed_dim  # 初始embedding的维度 32
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.ref_point256x512 = genSpherePoint(img_size, img_size*2, 3, 3).to(device)  # torch.load("network6/Equioffset256x512.pth")
        self.ref_point128x256 = genSpherePoint(img_size//2, img_size, 3, 3).to(device) # torch.load("network6/Equioffset128x256.pth")
        self.ref_point64x128 = genSpherePoint(img_size//4, img_size//2, 3, 3).to(device) # torch.load("network6/Equioffset64x128.pth")
        self.ref_point32x64 = genSpherePoint(img_size//8, img_size//4, 3, 3).to(device)  ##torch.load("network6/Equioffset32x64.pth")
        self.ref_point16x32 = genSpherePoint(img_size//16, img_size//8, 3, 3).to(device)  # torch.load("network6/Equioffset16x32.pth")

        self.size = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth  # drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，不经过主干，而直接由分支进行恒等映射。
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # 预训练特征提取
        config = _cfg(url='', file='/media/lby/lby_8t/backbone/resnet/resnet50.pth')
        self.feature_extract = timm.create_model('resnet50', pretrained=True, features_only=True, pretrained_cfg=config)

        self.con1_4 = nn.Conv2d(2048, embed_dim*16, kernel_size=1, stride=1)
        self.con1_3 = nn.Conv2d(1024, embed_dim*8, kernel_size=1, stride=1)
        self.con1_2 = nn.Conv2d(512, embed_dim*4, kernel_size=1, stride=1)
        self.con1_1 = nn.Conv2d(256, embed_dim*2, kernel_size=1, stride=1)
        self.con1_0 = nn.Conv2d(64, embed_dim, kernel_size=1, stride=1)

        self.output = OutputProj(in_channel=embed_dim * 2, out_channel=1, kernel_size=3, stride=1,          # 输出层
                                      input_resolution=(img_size, img_size * 2))

        # Bottleneck
        self.conv1 = BasicTransformerLayer(dim=embed_dim * 16,
                                          output_dim=embed_dim * 16,
                                          input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
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
                                          ref_point=self.ref_point16x32, flag=0)
        # self.conv1 = nn.Sequential(
        #     nn.Linear(512, 256, bias=False),
        #     nn.GELU(),
        #     nn.Linear(256, 64, bias=False),
        #     nn.GELU(),
        #     nn.Linear(64, 1, bias=False)
        # )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 16, embed_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 4, 1, bias=False)
        )
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
        b = x.shape[0]
        #print(x.shape)
        # Encoder
        y_list = self.feature_extract(x)   # 预训练提取
          # (embed_dim*8, 32, 64)
        y4 = self.con1_4(y_list[4]).flatten(2).transpose(1, 2).contiguous()    # (embed_dim*16, 16, 32)
        #y4 = self.con1_4(y_list[4]).flatten(1).contiguous()
        #y4 = y4.view(b*512, -1)
        # print(y4.shape)
        # Bottleneck
        out = self.conv1(y4)  # (16, 32)
        #print(out.shape)
        # out = out.reshape(b, 512, 512)
        # out = out.reshape(b, 512, self.size//16, self.size//8)
        out = out.permute(0,2,1).view(b, -1, self.size//16, self.size//8)
        #out = self.spherevit(out, self.sph_pos)
        out = self.gap(out)
        out = out.squeeze(-1).squeeze(-1)
        out = self.mlp(out)

        # Decoder
        # up0 = self.upsample_0(conv0)  # (embed_dim * 8, 32, 64)
        # deconv0 = torch.cat([up0, y3], -1)  # (embed_dim * 16, 32, 64)
        # deconv0 = self.decoderlayer_0(deconv0)  # (embed_dim * 8, 32, 64)
        #
        # up1 = self.upsample_1(deconv0)  # (embed_dim * 4, 64, 128)
        # deconv1 = torch.cat([up1, y2], -1)  # (embed_dim * 8, 64, 128)
        # deconv1 = self.decoderlayer_1(deconv1)  # (embed_dim * 4, 64, 128)
        #
        # up2 = self.upsample_2(deconv1)  # (embed_dim * 2, 128, 256)
        # deconv2 = torch.cat([up2, y1], -1)  # (embed_dim * 4, 128, 256)
        # deconv2 = self.decoderlayer_2(deconv2)  # (128, 256)
        #
        # up3 = self.upsample_3(deconv2)  # (256, 512)
        # deconv3 = torch.cat([up3, y0], -1)  # (embed_dim * 2, 256, 512)
        # deconv3 = self.decoderlayer_3(deconv3)  # (256, 512 )

        # Output Projection

        return out


if __name__ == '__main__':
    # a = genSpherePoint(128, 256, 3, 3).cuda()
    # print(a.shape)

    model = SphereTrans(device="cuda:0")
    params_conv1 = sum(p.numel() for p in model.conv1.parameters())
    print("Conv1 层的参数量：", params_conv1)

