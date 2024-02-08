import torch
import torchvision
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
import math
from network.sphereNet import SphereConv2D

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# deformable patch embedding
class SpherePatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(256, 512), patch_size=16, stride=2, in_chans=3, embed_dim=256, use_dcn=True, out_c = 64):    # stride=2 for downsample
        super(SpherePatchEmbed, self).__init__()

        patch_size=pair(patch_size)
        self.patch_size = patch_size
        self.img_size = img_size
        self.kernel_size = self.patch_size[0]
        self.padding = self.patch_size[0]// 2
        self.norm = nn.LayerNorm(embed_dim)
        self.stride = pair(stride)
        self.use_dcn = use_dcn
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.spherenet = SphereConv2D(in_c=in_chans, out_c=in_chans, stride=1)
        self.apply(self._init_weights)

        if use_dcn:
            self.offset_conv = nn.Conv2d(in_chans, 2 * self.kernel_size * self.kernel_size, kernel_size=self.kernel_size, stride=stride, padding=self.padding)
            nn.init.constant_(self.offset_conv.weight, 0.)
            nn.init.constant_(self.offset_conv.bias, 0.)

            self.modulator_conv = nn.Conv2d(in_chans, 1 * patch_size[0] * patch_size[0], kernel_size=self.kernel_size, stride=stride, padding=self.padding)
            nn.init.constant_(self.modulator_conv.weight, 0.)
            nn.init.constant_(self.modulator_conv.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.use_dcn:
            x = self.spherenet(x)
            x = self.deform_proj(x)
        else:
            x = self.spherenet(x)
            x = self.proj(x)
        _, _, H, W = x.shape
        #print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

if __name__ == "__main__":
    pic = torch.randn(2,3,256,512)
    model = SpherePatchEmbed()
    out,_, _ = model(pic)
    print(out.shape)