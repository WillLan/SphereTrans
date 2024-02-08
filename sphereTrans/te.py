import torch
import torch.nn as nn
class Upsample(nn.Module):
    def __init__(self, in_channel=256, out_channel=256, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),    # 反卷积操作
        )

    def forward(self, x):
        # B, L, C = x.shape
        # H, W = self.input_resolution
        # x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x)
        print("up", out.shape)
        #flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

pic = torch.randn(1,256,8,16)
model = Upsample()
out = model(pic)
print(out.shape)