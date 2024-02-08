import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention_Sphere(nn.Module):
    def __init__(self, dim, heads = 1, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.v_proj = nn.Linear(dim_head, 1, bias = False)
        self.point_proj = nn.Linear(dim_head, 1, bias = False)
    def forward(self, x, sphere_point):
        B, C, H_, W_ = x.shape
        H, W = int(H_/3), int(W_/3)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H_*W_, C)
        x = self.norm(x) # (B, 3H*3W, C)

        qkv = self.to_qkv(x).chunk(3, dim = -1) # 3 (B, 3H*3W, C)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # (B, h, 3H*3W, d_head)

        q_ = q.view(B, self.heads, H, 3, W, 3, self.dim_head).permute(0, 1, 2, 4, 3, 5, 6).contiguous() # (B, h, H, W, 3, 3, d_head)
        q_ = q_.view(-1, 9, self.dim_head) # (B*h*H*W, 9, d_head)
        k_ = k.view(B, self.heads, H, 3, W, 3, self.dim_head).permute(0, 1, 2, 4, 3, 5, 6).contiguous() # (B, h, H, W, 3, 3, d_head)
        k_ = k_.view(-1, 9, self.dim_head) # (B*h*H*W, 9, d_head)
        v_ = v.view(B, self.heads, H, 3, W, 3, self.dim_head).permute(0, 1, 2, 4, 3, 5, 6).contiguous() # (B, h, H, W, 3, 3, d_head)
        v_ = v_.view(-1, 9, self.dim_head) # (B*h*H*W, 9, d_head)
        v_project = self.v_proj(v_) # (B*h*H*W, 9, 1)

        sphere_point = sphere_point.view(B, self.heads, self.dim_head, H, 3, W, 3)
        sphere_point = sphere_point.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        sphere_point = sphere_point.view(-1, 9, self.dim_head)
        sph_pot_proj = self.point_proj(sphere_point) # (B*h*H*W, 9, 1)

        feat_score = v_project + sph_pot_proj  # 需要归一化吗
        feat_score = feat_score.squeeze(-1)
        feat_score = (feat_score - torch.mean(feat_score, dim=1, keepdim=True))/ torch.std(feat_score, dim=1, keepdim=True)
        feat_score = F.softmax(feat_score, dim=-1) # (B*h*H*W, 9)

        # q_ = self.norm(q_)
        # k_ = self.norm(k_)
        # v_ = self.norm(v_)

        # (B*h*H*W, 9, 9)
        dots = torch.matmul(q_, k_.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # (B*h*H*W, 9, d_head)
        out = torch.matmul(attn, v_)
        out = torch.matmul(out.transpose(-1, -2), feat_score.unsqueeze(-1)) # (B*h*H*W, d_head, 1)
        out = out.squeeze(-1) # (B*h*H*W, d_head)
        out1 = out.view(B, self.heads, H, W, self.dim_head).permute(0, 2, 3, 1, 4).contiguous()
        out1 = out1.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # B, C, H, W

        return out1

class AttentionBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Attention_Sphere(dim, heads = heads, dim_head = dim_head, dropout = dropout)
            )

    def forward(self, x, sphere_point):
        for attn in self.layers:
            x = attn(x, sphere_point)
        return x

class GlobalSphereAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, sph_dist):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        b, h, n, d = q.shape

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots_weighted = dots * (1 + sph_dist.unsqueeze(0).unsqueeze(1).repeat(b, h, 1, 1))
        dots_weighted = (dots_weighted - torch.mean(dots_weighted, dim=1, keepdim=True))/ torch.std(dots_weighted, dim=1, keepdim=True)

        attn = self.attend(dots_weighted)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                GlobalSphereAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
            )

    def forward(self, x, sph_dist):
        for attn in self.layers:
            x = attn(x, sph_dist) + x

        return self.norm(x)

class SphereViT(nn.Module):
    def __init__(self, patch_size, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        patch_height, patch_width = pair(patch_size)

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Conv2d(channels, dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))
        self.norm_linear = nn.Sequential(
            #nn.Conv2d(patch_dim, patch_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width)),  # (b, 768, 16, 16)
            #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), #  remove ?????????????????????
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Linear(2, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout)
        self.expnddim = nn.Linear(dim, patch_dim)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h =8, w=16, p1 = patch_height, p2 = patch_width)

    def forward(self, img, sph_pos_dis):

        x = self.to_patch_embedding(img)
        bz, channel, num_ph, num_pw = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(bz, num_ph*num_pw, channel)
        x = self.norm_linear(x)
        #print(sph_pos_dis)
        b, n, _ = x.shape
        sph_pos, sph_dist = sph_pos_dis[0], sph_pos_dis[1]  # (H/ph, W/pw, 2), (H/ph*W/pw, H/ph*W/pw)
        sph_pos = self.pos_embedding(sph_pos.unsqueeze(0).repeat(b, 1, 1, 1)) #   (b, H/ph, W/pw, 1024)
        sph_pos = sph_pos.view(b, n, -1)

        x += sph_pos
        x = self.dropout(x)

        x = self.transformer(x, sph_dist)
        x = self.rearrange(self.norm1(self.expnddim(x))) # b,c,h,w

        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)