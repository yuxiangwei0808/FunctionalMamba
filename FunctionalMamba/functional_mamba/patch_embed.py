import torch
import torch.nn as nn
from einops import rearrange

from timm.layers import PatchEmbed


class PatchEmbedConnV0(nn.Module):
    # patch embedding designed for functional connectivity
    def __init__(self, num_node, patch_size, in_chans, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dim, (1, patch_size), (1, patch_size), bias=bias)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, (patch_size, 1), (patch_size, 1), bias=bias)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.num_patches = (num_node // patch_size) ** 2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_layer(x)
        return x

class PatchEmbedConnV1(nn.Module):
    # patch embedding designed for functional connectivity
    def __init__(self, num_node, patch_size, in_chans, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dim, (1, patch_size), (1, patch_size), bias=bias)
        self.conv2 = nn.Conv2d(in_chans, embed_dim, (patch_size, 1), (patch_size, 1), bias=bias)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.num_patches = (num_node // patch_size) ** 2
    
    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.conv2(x)
        x = y1 @ y0

        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_layer(x)
        return x

class PatchEmbedConnV2(nn.Module):
    # patch embedding designed for functional connectivity
    def __init__(self, num_node, patch_size, in_chans, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dim, (1, patch_size), (1, patch_size), bias=bias)
        self.conv2 = nn.Conv2d(in_chans, embed_dim, (patch_size, 1), (patch_size, 1), bias=bias)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, (patch_size, 1), (patch_size, 1), bias=bias)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.num_patches = (num_node // patch_size) ** 2
    
    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.conv2(x)

        attn = torch.sigmoid(y1 @ y0)
        x = self.conv3(y0)
        x *= attn

        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_layer(x)
        return x

class PatchEmbedSingleDim(nn.Module):
    # patch embedding that only performs on one dimension
    def __init__(self, h, w, patch_size, in_chans, embed_dim, row_wise, norm_layer=None, bias=True):
        super().__init__()
        if row_wise:
            self.conv = nn.Conv2d(in_chans, embed_dim, (patch_size, 1), (patch_size, 1), bias=bias)
            self.num_patches = (h // patch_size) * w
        else:
            self.conv = nn.Conv2d(in_chans, embed_dim, (1, patch_size), (1, patch_size), bias=bias)
            self.num_patches = h * (w // patch_size)

        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_layer(x)  # LN
        return x


class PatchEmbedSpatioTemporal(nn.Module):
    def __init__(self, num_node, num_frame, patch_size, patch_size_t, in_chans, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        
        pz = (patch_size, patch_size, patch_size_t)
        self.conv = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=pz, stride=pz, bias=bias)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = (num_node // patch_size) ** 2 * (num_frame // patch_size_t)

    def forward(self, x):
        # x: B 1 H W T
        x = self.conv(x)
        x = self.norm_layer(x)  # BN
        return x.permute(0, 2, 3, 4, 1)  # B H W T C


class TimeMerging(nn.Module):
    def __init__(self, dim, patch_size, out_dim=None, bias=True, norm_layer=None, version='v2'):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(patch_size * dim) if norm_layer else nn.Identity()
        self.reduction = nn.Linear(patch_size * dim, self.out_dim, bias=bias)
        self.version = version

    def forward(self, x):
        B, H, W, T, C = x.shape
        if T % self.patch_size != 0:
            T = T // self.patch_size * self.patch_size
            x = x[:, :, :, :T, ...]
        if self.version == 'v1':
            x = x.reshape(B, H, W, T // self.patch_size, self.patch_size, C).flatten(-2)
        else:
            x = rearrange(x, 'b h w (p t) c -> b h w t (p c)', p=self.patch_size)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ComponentMerging(nn.Module):
    def __init__(self, dim, patch_size, out_dim=None, bias=True, norm_layer=None, version='v2'):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.out_dim = out_dim or 3 * dim
        self.norm = norm_layer((patch_size ** 2) * dim) if norm_layer else nn.Identity()
        self.reduction = nn.Linear((patch_size ** 2) * dim, self.out_dim, bias=bias)
        self.version = version

    def forward(self, x):
        B, H, W, T, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        if self.version == 'v1':
            x = x.reshape(B, H // self.patch_size, W // self.patch_size, T, self.patch_size, self.patch_size, C).flatten(-3)
        else:
            x = rearrange(x, 'b (p1 h) (p2 w) t c -> b h w t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbedSpatioTemporalV1(nn.Module):
    # patch merge time
    def __init__(self, num_node, num_frame, patch_size, patch_size_t, in_chans, embed_dim, norm_layer=nn.LayerNorm, bias=True):
        super().__init__()
        
        pz = (patch_size, patch_size)
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=pz, stride=pz, bias=bias)
        self.time_merge = TimeMerging(in_chans, patch_size_t, in_chans)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = (num_node // patch_size) ** 2 * (num_frame // patch_size_t)

    def forward(self, x):
        # x: B C H W T
        x = self.time_merge(x.permute(0, 2, 3, 4, 1))  # B H W T//2 C
        B, H, W, T, C = x.shape
        x = self.conv(rearrange(x, 'b h w t c -> (b t) c h w'))
        return self.norm_layer(rearrange(x, '(b t) c h w -> b h w t c', b=B, t=T))


class PatchEmbedSpatioTemporalV2(nn.Module):
    # patch merge time
    def __init__(self, num_node, num_frame, patch_size, patch_size_t, in_chans, embed_dim, time_merge=False, norm_layer=nn.LayerNorm, bias=True):
        super().__init__()
        
        pz = (patch_size, patch_size)
        self.component_merge = ComponentMerging(in_chans, patch_size, embed_dim, bias=bias)
        self.time_merge = nn.Conv1d(in_chans, in_chans, kernel_size=patch_size_t, stride=patch_size_t, bias=bias) if not time_merge \
            else TimeMerging(in_chans, patch_size_t, in_chans, bias=bias)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = (num_node // patch_size) ** 2 * (num_frame // patch_size_t)

    def forward(self, x):
        # x: B C H W T
        if isinstance(self.time_merge, nn.Conv1d):
            _, _, H, W, _ = x.shape
            x = self.time_merge(rearrange(x, 'b c h w t -> (b h w) c t'))
            x = rearrange(x, '(b h w) c t -> b h w t c', h=H, w=W)
        else:
            x = self.time_merge(x.permute(0, 2, 3, 4, 1))  # B H W T//2 C
        B, H, W, T, C = x.shape
        x = self.component_merge(x)
        return self.norm_layer(x)
