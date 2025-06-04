import numpy
from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class HeadConfig:
    dim_descriptor: int = 256
    subgrid:        int = 1

    @property
    def n_cells(self) -> int:
        return self.subgrid ** 2


class SuperPointViT(nn.Module):
    """
    DINOv2-Small (ViT-S/14) backbone + 1×1 detector head + descriptor head.
    """
    def __init__(self, head_cfg: HeadConfig, freeze_backbone: bool = False):
        super().__init__()
        # backbone
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=(224, 224),
            num_classes=0,  # remove classifier
        )
        self.backbone.set_grad_checkpointing(enable=False)
        # heads
        self.subgrid = head_cfg.subgrid
        self.n_cells = head_cfg.n_cells

        hidden_dim = self.backbone.num_features  # 384
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self._feat_mid = None  # grab tokens from an earlier block
        self.backbone.blocks[4].register_forward_hook(lambda _m, _in, out: setattr(self, '_feat_mid', out)) # CLS + tokens
        self.det_head = nn.Conv2d(hidden_dim, self.n_cells, kernel_size=1)
        self.offset_head = nn.Conv2d(hidden_dim, 2 * self.n_cells, kernel_size=1)
        self.desc_head = nn.Conv2d(hidden_dim, head_cfg.dim_descriptor, 1)

        anchor = []
        if self.subgrid == 1:
            anchor = [[0.0, 0.0]]
        else:
            step = 1.0 / self.subgrid # sub-path width
            for r in range(self.subgrid):
                for c in range(self.subgrid):
                    ay = (r + 0.5) * step - 0.5
                    ax = (c + 0.5) * step - 0.5
                    anchor.append([ay, ax])
        self.register_buffer(
            "anchor_offsets",
            torch.tensor(anchor).view(1, self.n_cells, 2, 1, 1)) # [1,M,2,1,1]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, return_token=False):
        """
        x        : [B,3,H,W]  (H,W divisible by patch size)
        returns
            heat   : [B,1,h,w]      probability map
            offset : [B,2,h,w]      Δy,Δx  ∈ (-0.5,0.5)  (patch-normalised)
            desc   : [B,D,h,w]      L2-normalised descriptors
        """
        B = x.shape[0]
        tokens_hi = self.backbone.forward_features(x)      # [B, N+1, C]
        tokens_mid = self._feat_mid                        # hook
        assert tokens_mid is not None, "Forward hook did not fire"
        gH, gW = self.backbone.patch_embed.grid_size
        # remove CLS
        hi = rearrange(tokens_hi[:, 1:],  'b (h w) c -> b c h w', h=gH, w=gW)
        lo = rearrange(tokens_mid[:, 1:], 'b (h w) c -> b c h w', h=gH, w=gW)
        feat = self.fuse_conv(torch.cat([hi, lo], dim=1))  # fuse early+late
        heat_multi = torch.sigmoid(self.det_head(feat)) # [B,M,h,w]
        off_multi = torch.tanh(self.offset_head(feat)) # [B,2M,h,w]
        off_multi = off_multi.view(B, self.n_cells, 2, gH, gW) # [B,M,2,h,w]
        # scale refinement to the size of the sub-cell (Δ ∈ ±0.5/subgrid)
        off_multi = off_multi * (0.5 / self.subgrid)
        # add anchor so final offsets are again in (-0.5 … +0.5)
        off_multi = off_multi + self.anchor_offsets # [B,M,2,h,w]
        
        if self.n_cells == 1:
            heat = heat_multi
            offset = off_multi.squeeze(1) # [B,2,h,w]
        else:
            heat, idx  = heat_multi.max(dim=1, keepdim=True) # idx: [B,1,h,w]
            idx_exp = idx.unsqueeze(2).expand(-1, -1, 2, -1, -1)
            offset = torch.gather(off_multi, 1, idx_exp).squeeze(1)

        desc = F.normalize(self.desc_head(feat), p=2, dim=1)
        if return_token:
            return heat, offset, desc, feat
        return heat, offset, desc

    @torch.no_grad()
    def sample_desc(self, pts_xy, desc_map, patch_size=14):
        """
        Bilinear-sample descriptors at sub-pixel coordinates.

        pts_xy   : [N, 2] pixel coordinates in full-resolution image space
        desc_map : [1,D,h,w]  output 'desc' from forward()
        returns  : [N, D]  L2-normalised descriptors

        # normalise to [-1,1] grid coords expected by grid_sample
        """
        D, h, w = desc_map.shape[1:]
        xs = pts_xy[:, 0] / (w * patch_size - 1) * 2 - 1
        ys = pts_xy[:, 1] / (h * patch_size - 1) * 2 - 1
        grid = torch.stack([xs, ys], dim=-1).view(1, 1, -1, 2)     # [1,1,N,2]
        desc = F.grid_sample(desc_map, grid, align_corners=True)   # [1,D,1,N]
        return desc.squeeze().t()                                  # [N, D]
