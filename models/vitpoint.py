import math
from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class HeadConfig:
    dim_descriptor: int = 256


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
            num_classes=0,  # remove classifier
        )
        # 2. heads
        hidden_dim = self.backbone.num_features  # 384
        self.det_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.desc_head = nn.Conv2d(hidden_dim, head_cfg.dim_descriptor, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, return_token=False):
        """
        Input : x [B,3,H,W]  (must be divisible by patch size).
        Output: heatmap [B,1,h,w], descriptor [B,D,h,w]
        """
        B = x.shape[0]
        tokens = self.backbone.forward_features(x)  # [B, N+1, C]
        # remove CLS
        tokens = tokens[:, 1:]                      # [B, N, C]
        h = w = int(math.sqrt(tokens.size(1)))
        feat = rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)  # [B,C,h,w]

        heat = torch.sigmoid(self.det_head(feat))
        desc = F.normalize(self.desc_head(feat), p=2, dim=1)
        if return_token:
            return heat, desc, feat
        return heat, desc
