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
        self.backbone.set_grad_checkpointing(enable=True)
        # 2. heads
        hidden_dim = self.backbone.num_features  # 384
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self._feat_mid = None  # grab tokens from an earlier block
        self.backbone.blocks[4].register_forward_hook(lambda _m, _in, out: setattr(self, '_feat_mid', out)) # CLS + tokens
        self.det_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
        self.desc_head = nn.Conv2d(hidden_dim, head_cfg.dim_descriptor, 1)

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
        heat = torch.sigmoid(self.det_head(feat))          # [B,1,h,w]
        offset = torch.tanh(self.offset_head(feat)) * 0.5  # [B,2,h,w]
        desc = F.normalize(self.desc_head(feat), p=2, dim=1)
        if return_token:
            return heat, offset, desc, feat
        return heat, offset, desc
