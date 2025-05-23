import torch
import torch.nn.functional as F


def focal_bce(pred, gt, alpha=0.25, gamma=2):
    """Binary focal loss."""
    bce = F.binary_cross_entropy(pred, gt, reduction='none')
    pt = gt*pred + (1-gt)*(1-pred)
    loss = alpha*(1-pt)**gamma * bce
    return loss.mean()


def info_nce(desc1, desc2, temp=0.07):
    """
    desc1, desc2: [B,C,H,W]  positive pairs are same spatial location after warp.
    """
    B, C, H, W = desc1.shape
    # flatten
    d1 = desc1.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
    d2 = desc2.permute(0, 2, 3, 1).reshape(-1, C)
    logits = d1 @ d2.t() / temp                 # [N,N]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
