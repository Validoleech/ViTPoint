import torch
import torch.nn.functional as F


def focal_bce(logits, gt, alpha=0.25, gamma=2):
    """
    Focal loss on raw logits, AMP-safe.
    """
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
    prob = torch.sigmoid(logits)
    pt = prob*gt + (1-prob)*(1-gt)
    loss = alpha * (1-pt)**gamma * bce
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


@torch.no_grad()
def compute_val_loss(model, loader, loss_fn, device="cuda"):
    """
    loss_fn must be a callable that takes (model, batch, device) and returns
    a scalar tensor. We accumulate it over the validation set.
    """
    model.eval()
    running = 0.
    for batch in loader:
        running += loss_fn(model, batch, device).item()
    return running / len(loader)


def l1_offset(offset_pr, offset_gt, heat_gt):
    """
    offset_pr / offset_gt : [B,2,h,w]   Δy,Δx in (-0.5,0.5)
    heat_gt               : [B,1,h,w]   binary mask of positives
    """
    mask = heat_gt.to(dtype=torch.bool, device=offset_pr.device) \
                  .expand_as(offset_pr)      # [B,2,h,w]

    if not mask.any():
        return offset_pr.new_zeros(())

    # if mask.sum() == 0:
    #     return torch.tensor(0., device=offset_pr.device)
    return F.l1_loss(offset_pr[mask], 
                     offset_gt.to(offset_pr.device)[mask],
                     reduction='mean')

def soft_nms_penalty(heat, k=3):
    """
    Soft-NMS: penalise secondary peaks around each maximum.
    heat : sigmoid output [B,1,h,w]
    """
    maxpool = F.max_pool2d(heat, kernel_size=k, stride=1, padding=k//2)
    # positive values only where heat < neighbourhood max
    return (maxpool - heat).relu().mean()


@torch.no_grad()
def nms_simple(heat: torch.Tensor,
               thresh: float = 0.5,
               radius: int = 4) -> torch.Tensor:
    """
    Non-Maximum Suppression for dense corner / key-point heat-maps.

    Args
    ----
    heat   : tensor  [..., H, W] – probability map in [0,1].
    thresh : float   – minima score to consider a candidate.
    radius : int     – NMS radius in pixels (3 ⇒ 7×7 window).

    Returns
    -------
    keep : bool tensor, same spatial size as `heat`, True = keep pixel.
    """
    # normalise to shape  [B,1,H,W]
    if heat.dim() == 2:                       # [H,W]
        heat = heat.unsqueeze(0).unsqueeze(0)
    elif heat.dim() == 3:                     # [B,H,W]
        heat = heat.unsqueeze(1)
    assert heat.dim() == 4, "NMS expects 4-D tensor"

    heat_f = heat.float()
    
    mask = heat_f >= thresh
    cand = heat_f * mask

    # local max with max-pool
    k = 2 * radius + 1
    pad = radius
    pooled = F.max_pool2d(cand, kernel_size=k, stride=1, padding=radius)

    # 3. a pixel is a key-point if
    # 1) it survived the threshold &
    # 2) it equals the local max
    keep = (cand == pooled) & mask

    # squeeze back to original dims
    keep = keep.squeeze(1)              # [B,1,H,W] → [B,H,W]
    return keep.bool()
