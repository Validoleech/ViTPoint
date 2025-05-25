import torch
import numpy as np
import torch.nn.functional as F
import kornia as K
from tqdm import tqdm
from utils.loss import focal_bce, info_nce, _nms
from torch.amp import autocast

def sweep_best_thr(model, val_loader, device, n_steps=14, verbose=False):
    """
    Returns the threshold that maximises F1 on the whole validation set.
    """
    ts, Ps, Rs = [], [], []
    for t in np.linspace(0.30, 0.95, n_steps):
        m = evaluate_detector_pr(
            model, val_loader, device=device, thr=float(t))
        ts.append(t)
        Ps.append(m['P'])
        Rs.append(m['R'])
        if verbose:
            print(f"Evaluated model with threshold {t}")
    Ps = np.array(Ps)
    Rs = np.array(Rs)
    F1s = 2 * Ps * Rs / (Ps + Rs + 1e-9)
    best = np.argmax(F1s)
    return ts[best], F1s[best], Ps[best], Rs[best]

def grid_to_pixel(y, x, dy, dx, patch):
    """
    y , x   : integer location on h×w detector grid
    dy, dx  : sub-pixel offsets in (−0.5, 0.5)
    patch   : ViT patch size in pixels (e.g. 14)
    returns : (y_px , x_px) in original image coordinates
    """
    return (y + 0.5 + dy) * patch, (x + 0.5 + dx) * patch

@torch.no_grad()
def evaluate_detector_pr(model, loader, thr=0.5, device="cuda"):
    """
    Works at the heat-map resolution; offset channel is ignored.
    """
    model.eval()
    tp = fp = fn = 0
    scores, labels = [], []

    for batch in loader:
        if len(batch) == 2:
            img, heat_gt = batch
        else:
            img, heat_gt, *_ = batch

        img = img.to(device)
        heat_gt = heat_gt.to(device)

        heat_pr, _, _ = model(img)
        heat_pr = heat_pr.squeeze(1)           # [B,h,w]
        heat_gt = heat_gt.squeeze(1).bool()    # [B,h,w]

        # PR counts
        pr_mask = heat_pr > thr
        tp += ( pr_mask &  heat_gt).sum().item()
        fp += ( pr_mask & ~heat_gt).sum().item()
        fn += (~pr_mask &  heat_gt).sum().item()

        # AP curve
        scores.append(heat_pr.flatten().cpu())
        labels.append(heat_gt.float().flatten().cpu())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2*prec*rec / (prec + rec + 1e-8)

    # Average Precision (numerical integration)
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    sorted_idx = scores.argsort(descending=True)
    sorted_lbl = labels[sorted_idx]
    tp_cum = torch.cumsum(sorted_lbl, 0)
    fp_cum = torch.cumsum(1 - sorted_lbl, 0)
    rec_curve = tp_cum / (labels.sum() + 1e-8)
    prec_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = torch.trapz(prec_curve, rec_curve).item()

    return {"P": prec, "R": rec, "F1": f1, "AP": ap}


def nms_heatmap(heat, offset=None, patch=14, topk=1000,
                dist=4, thr=0.5):
    """
    Very small NMS on detector grid.
    If `offset` is given, returned keypoints are pixel coords
    refined by Δoffset.
    Returns: list[batch] of (y_px , x_px) tuples
    """
    B, _, H, W = heat.shape
    heat_nms = F.max_pool2d(heat, kernel_size=2*dist+1,
                            stride=1, padding=dist)
    mask = (heat == heat_nms) & (heat > thr)
    ys, xs = torch.where(mask.squeeze(1))      # indices in grid space

    kpts = [[] for _ in range(B)]
    for y, x in zip(ys.cpu(), xs.cpu()):
        b = y // H
        if len(kpts[b]) >= topk:
            continue

        if offset is not None:
            dy = offset[b, 0, y % H, x].cpu().item()
            dx = offset[b, 1, y % H, x].cpu().item()
            y_px, x_px = grid_to_pixel(y % H, x, dy, dx, patch)
            kpts[b].append((float(y_px), float(x_px)))
        else:
            # coarse centre of the patch
            y_px = (y % H + 0.5) * patch
            x_px = (x + 0.5) * patch
            kpts[b].append((float(y_px), float(x_px)))
    return kpts


def evaluate_repeatability(model, loader, thr=3, dist=4, device="cuda"):
    """
    loader yields (img1 , img2 , H12) where H12 maps img1 → img2
    in pixel coordinates. HPatches supplies this directly.
    """
    model.eval()
    rep_list = []
    patch = model.backbone.patch_embed.patch_size[0] # 14

    for I1, I2, H12 in tqdm(loader, leave=False):
        I1, I2, H12 = I1.to(device), I2.to(device), H12.to(device)
        B, _, H, W = I1.shape

        # forward
        h1, off1, _ = model(I1)
        h2, off2, _ = model(I2)

        # grid → pixel kpts
        k1 = nms_heatmap(h1, off1, patch=patch, dist=dist)
        k2 = nms_heatmap(h2, off2, patch=patch, dist=dist)

        for b in range(B):
            if len(k1[b]) == 0 or len(k2[b]) == 0:
                rep_list.append(0.0)
                continue

            # warp k1 → I2
            pts1 = torch.tensor(k1[b], dtype=torch.float,
                                device=device)     # (N,2)  (y,x)
            pts1_h = torch.cat([pts1[:, 1:], pts1[:, :1],
                                torch.ones_like(pts1[:, :1])], dim=1)   # (N,3)
            pts2_h = (H12[b] @ pts1_h.t()).t()
            pts2_h = pts2_h[:, :2] / pts2_h[:, 2:].clamp(min=1e-6)

            pts2 = torch.tensor(k2[b], dtype=torch.float,
                                device=device)      # (M,2)
            dists = torch.cdist(pts2_h, pts2)       # N×M
            ok1 = (dists.min(1).values < thr).cpu()
            ok2 = (dists.min(0).values < thr).cpu()
            rep = 0.5 * (ok1.sum()+ok2.sum()) / max(len(k1[b]), len(k2[b]))
            rep_list.append(rep.item())

    return {"repeatability": np.mean(rep_list)}


def val_step(model, batch, device, amp_dtype=torch.float16):
    img, heat, _ = batch
    img = img.to(device, non_blocking=True)
    heat = heat.to(device, non_blocking=True)
    with autocast(device_type=device.type, dtype=amp_dtype):
        heat_pr, _, _ = model(img)
        loss = focal_bce(heat_pr, heat)

    return loss


def val_step_stage2(model, batch, device, amp_dtype=torch.bfloat16):
    imgs, H_mat, _ = batch
    imgs = imgs.to(device, non_blocking=True)
    H_mat = H_mat.to(device, non_blocking=True)

    # forward original + warped
    heat1, _, desc1 = model(imgs)
    imgs_warped = K.geometry.warp_perspective(imgs, H_mat,
                                              dsize=imgs.shape[-2:])
    heat2, _, desc2 = model(imgs_warped)

    # detector consistency
    heat1_w = K.geometry.warp_perspective(heat1, H_mat,
                                          dsize=heat1.shape[-2:])
    L_det = focal_bce(heat1_w, heat2) + focal_bce(heat2, heat1_w)

    # descriptor InfoNCE
    desc1_w = K.geometry.warp_perspective(desc1, H_mat,
                                          dsize=desc1.shape[-2:])
    L_desc = info_nce(desc1_w, desc2)

    return L_det + 0.1 * L_desc


def pr_single_batch(heat_pr, heat_gt, thresh=0.5, topk=256):
    # NMS + threshold
    det_xy = _nms(heat_pr, thresh, topk=topk)          # (N,2)
    det_mask = torch.zeros_like(heat_gt.squeeze().bool())
    if det_xy.numel():
        ys, xs = det_xy.long().unbind(1)
        det_mask[ys, xs] = True
    gt_mask = (heat_gt.squeeze() > 0.5)
    TP = (det_mask & gt_mask).sum().item()
    FP = (det_mask & ~gt_mask).sum().item()
    FN = (~det_mask & gt_mask).sum().item()
    P = TP / (TP + FP + 1e-6)
    R = TP / (TP + FN + 1e-6)
    return P, R



