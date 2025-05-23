# SPDX-License-Identifier: MIT
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------------------------------------------
# STAGE-1  ────────────────────────────────────────────────────
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_detector_pr(model, loader, thr=0.5, device="cuda"):
    model.eval()
    tp = fp = fn = 0
    scores, labels = [], []
    for img, heat_gt in loader:
        img, heat_gt = img.to(device), heat_gt.to(device)
        heat_pr, _ = model(img)
        heat_pr = heat_pr.squeeze(1)           # [B,h,w]
        heat_gt = heat_gt.squeeze(1).bool()    # [B,h,w]

        # accumulate PR counts
        pr_mask = heat_pr > thr
        tp += (pr_mask & heat_gt).sum().item()
        fp += (pr_mask & ~heat_gt).sum().item()
        fn += (~pr_mask & heat_gt).sum().item()

        # for AP: store every pixel prob+label
        scores.append(heat_pr.flatten().cpu())
        labels.append(heat_gt.float().flatten().cpu())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2*prec*rec / (prec + rec + 1e-8)

    # AP via sklearn-like manual integration
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    sorted_idx = scores.argsort(descending=True)
    sorted_lbl = labels[sorted_idx]
    tp_cum = torch.cumsum(sorted_lbl, 0)
    fp_cum = torch.cumsum(1-sorted_lbl, 0)
    rec_curve = tp_cum / (labels.sum() + 1e-8)
    prec_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = torch.trapz(prec_curve, rec_curve).item()

    return {"P": prec, "R": rec, "F1": f1, "AP": ap}



def nms_heatmap(heat, topk=1000, dist=4):
    """very small NMS on heatmaps, returns list of (y,x) in pixel coords."""
    B, _, H, W = heat.shape
    heat_nms = F.max_pool2d(heat, 2*dist+1, stride=1, padding=dist)
    mask = (heat == heat_nms) & (heat > 0.5)
    ys, xs = torch.where(mask.squeeze(1))
    kpts = [[] for _ in range(B)]
    for y, x in zip(ys.cpu(), xs.cpu()):
        b = y // H
        if len(kpts[b]) < topk:
            kpts[b].append((int(y % H), int(x)))
    return kpts


@torch.no_grad()
def evaluate_repeatability(model, loader, thr=3, dist=4, device="cuda"):
    """
    loader should yield (img1, img2, H12), where H12 maps img1->img2 (pixel coord).
    HPatches provides that directly.
    """
    model.eval()
    rep_list = []
    for I1, I2, H12 in tqdm(loader, leave=False):
        I1, I2, H12 = I1.to(device), I2.to(device), H12.to(device)
        B, _, H, W = I1.shape

        # ---- forward
        h1, _ = model(I1)  # [B,1,h,w]
        h2, _ = model(I2)

        # ---- keypoints in *pixel* coords (original image)
        k1 = nms_heatmap(h1, dist=dist)
        k2 = nms_heatmap(h2, dist=dist)

        for b in range(B):
            if len(k1[b]) == 0 or len(k2[b]) == 0:
                rep_list.append(0.0)
                continue

            # warp k1→I2
            pts1 = torch.tensor(k1[b], dtype=torch.float,
                                device=device)  # (N,2)  (y,x)
            pts1_h = torch.cat(
                # (x,y,1)
                [pts1[:, 1:], pts1[:, :1], torch.ones_like(pts1[:, :1])], 1)
            pts2_h = (H12[b] @ pts1_h.t()).t()
            pts2_h = pts2_h[:, :2] / pts2_h[:, 2:].clamp(min=1e-7)

            # distance to all k2
            pts2 = torch.tensor(k2[b], dtype=torch.float, device=device)  # M,2
            dists = torch.cdist(pts2_h, pts2)  # N×M
            ok1 = (dists.min(1).values < thr).cpu()
            ok2 = (dists.min(0).values < thr).cpu()
            rep = 0.5*(ok1.sum()+ok2.sum()) / max(len(k1[b]), len(k2[b]))
            rep_list.append(rep.item())
    return {"repeatability": np.mean(rep_list)}
