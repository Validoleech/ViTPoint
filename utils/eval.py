from torch.cuda.amp import autocast
import torch
import numpy as np
import torch.nn.functional as F
import kornia as K
from tqdm import tqdm
from utils.loss import focal_bce, info_nce, _nms
from utils.geometry import random_warp
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
    assert heat_pr.shape[0] == heat_gt.shape[0] == 1, \
        "Tensors of multiple images passed"
    heat_pr = heat_pr[0]          # [1,h,w] or [h,w]
    heat_gt = heat_gt[0]          # [1,h,w] or [h,w]
    det_xy = _nms(heat_pr, thresh, topk)
    det_mask = torch.zeros_like(heat_gt.squeeze().bool())  # [h,w]
    if det_xy.numel():
        xs, ys = det_xy.long().unbind(1)
        det_mask[ys, xs] = True

    gt_mask = (heat_gt.squeeze() > 0.5)

    TP = (det_mask & gt_mask).sum().item()
    FP = (det_mask & ~gt_mask).sum().item()
    FN = (~det_mask & gt_mask).sum().item()

    P = TP / (TP + FP + 1e-6)
    R = TP / (TP + FN + 1e-6)
    return P, R

def log_grad_norm(model, writer, tag, step):
    norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            norm += p.grad.norm().item()
    writer.add_scalar(tag, norm, step)


def calc_ap(model,
                      loader,
                      device,
                      *,
                      max_batches: int = 200,
                      topk: int = 300,
                      dist_thr: float = 3.0,
                      heat_thr: float = 0.0):

    def peaks_plus_offset(hm, offs, K=topk, thr=heat_thr):
        """
        hm   : (B,1,H,W)  logits
        offs : (B,2,H,W)
        Return: tuple(list[B], list[B])  -- coords [Ni,2] and scores [Ni]
        """
        B, _, H, W = hm.shape
        flat = hm.view(B, -1)                       # (B,H*W)
        if K > 0:
            scores, idx = torch.topk(flat, K, dim=1)
            mask = scores > thr
            idx = idx[mask]
            scores = scores[mask]
        else:
            mask = flat > thr
            idx = mask.nonzero(as_tuple=False)      # (M,2)  [batch,row]
            scores = flat[mask]

        ys = (idx % (H*W) // W).float()
        xs = (idx % (H*W) % W).float()
        dx = offs[idx[:, 0], 0, ys.long(), xs.long()]
        dy = offs[idx[:, 0], 1, ys.long(), xs.long()]
        xs = xs + dx
        ys = ys + dy
        coords = torch.stack([xs, ys], 1)           # (N,2)

        # split by batch
        batch_ids = idx[:, 0]
        pts_per_im = [[] for _ in range(B)]
        scr_per_im = [[] for _ in range(B)]
        for b, p, s in zip(batch_ids, coords, scores):
            pts_per_im[b].append(p)
            scr_per_im[b].append(s)

        for b in range(B):
            if len(pts_per_im[b]) == 0:
                pts_per_im[b] = torch.empty(0, 2, device=hm.device)
                scr_per_im[b] = torch.empty(0,   device=hm.device)
            else:
                pts_per_im[b] = torch.stack(pts_per_im[b], 0)
                scr_per_im[b] = torch.stack(scr_per_im[b], 0)

        return pts_per_im, scr_per_im

    def repeatability_metrics(h0, off0, d0,
                              h1, off1, d1,
                              H, topk=topk,
                              dist_thr=dist_thr):
        B = h0.shape[0]
        rep, desc_ap, loc_err, n_kp = 0., 0., 0., 0.

        pts0, scr0 = peaks_plus_offset(h0, off0)
        pts1, scr1 = peaks_plus_offset(h1, off1)

        d0 = F.normalize(d0, p=2, dim=1)
        d1 = F.normalize(d1, p=2, dim=1)

        for b in range(B):
            if pts0[b].numel() == 0 or pts1[b].numel() == 0:
                continue

            desc0 = F.grid_sample(d0[b:b+1],            # [1,C,H,W]
                                  pts0[b].view(1, -1, 1, 2)      # [1,N,1,2]
                                  .flip(-1) * 2 / torch.tensor(
                                      [h0.shape[-1]-1,
                                       h0.shape[-2]-1],
                                      device=h0.device) - 1,
                                  mode='bilinear', align_corners=True
                                  ).squeeze()                     # (C,N)
            desc1 = F.grid_sample(d1[b:b+1],
                                  pts1[b].view(1, -1, 1, 2)
                                  .flip(-1) * 2 / torch.tensor(
                                      [h1.shape[-1]-1,
                                       h1.shape[-2]-1],
                                      device=h1.device) - 1,
                                  mode='bilinear', align_corners=True
                                  ).squeeze()                     # (C,M)

            # ground-truth correspondence via homography
            ones = torch.ones_like(pts0[b][:, :1])
            pts0_h = torch.cat([pts0[b], ones], 1).t()           # [3,N]
            proj = (H[b] @ pts0_h).t()                          # [N,3]
            proj = proj[:, :2] / (proj[:, 2:3] + 1e-8)

            dists = torch.cdist(proj, pts1[b], p=2)              # [N,M]
            match = dists < dist_thr

            # repeatability
            n_rep = float(match.any(1).sum())
            rep += n_rep / max(1e-6, float(pts0[b].shape[0]))

            # localisation error
            if n_rep > 0:
                err = (dists * match.float()).sum() / n_rep
                loc_err += err.item()

            # descriptor AP (image-level)
            if desc0.numel() and desc1.numel():
                # cosine sim
                sim = desc0.t() @ desc1                       # N×M
                lbl = match.float()
                ap_im = average_precision(sim.flatten(),
                                          lbl.flatten())
                desc_ap += ap_im

            n_kp += 0.5 * (pts0[b].shape[0] + pts1[b].shape[0])

        return (rep / B,
                desc_ap / B,
                loc_err / max(1e-6, rep * B),
                n_kp / B)

    def average_precision(scores, labels):
        ord = torch.argsort(scores, descending=True)
        labels = labels[ord]
        cum_pos = torch.cumsum(labels, 0)
        total_pos = labels.sum()
        if total_pos == 0:
            return 0.0
        prec = cum_pos / torch.arange(1, labels.numel()+1,
                                      device=labels.device)
        ap = (prec * labels).sum() / total_pos
        return ap.item()

    rep_sum = ap_sum = err_sum = kp_sum = 0.0
    n_batches = 0
    model.eval()

    with torch.no_grad():
        for b_idx, (img, _, _) in enumerate(tqdm(loader,
                                                 desc="val-quick",
                                                 leave=False)):
            if b_idx >= max_batches:
                break
            img = img.to(device, non_blocking=True)

            from utils.geometry import random_warp
            img_w, H = random_warp(img)
            H = H.to(device)

            with autocast(device_type=device.type,
                          dtype=torch.bfloat16):
                h0, off0, d0 = model(img)
                h1, off1, d1 = model(img_w)

            rep, ap, locerr, nkp = repeatability_metrics(
                h0, off0, d0,
                h1, off1, d1,
                H)

            rep_sum += rep
            ap_sum += ap
            err_sum += locerr
            kp_sum += nkp
            n_batches += 1

    return dict(rep=rep_sum / n_batches,
                desc_ap=ap_sum / n_batches,
                loc_err=err_sum / n_batches,
                num_kp= kp_sum  / n_batches)
