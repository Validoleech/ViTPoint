import torch
import torch.nn.functional as F
import kornia as K
import cv2


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


def _nms(heat, thr, topk):
    """
    heat : [..., H, W] – anything that ends with the 2-D score map.
    returns  : (N, 2) xy coordinates kept after NMS
    """
    from torchvision.ops import nms
    heat2d = heat.reshape(-1, *heat.shape[-2:])[-1]   # (H,W)
    ys, xs = torch.where(heat2d > thr)
    if ys.numel() == 0:
        return torch.empty((0, 2), device=heat.device)
    scores = heat2d[ys, xs]
    boxes = torch.stack([xs, ys, xs + 1, ys + 1], 1).float()
    keep = nms(boxes, scores, iou_threshold=0.1)[:topk]

    return torch.stack([xs[keep], ys[keep]], 1).float()

def infonce_loss(desc0, desc1, H,   # desc:[B,D,h,w]  H:[B,3,3]
                 patch_sz=14, tau=0.1, max_pts=1024):
    """
    For each image in the batch:
        • sample up to `max_pts` random grid points
        • warp them by H to the second view
        • use cosine similarity & InfoNCE
    Assumes desc maps are already L2-normalised.
    """
    B, D, h, w = desc0.shape
    device = desc0.device
    losses = []

    # build grid in patch coords (0 .. w-1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device), indexing='ij')
    pts = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)       # h×w×3

    pts = (pts.view(-1, 3).t().float()) * patch_sz + patch_sz/2.   # 3×N pix

    for b in range(B):
        # random subset
        idx = torch.randperm(pts.shape[1], device=device)[:max_pts]
        p0 = pts[:, idx]                              # 3×N
        p1 = H[b] @ p0                                # 3×N
        p1 = p1 / p1[2:]                              # normalise

        # filter inside image
        mask = (p1[0] > 0) & (p1[0] < w*patch_sz) & (p1[1]
                                                     > 0) & (p1[1] < h*patch_sz)
        p0, p1 = p0[:, mask], p1[:, mask]
        if p0.shape[1] < 10:    # nothing valid
            continue

        # ─ descriptors
        # map from pixel → patch grid → [-1,1]
        def _grid(pix):
            xs = pix[0] / (w*patch_sz - 1) * 2 - 1
            ys = pix[1] / (h*patch_sz - 1) * 2 - 1
            return torch.stack([xs, ys], dim=-1).view(1, 1, -1, 2)
        d0 = F.grid_sample(desc0[b:b+1], _grid(p0),
                           align_corners=True).squeeze().t()  # N×D
        d1 = F.grid_sample(desc1[b:b+1], _grid(p1),
                           align_corners=True).squeeze().t()

        # cos-sim / tau
        sim = d0 @ d1.t() / tau
        target = torch.arange(sim.size(0), device=device)
        losses.append(F.cross_entropy(sim, target))

    return torch.stack(losses).mean() if losses else torch.tensor(0., device=device)

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
def detector_repeatability(model, pairs, device, thr=0.5, topk=500):
    """
    pairs : iterable of (img0,img1,H01)  where H01 warps 0→1
    returns (repeat, loc_err)
    """
    n_hit, n_tot, errs = 0, 0, []
    for im0, im1, H in pairs:
        im0, im1, H = im0.to(device), im1.to(device), H.to(device)
        h0, _, _ = model(im0[None])
        h1, _, _ = model(im1[None])

        k0 = _nms(h0[0, 0], thr, topk)         # N×2 xy
        k1 = _nms(h1[0, 0], thr, topk)

        if k0.numel() == 0 or k1.numel() == 0:
            continue
        k0w = K.geometry.homography_warp_points(k0[None], H[None])[0]

        dists = torch.cdist(k0w, k1)           # N0×N1
        hit = dists.min(1).values
        mask = hit < 3
        n_hit += mask.sum().item()
        n_tot += k0.shape[0]
        errs += hit[mask].tolist()
    if n_tot == 0:
        return 0., 0.
    return n_hit/n_tot, (sum(errs)/len(errs) if errs else 0.)


def descriptor_fpr95(model, pairs, device, temp=0.1, topk=500):
    """
    Very light version: one homography pair per image.
    """
    pos_sims, neg_sims = [], []
    for im0, im1, H in pairs:
        im0, im1, H = im0.to(device), im1.to(device), H.to(device)
        _, _, d0 = model(im0[None])
        _, _, d1 = model(im1[None])
        k0 = _nms(torch.sigmoid(model.det_head(d0))[0, 0], .5, topk)
        if k0.numel() == 0:
            continue
        k0w = K.geometry.homography_warp_points(k0[None], H[None])[0]
        # sample descriptors

        def _sample(desc, pts):
            h, w = desc.shape[2:]
            grid = pts.clone()
            grid[:, 0] = grid[:, 0] / (w-1) * 2 - 1
            grid[:, 1] = grid[:, 1] / (h-1) * 2 - 1
            return F.grid_sample(desc, grid.view(1, 1, -1, 2),
                                 align_corners=True)[0, :, 0].t()  # N×D
        q = _sample(d0, k0)
        p = _sample(d1, k0w)
        sim = (q*p).sum(1)
        pos_sims.append(sim)

        # negatives: shuffle
        p_neg = p[torch.randperm(p.shape[0])]
        neg_sims.append((q*p_neg).sum(1))

    if not pos_sims:
        return 1.0   # worst case
    pos = torch.cat(pos_sims)
    neg = torch.cat(neg_sims)
    thr = torch.quantile(pos, 0.05)       # 95 % TPR
    fpr = (neg > thr).float().mean().item()
    return fpr


@torch.no_grad()
def homography_auc(model, pairs, device, thr=(3, 5, 10), topk=500):
    acc = [0]*len(thr)
    total = 0
    for im0, im1, H in pairs:
        im0, im1, H = im0.to(device), im1.to(device), H.to(device)
        h0, _, _ = model(im0[None])
        pts0 = _nms(h0[0, 0], .5, topk)
        if pts0.numel() == 0:
            continue
        _, _, d0 = model(im0[None])
        _, _, d1 = model(im1[None])
        pts0w = K.geometry.homography_warp_points(pts0[None], H[None])[0]

        # build matches by nearest descriptor
        def samp(desc, xys):
            h, w = desc.shape[2:]
            g = xys.clone()
            g[:, 0] = g[:, 0]/(w-1)*2-1
            g[:, 1] = g[:, 1]/(h-1)*2-1
            return F.grid_sample(desc, g.view(1, 1, -1, 2), align_corners=True)[0, :, 0].t()
        q = samp(d0, pts0)
        p = samp(d1, pts0w)
        sim = (q@p.t())
        idx = torch.arange(sim.shape[0])
        matches = torch.stack([idx, sim.argmax(1)], 1)
        src = pts0[matches[:, 0]]
        dst = pts0w[matches[:, 1]]

        if len(src) < 4:
            continue
        H_est, _ = cv2.findHomography(src.cpu().numpy(),
                                      dst.cpu().numpy(), cv2.RANSAC)
        if H_est is None:
            continue
        H_est = torch.tensor(H_est, device=device).float()
        # compute corner transfer error
        corners = torch.tensor([[0, 0], [223, 0], [223, 223], [0, 223]],
                               device=device).float()
        est = K.geometry.warp_points(corners[None], H_est[None])[0]
        gt = K.geometry.warp_points(corners[None], H[None])[0]
        err = ((est-gt).pow(2).sum(1).sqrt()).mean().item()
        for i, t in enumerate(thr):
            if err < t:
                acc[i] += 1
        total += 1
    return [a/total if total else 0. for a in acc]