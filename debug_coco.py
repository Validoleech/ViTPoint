"""
Overlay teacher (green) vs. student predictions (red) on COCO images.
Optional --warp flag shows the same image after a random homography, so you
can eyeball repeatability.

Usage
-----
python overlay_coco_debug.py                         # first image
python overlay_coco_debug.py --idx 42 --thr 0.6      # custom threshold
python overlay_coco_debug.py --warp                  # original + warped
"""
from pathlib import Path
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.amp import autocast

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.coco_dataset import COCODataset
from utils.loss import _nms
from utils.homography import sample_homography


def build_model(cfg, device, ckpt):
    model = SuperPointViT(
        HeadConfig(cfg["model"]["dim_descriptor"]), freeze_backbone=True
    ).to(device)

    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[overlay] loaded {ckpt}")
    model.eval()
    return model


def upsample(t, size):
    if t.ndim == 2:
        t = t[None]
    return TF.resize(t, size, TF.InterpolationMode.BILINEAR, antialias=False)[0]


def rgba_overlay(rgb_t, gt_heat, pr_heat, thr):
    """
    rgb_t   : [3,H,W] 0-1 tensor
    gt_heat : [h,w]   teacher mask   (values 0/1)
    pr_heat : [h,w]   student sigmoid output
    returns : [H,W,4] float32  RGBA
    """
    H, W = rgb_t.shape[1:]
    gt_up = upsample(gt_heat, (H, W))
    pr_up = upsample(pr_heat, (H, W))

    rgba = torch.cat([rgb_t, torch.ones(1, H, W)], 0).permute(1, 2, 0).clone()

    # teacher – GREEN
    m_gt = gt_up > 0.5
    rgba[m_gt, :3] = torch.tensor([0, 1, 0])
    rgba[m_gt,  3] = 0.6

    # student – RED (after NMS + thr for clarity)
    pts = _nms(pr_heat, thr, topk=500).long()
    if pts.numel():
        xs, ys = pts.unbind(1)
        rgba[ys * (H // gt_heat.shape[0]), xs *
             (W // gt_heat.shape[1]), :3] = torch.tensor([1, 0, 0])
        rgba[ys * (H // gt_heat.shape[0]), xs *
             (W // gt_heat.shape[1]),  3] = 0.7

    return rgba.numpy()


# ------------------------------------------------------------
def create_overlay_pair(model, ds, idx, device, thr, warp):
    """
    Returns one (overlay,) or two (overlay_orig, overlay_warp) numpy arrays.
    """
    img_t, heat_gt, _ = ds[idx]        # tensors
    H_img, W_img = img_t.shape[1:]
    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
        pr, _, _ = model(img_t[None].to(device))
    pr = pr[0, 0].cpu()

    out_imgs = [rgba_overlay(img_t, heat_gt[0], pr, thr)]

    if warp:
        # build a homography & warp the image tensor
        H_mat, grid = sample_homography(1, H_img, W_img, img_t.device)
        warp_t = torch.nn.functional.grid_sample(
            img_t.unsqueeze(0),          # [1,3,H,W]
            grid, align_corners=True
        )[0]
        with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
            pr_w, _, _ = model(warp_t[None].to(device))
        pr_w = pr_w[0, 0].cpu()
        dummy_gt = torch.zeros_like(pr_w)     # no GT in warped view
        out_imgs.append(rgba_overlay(warp_t.cpu(), dummy_gt, pr_w, thr))

    return tuple(out_imgs)


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--root",   default="data/coco_processed")
    ap.add_argument("--split",  default="val2017")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--warp", action="store_true",
                    help="add a homography-warped second view")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))

    ds = COCODataset(split=args.split,
                     coco_root=f"{args.root}/{args.split}/images",
                     teacher_root=f"{args.root}/{args.split}/processed",
                     img_size=cfg["data"]["img_size"],
                     patch_size=cfg["data"]["patch"])

    model = build_model(cfg, device, args.ckpt)

    overlays = create_overlay_pair(model, ds, args.idx, device,
                                   thr=args.thr, warp=args.warp)

    # if run as script → show
    for i, im in enumerate(overlays):
        plt.figure(figsize=(6, 6))
        plt.imshow(im)
        plt.title("original" if i == 0 else "warped")
        plt.axis("off")
    plt.show()
