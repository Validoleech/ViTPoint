"""
Visualise GT vs. predicted corners for a single validation sample.

Usage:
    python overlay_debug.py
    python overlay_debug.py --ckpt runs/stage0/best.pth --idx 3 --thr 0.7
"""

import argparse
import yaml
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.amp import autocast

from models.vitpoint import SuperPointViT, HeadConfig
from datasets.synthetic_dataset import PersistentSyntheticShapes
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig


def build_model(cfg, device, ckpt_path=None):
    """Construct backbone + heads, load a checkpoint."""
    model = SuperPointViT(
        HeadConfig(cfg["model"]["dim_descriptor"]),
        freeze_backbone=True).to(device)

    if ckpt_path:
        print(f"[overlay] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        # accept either pure state_dict or {'model': …}
        model_sd = state["model"] if "model" in state else state
        model.load_state_dict(model_sd, strict=True)

    model.eval()
    return model


def upsample(mask, size):
    """Nearest/bilinear upsample helper (torch → torch)."""
    if mask.ndim == 2:                   # [h,w] → [1,h,w]
        mask = mask.unsqueeze(0)
    return TF.resize(mask,
                     size=size,
                     interpolation=TF.InterpolationMode.BILINEAR,
                     antialias=False)[0]


def debug_overlay(model, sample, device, thr=0.5):
    """
    Plot RGB image with GT (green) and prediction (red) masks.
    sample = (img, heat_gt, offset) as returned by dataset.
    """
    starttime = time.time()
    img, heat_gt, _ = sample                                   # img [3,H,W]
    img = img.to(device)[None]                                 # [1,3,H,W]

    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
        heat_pr, _, _ = model(img)                             # [1,1,h,w]
    result_time = time.time() - starttime

    heat_pr = heat_pr[0, 0].cpu()
    heat_gt = heat_gt[0].cpu()

    H_img, W_img = img.shape[2:]
    heat_gt_up = upsample(heat_gt,  (H_img, W_img))
    heat_pr_up = upsample(heat_pr,  (H_img, W_img))

    base = img[0].cpu().permute(1, 2, 0).numpy()               # [H,W,3] 0–1
    rgba = np.concatenate([base, np.ones_like(base[:, :, :1])], axis=2)

    # ground-truth → GREEN
    mask_gt = heat_gt_up > 0.5
    rgba[mask_gt, :3] = [0.0, 1.0, 0.0]        # pure green
    rgba[mask_gt, 3] = 0.6                    # α

    # prediction → RED
    mask_pr = heat_pr_up > thr
    rgba[mask_pr, :3] = [1.0, 0.0, 0.0]        # pure red
    rgba[mask_pr, 3] = 0.6

    plt.figure(figsize=(6, 6))
    plt.title(f"GT (green)  vs.  prediction (red)  thr={thr}")
    plt.imshow(rgba)
    plt.axis("off")
    plt.show()
    return result_time


def main():
    parser = argparse.ArgumentParser(description="Overlay debug visualiser")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="path to .pth checkpoint (optional)")
    parser.add_argument("--idx", type=int, default=0,
                        help="index of the validation sample")
    parser.add_argument("--thr", type=float, default=0.5,
                        help="threshold for prediction mask")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read config.yaml
    cfg = yaml.safe_load(open("config.yaml"))
    # synthetic dataset cfg (same as in train_0.py)
    ps_cfg = PersistentSynthConfig()
    ds_val = PersistentSyntheticShapes("val", ps_cfg)

    # build & load model
    model = build_model(cfg, device, ckpt_path=args.ckpt)

    # pick and visualise one sample
    if not (0 <= args.idx < len(ds_val)):
        raise IndexError(
            f"idx out of range: {args.idx} (dataset len {len(ds_val)})")
    sample = ds_val[args.idx]
    print(f"[overlay] sample #{args.idx}  image shape {sample[0].shape}")

    print(f"in {debug_overlay(model, sample, device, thr=args.thr)}")


if __name__ == "__main__":
    main()
