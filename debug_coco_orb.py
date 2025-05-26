import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
import numbers

from datasets.coco_dataset import COCODataset
from utils.geometry import random_warp


def tensor_to_cv(img_t):
    """
    [3,H,W] float tensor (0-1) →  H×W×3 uint8 BGR
    """
    img = (img_t.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_orb(img_bgr, kp_color=(0, 255, 0)):
    """
    Takes a BGR image, detects ORB key-points and draws them in place.
    Returns the image with keypoints.
    """
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
    kps = orb.detect(img_bgr, None)
    for kp in kps:
        kp.size *= 0.1
    img = cv2.drawKeypoints(img_bgr, kps, None, kp_color,
                            #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return kps, img

def make_mosaic(img1_bgr, img2_bgr):
    """
    Put two BGR images side by side with a separating white bar.
    """
    h, w, _ = img1_bgr.shape
    bar = 255 * np.ones((h, 8, 3), dtype=np.uint8)
    bar_w = bar.shape[1]
    return np.concatenate([img1_bgr, bar, img2_bgr], axis=1), bar_w


def find_matches(kps1, kps2, H, thr=4):
    if len(kps1) == 0 or len(kps2) == 0:
        return []

    # kp.pt order is (x, y)
    pts1 = np.array([kp.pt for kp in kps1], dtype=np.float32).reshape(-1, 1, 2)
    pts1_warp = cv2.perspectiveTransform(pts1, H)[:, 0, :]     # N×2

    pts2 = np.array([kp.pt for kp in kps2], dtype=np.float32)  # M×2

    matches = []
    for i, (xw, yw) in enumerate(pts1_warp):
        dists = np.linalg.norm(pts2 - np.array([xw, yw]), axis=1)
        j = dists.argmin()
        if dists[j] < thr:
            matches.append((i, j))
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualise COCO sample + warp + ORB kpts")
    parser.add_argument("--idx", type=int, default=0,
                    help="image index in the val2017 split")
    parser.add_argument("--save", type=Path,
                    help="optional path to save the mosaic (png/jpg)")
    parser.add_argument("--cfg", default="config.yaml", type=Path)
    parser.add_argument("--matches", default=False,
                    help="draw lines between homography-consistent points"),
    parser.add_argument("--thr", default=0.5)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))

    # build dataset ----------------------------------------------------
    ds = COCODataset("val2017",
                     coco_root="data/coco_processed/val2017/images",
                     teacher_root="data/coco_processed/val2017/processed",
                     img_size=cfg["data"]["img_size"],
                     patch_size=cfg["data"]["patch_size"])

    # fetch sample
    img_t, *_ = ds[args.idx]                 # [3,H,W] float32 0-1
    img_bgr = tensor_to_cv(img_t)

    # warp
    img_warp_t, H_t = random_warp(img_t.unsqueeze(0))  # (1,3,H,W)
    img_warp_bgr = tensor_to_cv(img_warp_t[0])
    H = H_t[0].cpu().numpy()

    # ORB drawing
    kps1, img1_vis = draw_orb(img_bgr,      (0, 255,   0))  # green
    kps2, img2_vis = draw_orb(img_warp_bgr, (0,   0, 255))  # red
    
    mosaic, bar_w = make_mosaic(img1_vis, img2_vis)
    h, w1, _ = img1_vis.shape

    if args.matches:
        corr = find_matches(kps1, kps2, H, thr=args.thr)
        for i, j in corr:
            x1, y1 = kps1[i].pt
            x2, y2 = kps2[j].pt
            # shift x2 by original width + bar width to lie in the right image
            x2_shift = x2 + w1 + bar_w
            cv2.line(mosaic,
                     (int(x1), int(y1)),
                     (int(x2_shift), int(y2)),
                     color=(255, 255, 0), thickness=1)

        print(f"{len(corr)} correspondences (thr={args.thr}px)")
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(mosaic_rgb)
    plt.axis("off")
    title = f"idx {args.idx}   —   left: original   |   right: warped"
    if args.matches:
        title += f"   ({len(corr)} matches)"
    plt.title(title)
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    plt.imshow(mosaic_rgb)
    plt.axis("off")
    plt.title(f"COCO val2017 idx {args.idx}  —  left: original  |  right: warped")
    plt.tight_layout()

    if args.save:
        cv2.imwrite(str(args.save), mosaic)
        print(f"saved to {args.save}")
    else:
        plt.show()