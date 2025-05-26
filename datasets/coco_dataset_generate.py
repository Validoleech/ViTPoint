import os
import sys
import argparse
import glob
import time
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count


def center_crop(img, target_wh):
    """Centre-crop np.uint8 (H,W,[C]) to desired aspect."""
    H, W = img.shape[:2]
    tgt_w, tgt_h = target_wh
    tgt_ratio = tgt_w / tgt_h
    cur_ratio = W / H

    if cur_ratio > tgt_ratio:  # crop width
        new_W = int(H * tgt_ratio)
        x0 = (W - new_W) // 2
        return img[:, x0:x0 + new_W]
    else:                      # crop height
        new_H = int(W / tgt_ratio)
        y0 = (H - new_H) // 2
        return img[y0:y0 + new_H, :]


_orb = None


def ensure_orb(nfeatures: int, fast_thr: int):
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(
            nfeatures=nfeatures, fastThreshold=fast_thr,
            scaleFactor=1.2, nlevels=8, edgeThreshold=31,
            scoreType=cv2.ORB_FAST_SCORE, patchSize=31
        )
    return _orb


def process_one(p: Path, tgt_w: int, tgt_h: int,
                out_img: Path, out_npz: Path,
                nfeatures: int, fast_thr: int, q: int) -> None:
    stem = Path(p).stem
    jpg_out = out_img / f"{stem}.jpg"
    npz_out = out_npz / f"{stem}.npz"
    if jpg_out.exists() and npz_out.exists():
        return
    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f" ! Could not read {p}")
        return
    img_bgr = center_crop(img_bgr, (tgt_w, tgt_h))
    img_bgr = cv2.resize(img_bgr, (tgt_w, tgt_h),
                         interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    orb = ensure_orb(nfeatures, fast_thr)
    kps = orb.detect(gray, None)

    if kps:
        pts = np.array([kp.pt for kp in kps],  dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        angles = np.array([kp.angle for kp in kps],    dtype=np.float32)
        sizes = np.array([kp.size for kp in kps],    dtype=np.float32)
    else:
        pts = scores = angles = sizes = np.empty((0,), np.float32)

    cv2.imwrite(str(jpg_out), img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, q])
    np.savez_compressed(npz_out,
                        kps=pts, scores=scores,
                        angles=angles, size=sizes)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", default="data/coco",
                    help="Folder that has train2017/, val2017 and test2017/ sub-dirs")
    parser.add_argument("--out_root", default="data/coco_processed",
                        help="where to write the .npz files")
    parser.add_argument("--split", choices=["train2017", "val2017", "test2017"],
                    default="train2017")
    parser.add_argument("--nfeatures", type=int, default=2000,
                    help="Max features per image ORB will return")
    parser.add_argument("--fast_thr", type=int, default=20,
                    help="FAST threshold inside ORB")
    parser.add_argument("--jpg_quality", type=int, default=99)
    
    args = parser.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    tgt_h, tgt_w = cfg["data"]["img_size"]
    img_dir = Path(args.coco_root) / args.split
    if not img_dir.exists():
        sys.exit(f"Folder {img_dir} not found.")
    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    orb = cv2.ORB_create(
        nfeatures=args.nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_FAST_SCORE,
        patchSize=31,
        fastThreshold=args.fast_thr,
        
    )
    img_dir = Path(args.coco_root) / args.split
    out_img = Path(args.out_root) / args.split / "images"
    out_npz = Path(args.out_root) / args.split / "processed"
    out_img.mkdir(parents=True, exist_ok=True)
    out_npz.mkdir(parents=True, exist_ok=True)

    images = sorted(glob.glob(str(img_dir / "*.jpg")))

    if not images:
        sys.exit(f"No images found in {img_dir}")
    procfun = partial(process_one,
                  tgt_w=tgt_w, tgt_h=tgt_h,
                  out_img=out_img, out_npz=out_npz,
                  nfeatures=args.nfeatures,
                  fast_thr=args.fast_thr,
                  q=args.jpg_quality)

    t0 = time.time()

    with Pool(processes=16, maxtasksperchild=200) as pool:
        list(tqdm(pool.imap_unordered(procfun, images),
                  total=len(images), ncols=100))
    dt = time.time() - t0
    print(f"\nProcessed {len(images)} images in {dt/60:.1f} min.\n"
          f"  Crops saved to {out_img}\n"
          f"  Results saved to {out_npz}")
