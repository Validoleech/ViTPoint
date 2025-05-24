
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import datasets.utils.synthetic_drawing as sdu
from datasets.utils.synthetic_dataset_config import PersistentSynthConfig


cfg = PersistentSynthConfig()
rng = np.random.RandomState(cfg.random_seed)
sdu.set_random_state(rng)


def gen_one(img, primitive):
    fn_name = f"draw_{primitive}"
    if not hasattr(sdu, fn_name):
        raise ValueError(f"synthetic_drawing_utils missing {fn_name}()")
    return getattr(sdu, fn_name)(img)


def generate_single_sample(args):
    prim, split, idx = args
    base = Path(cfg.data_root)
    im_dir = base / prim / "images" / split
    pts_dir = base / prim / "points" / split

    try:
        big = sdu.generate_background(size=cfg.gen_img_size)
        pts = gen_one(big, prim)
    except Exception as e:
        print(f"! {prim} sample {idx} failed: {e}")
        pts = np.empty((0, 2), np.int32)

    if cfg.blur % 2 == 1 and cfg.blur > 0:
        big = cv2.GaussianBlur(big, (cfg.blur, cfg.blur), 0)

    small = cv2.resize(big, cfg.resize[::-1], interpolation=cv2.INTER_LINEAR)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    scale_x = cfg.resize[1] / cfg.gen_img_size[1]
    scale_y = cfg.resize[0] / cfg.gen_img_size[0]
    pts = pts.astype(np.float32)
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y

    cv2.imwrite(str(im_dir / f"{idx}.png"), small)
    #assert pts.shape[0] >= 30, f"Too sparse: {pts.shape[0]} pts"
    np.save(str(pts_dir / f"{idx}.npy"), pts)


def generate_all(force=False):
    base = Path(cfg.data_root)

    for prim in cfg.primitives:
        for split, n_imgs in cfg.split_sizes.items():
            im_dir = base / prim / "images" / split
            pts_dir = base / prim / "points" / split

            if im_dir.exists() and not force and len(list(im_dir.glob("*.png"))) == n_imgs:
                continue

            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            args_list = [(prim, split, i) for i in range(n_imgs)]
            with Pool(processes=cpu_count()) as pool:
                list(tqdm(pool.imap_unordered(generate_single_sample, args_list),
                          total=n_imgs, desc=f"{prim:18s} {split:<5s}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing data")
    args = parser.parse_args()
    generate_all(force=args.force)
