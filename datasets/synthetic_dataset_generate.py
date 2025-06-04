
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


def gen_one(img: np.ndarray, primitive: str) -> np.ndarray:
    """
    Generates a single geometric primitive on the given image using the
    synthetic drawing utility module `sdu`. Requires `draw_' as a start of the name function.

    Args:
        img (np.ndarray): The input image (background) on which to draw the primitive.
                          Expected to be a NumPy array representing an image.
        primitive (str): The name of the primitive to draw (e.g., "line", "circle").
                         This name corresponds to a function in the `sdu` module
                         like `draw_line`, `draw_circle`.

    Returns:
        np.ndarray: A NumPy array of shape `(N, 2)` containing the (x, y) coordinates
                    of the points generated for the primitive.
                    Returns an empty array if no points are generated or an error occurs.

    Raises:
        ValueError: If a drawing function corresponding to the `primitive` name
                    (e.g., `draw_line` for `primitive="line"`) is not found
                    in the `synthetic_drawing_utils` module (`sdu`).
    """
    fn_name = f"draw_{primitive}"
    if not hasattr(sdu, fn_name):
        raise ValueError(f"synthetic_drawing_utils missing {fn_name}()")
    return getattr(sdu, fn_name)(img)


def photometric(img: np.ndarray) -> None:
    """Colour-jitter + Gaussian noise"""
    lab = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    L *= 0.8 + 0.4 * sdu.random_state.rand()
    L += sdu.random_state.randn(*L.shape) * 5
    A += sdu.random_state.randn(*A.shape) * 10
    B += sdu.random_state.randn(*B.shape) * 10
    lab = cv2.merge([np.clip(L, 0, 255),
                     np.clip(A, 0, 255),
                     np.clip(B, 0, 255)]).astype(np.uint8)
    img[:] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[:, :, 0]
    # Additive noise
    sigma = sdu.random_state.uniform(3, 10)
    img[:] = np.clip(img.astype(np.float32) +
                     sdu.random_state.randn(*img.shape) * sigma,
                     0, 255).astype(np.uint8)


def generate_single_sample(prim: str, split: str, idx: int) -> None:
    """
    Generates a single synthetic data sample. This involves drawing a primitive,
    applying transformations (blur, resize), scaling the corresponding points,
    and saving the resulting image and point data to disk.

    Args:
        prim (str): The name of the primitive to generate (e.g., "line", "circle").
        split (str): The dataset split (e.g., "train", "val", "test").
        idx (int): The unique index for the current sample within its split.

    Returns:
        None: The function saves the generated image and points directly to disk in
              the configured directories.
    """
    base = Path(cfg.data_root)
    im_dir = base / prim / "images" / split
    pts_dir = base / prim / "points" / split

    attempt = 0
    while True:
        try:
            big = sdu.generate_background(size=cfg.gen_img_size)
            pts = gen_one(big, prim)
        except Exception as e:
            print(f"! {prim} sample {idx} failed: {e}")
            big = np.zeros(cfg.gen_img_size, np.uint8)
            pts = np.empty((0, 2), np.int32)

        if not cfg.hard_mode:
            break

        scale_x = cfg.resize[1] / cfg.gen_img_size[1]
        scale_y = cfg.resize[0] / cfg.gen_img_size[0]
        pts_scaled = pts.astype(float)
        pts_scaled[:, 0] *= scale_x
        pts_scaled[:, 1] *= scale_y

        grid_h = cfg.resize[0] // cfg.patch_size
        grid_w = cfg.resize[1] // cfg.patch_size
        binc = np.zeros((grid_h, grid_w), int)

        cols = (pts_scaled[:, 0] // cfg.patch_size).astype(int)
        rows = (pts_scaled[:, 1] // cfg.patch_size).astype(int)
        valid = (cols >= 0) & (cols < grid_w) & (rows >= 0) & (rows < grid_h)
        np.add.at(binc, (rows[valid], cols[valid]), 1)
        
        if (binc >= cfg.min_kpts_per_patch).all():
            break

        # add dots in missing patches
        missing = np.argwhere(binc < cfg.min_kpts_per_patch)
        for r, c in missing:
            cx_big = int((c + 0.5) * cfg.patch_size / scale_x)
            cy_big = int((r + 0.5) * cfg.patch_size / scale_y)
            rad = int(cfg.patch_size * 0.35 / scale_x)
            cv2.circle(big, (cx_big, cy_big), rad,
                       sdu.get_random_color(int(np.mean(big))), -1)
            pts = np.vstack([pts, [cx_big, cy_big]])

        attempt += 1
        if attempt >= cfg.max_gen_attempts:
            break

    if cfg.hard_mode:
        photometric(big)

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


def generate_all(force: bool = False) -> None:
    """
    Orchestrates the generation of all synthetic data samples for all
    specified primitives and splits defined in the configuration.

    Existing data for a primitive/split will be skipped if `force` is False
    and the expected number of images already exist.

    Args:
        force (bool): If True, overwrites existing data and regenerates all samples,
                      regardless of whether they already exist. If False, skips
                      generation for primitives and splits that already have the
                      expected number of images. Defaults to False.

    Returns:
        None: This function manages the generation process and saves files to disk.
    """

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
