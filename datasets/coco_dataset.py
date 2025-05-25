import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, tv_tensors


class COCODataset(Dataset):
    """
    COCO dataset whose labels come from a teacher detector stored in
    in *.npz files as produced by datasets/coco_dataset_generate.py.

    Parameters
    ----------
    split         "train2017" | "val2017"
    coco_root     folder that contains   images/<id>.jpg
    teacher_root  folder that contains   processed/<id>.npz
    img_size      (H, W) – must match the extractor, e.g. (224, 224)
    patch_size    ViT patch
    score_thr     discard teacher points below this score
    max_kp        keep at most K strongest points (None = keep all)
    """

    def __init__(self,
                 split: str,
                 coco_root: str = "coco_processed/train2017/images",
                 teacher_root: str = "coco_processed/train2017/processed",
                 img_size=(224, 224),
                 patch_size=14,
                 score_thr: float = 0.0,
                 max_kp: int | None = None):

        self.H, self.W = img_size
        self.ps = patch_size
        self.h, self.w = self.H // self.ps, self.W // self.ps

        self.img_dir = Path(coco_root)
        self.lbl_dir = Path(teacher_root)

        assert self.img_dir.exists(
        ), f"images folder not found: {self.img_dir}"
        assert self.lbl_dir.exists(
        ), f"processed folder not found: {self.lbl_dir}"

        self.paths = sorted(self.img_dir.glob("*.jpg"))
        assert self.paths, f"No .jpg in {self.img_dir}"

        self.to_tensor = transforms.ToTensor()
        self.score_thr = score_thr
        self.max_kp = max_kp

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        jpg = self.paths[idx]
        stem = jpg.stem
        npz = self.lbl_dir / f"{stem}.npz"

        img = Image.open(jpg).convert("RGB")
        assert img.size == (self.W, self.H), \
            f"image {jpg} size {img.size}, expected {(self.W, self.H)}"
        img_t = self.to_tensor(img)           # [3,H,W] float32 0-1

        data = np.load(npz)
        kps = data["kps"]                     # (N,2)
        scores = data["scores"]               # (N,)

        # filter / top-K
        m = scores >= self.score_thr
        kps, scores = kps[m], scores[m]
        if self.max_kp is not None and len(scores) > self.max_kp:
            idx_top = np.argsort(scores)[-self.max_kp:]
            kps, scores = kps[idx_top], scores[idx_top]

        # build heat & offset maps
        heat = torch.zeros((1, self.h, self.w), dtype=torch.float32)
        off = torch.zeros((2, self.h, self.w), dtype=torch.float32)

        for (x, y) in kps:
            j, i = int(x // self.ps), int(y // self.ps)   # patch coords
            if 0 <= i < self.h and 0 <= j < self.w:
                heat[0, i, j] = 1.0
                cy = i * self.ps + self.ps / 2
                cx = j * self.ps + self.ps / 2
                off[0, i, j] = (y - cy) / self.ps         # Δy
                off[1, i, j] = (x - cx) / self.ps         # Δx
        off.clamp_(-0.5, 0.5)

        return img_t, heat, off
