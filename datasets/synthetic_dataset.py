import torch
import cv2
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from .utils.synthetic_dataset_config import PersistentSynthConfig


class PersistentSyntheticShapes(Dataset):
    def __init__(self, split="train", cfg: PersistentSynthConfig = PersistentSynthConfig()):
        self.cfg = cfg
        self.split = split
        self.base = Path(cfg.data_root)
        self.patch = cfg.patch_size
        self.Hf, self.Wf = cfg.resize[0]//self.patch, cfg.resize[1]//self.patch

        self.samples = []
        for prim in cfg.primitives:
            imgs = sorted((self.base/prim/"images"/split).glob("*.png"))
            self.samples += [
                (str(p), str(p).replace("images", "points").replace(".png", ".npy"))
                for p in imgs
            ]
        random.shuffle(self.samples)
        if len(self.samples) == 0:
            raise RuntimeError(f"No synthetic data for split={split}. "
                               "Run datasets/persistent_synth_generate.py first.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts_path = self.samples[idx]
        # image  [3,H,W]  in 0-1 float32  (RGB)
        img_bgr = cv2.imread(img_path)                     # H×W×3  uint8, BGR
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        # corner keypoints  (x,y) in full-res pixel coords
        kp = np.load(pts_path).astype(np.float32)          # N×2

        # coarse grid size (h,w)  =  Hf, Wf
        heat = torch.zeros(1, self.Hf, self.Wf)
        offset_gt = torch.zeros(2, self.Hf, self.Wf)       # Δy,Δx

        for x, y in kp:
            xf = int(x // self.patch)                      # grid col index
            yf = int(y // self.patch)                      # grid row index
            if 0 <= xf < self.Wf and 0 <= yf < self.Hf:
                        heat[0, yf, xf] = 1.0

                        # sub-pixel offset  ∈ (-0.5, +0.5)
                        dx = (x % self.patch) / self.patch - 0.5   # +→right
                        dy = (y % self.patch) / self.patch - 0.5   # +→down
                        offset_gt[0, yf, xf] = torch.tensor(
                            dy, dtype=offset_gt.dtype)             # channel-0 = Δy
                        offset_gt[1, yf, xf] = torch.tensor(
                            dx, dtype=offset_gt.dtype)             # channel-1 = Δx

        return img, heat, offset_gt
