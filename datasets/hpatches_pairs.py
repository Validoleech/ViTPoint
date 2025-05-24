import glob
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class HPatchesPairs(Dataset):
    """
    Loads the first image ('1.ppm') of each sequence as ref,
    and the 2nd-6th images as target.  Homographies are in 'H_1to?.txt'.
    """

    def __init__(self, root="data/hpatches", img_size=(224, 224)):
        seqs = sorted(glob.glob(os.path.join(root, "*")))
        self.pairs = []
        self.H, self.W = img_size
        for seq in seqs:
            im1 = os.path.join(seq, "1.ppm")
            for i in range(2, 7):
                im2 = os.path.join(seq, f"{i}.ppm")
                Htxt = os.path.join(seq, f"H_1_{i}.txt")
                if os.path.isfile(im2) and os.path.isfile(Htxt):
                    self.pairs.append((im1, im2, Htxt))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, htxt = self.pairs[idx]
        img1 = cv2.resize(cv2.imread(path1)[:, :, ::-1], (self.W, self.H))
        img2 = cv2.resize(cv2.imread(path2)[:, :, ::-1], (self.W, self.H))
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()/255.
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()/255.
        H = torch.tensor(np.loadtxt(htxt), dtype=torch.float)
        return img1, img2, H
