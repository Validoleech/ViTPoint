import os
import glob
import random
import cv2
import torch
from torch.utils.data import Dataset
from utils.homography import sample_homography


class COCOWarped(Dataset):
    """
    Returns (img, img_warped, H, warped_grid) for self-sup Stage-2.
    """
    def __init__(self, root, img_size=(240, 320)):
        self.paths = glob.glob(os.path.join(root, '*.jpg'))
        self.H, self.W = img_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # BGR->RGB
        img = torch.from_numpy(img).permute(2, 0, 1).float()/255.

        with torch.no_grad():
            H_mat, warped_grid = sample_homography(
                1, self.H, self.W, img.device)
        return img, H_mat[0], warped_grid[0]
