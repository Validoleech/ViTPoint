import math
from typing import Tuple

import numpy as np
import torch
import kornia.geometry.transform as KG

def _make_random_homographies(
        batch: int,
        height: int,
        width: int,
        *,
        degrees: float = 25.,
        scale: float = 0.15,
        translate: float = 0.15,
        device=None,
        dtype=None
) -> torch.Tensor:
    """
    Returns homography H that maps source ➜ destination pixel coordinates.
    The transform is:  translate → scale → rotate  around the image centre.
    """
    device = device or torch.device('cpu')
    dtype = dtype or torch.float32

    rng = np.random.default_rng()
    angles = rng.uniform(-degrees, degrees, size=batch) * \
        math.pi / 180.
    scales = rng.uniform(1 - scale, 1 + scale, size=batch)
    tx_pix = rng.uniform(-translate, translate, size=batch) * width
    ty_pix = rng.uniform(-translate, translate, size=batch) * height

    # --- build the 3×3 matrices in torch ---------------------------------- #
    cx, cy = (width - 1) / 2., (height - 1) / 2.              # image centre
    H = torch.empty(batch, 3, 3, device=device, dtype=dtype)

    for i in range(batch):
        a, s, tx, ty = angles[i], scales[i], tx_pix[i], ty_pix[i]

        # rotation (about origin)
        R = torch.tensor([[math.cos(a), -math.sin(a), 0.],
                          [math.sin(a),  math.cos(a), 0.],
                          [0.,           0., 1.]], device=device, dtype=dtype)

        # isotropic scaling
        S = torch.tensor([[s, 0., 0.],
                          [0., s, 0.],
                          [0., 0., 1.]], device=device, dtype=dtype)

        # translation in pixels
        T = torch.tensor([[1., 0., tx],
                          [0., 1., ty],
                          [0., 0., 1.]], device=device, dtype=dtype)

        # move origin to centre, apply R·S, move origin back, finally translate
        C = torch.tensor([[1., 0.,  cx],
                          [0., 1.,  cy],
                          [0., 0., 1.]], device=device, dtype=dtype)
        C_inv = torch.tensor([[1., 0., -cx],
                              [0., 1., -cy],
                              [0., 0.,  1.]], device=device, dtype=dtype)

        H[i] = T @ C @ (R @ S) @ C_inv

    return H

def random_warp(img: torch.Tensor,
                d_deg:   float = 25.,
                d_scale: float = 0.15,
                d_trans: float = 0.15
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random perspective warp (pure torch + numpy, only Kornia's warp used).

    Args
    ----
    img    : (B,3,H,W) or (3,H,W) tensor, float32/float64
    d_deg  : max rotation in degrees  (±)
    d_scale: max relative scale       (±)
    d_trans: max relative translation (± of image size)

    Returns
    -------
    warped_img : same shape as `img`
    H          : (B,3,3) homography, maps **source ➜ destination** pixels
    """
    single = img.ndim == 3
    if single:
        img = img.unsqueeze(0)                   # fake batch dim

    B, _, H_img, W_img = img.shape
    H_mat = _make_random_homographies(
        B, H_img, W_img,
        degrees=d_deg, scale=d_scale, translate=d_trans,
        device=img.device, dtype=img.dtype
    )

    warped = KG.warp_perspective(
        img, H_mat, dsize=(H_img, W_img),
        mode="bilinear", align_corners=True
    )

    return (warped.squeeze(0) if single else warped, H_mat)
