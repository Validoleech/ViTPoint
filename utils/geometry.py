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


def _apply_homography(pts, H):
    """
    pts :  [B,3,N]  – homogeneous coordinates
    H   :  [B,3,3]
    """
    warped = torch.bmm(H, pts)                             # [B,3,N]
    warped = warped / (warped[:, 2:3, :] + 1e-8)           # z-normalise
    return warped[:, :2, :]

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


def warp_offsets(offset_map: torch.Tensor,
                 H: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    offset_map : torch.FloatTensor  (B, 2, H, W)
        (dx,dy) ground-truth offsets defined in the *source* image.
    H : torch.FloatTensor  (B, 3, 3)
        Homography that transforms source → warped image.

    Returns
    -------
    torch.FloatTensor  (B, 2, H, W)
        Offsets expressed in the *warped* image frame, so they can be
        supervised against the prediction h1/off1 coming from img_w.
    """
    B, _, Hh, Wh = offset_map.shape
    device = offset_map.device

    # coords = (x,y) for the centre of every cell (same resolution as map)
    y, x = torch.meshgrid(
        torch.arange(Hh, device=device),
        torch.arange(Wh, device=device),
        indexing='ij'
    )
    coords = torch.stack((x, y), 0).float()                # (2,H,W)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)        # (B,2,H,W)

    src_pts = coords + offset_map                      # (B,2,H,W)
    src_pts_f = torch.cat((src_pts.view(B, 2, -1),
                           torch.ones(B, 1, Hh*Wh, device=device)), 1)
    coords_f = torch.cat((coords.view(B, 2, -1),
                          torch.ones(B, 1, Hh*Wh, device=device)), 1)

    dst_pts = _apply_homography(src_pts_f, H)             # (B,2,N)
    dst_ctr = _apply_homography(coords_f,  H)             # (B,2,N)

    warped_off = (dst_pts - dst_ctr).view(B, 2, Hh, Wh)    # (B,2,H,W)
    return warped_off
