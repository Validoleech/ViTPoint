import torch
import kornia as K
import torch.nn.functional as F


def sample_homography(batch, height, width, device, scale=(0.8, 1.2),
                      ang=45, trans=0.1):
    """
    Returns H (B,3,3) and warped grid for grid_sample().
    """
    pts1 = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]],
                        dtype=torch.float, device=device)
    pts1 = pts1.unsqueeze(0).repeat(batch, 1, 1)  # [B,4,2]
    # random params
    a = torch.rand(batch, device=device)*ang/180*3.1416 - ang/360*3.1416
    s = torch.rand(batch, device=device)*(scale[1]-scale[0])+scale[0]
    t = (torch.rand(batch, 2, device=device)-0.5)*2*trans
    # build matrices
    cosa, sina = torch.cos(a), torch.sin(a)
    A = torch.stack([cosa*s, -sina*s, t[:, 0],
                     sina*s,  cosa*s, t[:, 1],
                     torch.zeros_like(a), torch.zeros_like(a), torch.ones_like(a)],
                    dim=-1).view(batch, 3, 3)
    # Generate grid
    grid = K.utils.create_meshgrid(
        height, width, normalized_coordinates=True).to(device)  # [1,H,W,2]
    warped_grid = K.geometry.transform.homography_warp(
        grid.repeat(batch, 1, 1, 1), A)
    return A, warped_grid


def warp_offsets(off, H):
    """
    Very light-weight offset-map warping through a dense grid generated
    from the homography H ∈ ℝ^{B×3×3}. Works for NCHW tensors.
    """
    B, _, Hh, Hw = off.shape
    device = off.device
    ys, xs = torch.linspace(-1, 1, Hh,
                            device=device), torch.linspace(-1, 1, Hw, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y, torch.ones_like(
        grid_x)], dim=-1)        # (H, W, 3)

    grid = grid.view(1, Hh, Hw, 3).repeat(
        B, 1, 1, 1)                            # (B, H, W, 3)
    # (B, H, W, 3)
    grid = grid @ H.transpose(1, 2)
    grid = grid / (grid[..., 2:3] + 1e-8)
    # (B, H, W, 2)
    grid = grid[..., :2]

    off_w = F.grid_sample(off, grid, align_corners=True)
    return off_w


def _apply_homography(pts, H):
    """
    pts :  [B,3,N]  – homogeneous coordinates
    H   :  [B,3,3]
    """
    warped = torch.bmm(H, pts)                             # [B,3,N]
    warped = warped / (warped[:, 2:3, :] + 1e-8)           # z-normalise
    return warped[:, :2, :]                                # drop z


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

    # --- build pixel grid ----------------------------------------------------
    #   coords = (x,y) for the centre of every cell (same resolution as map)
    y, x = torch.meshgrid(
        torch.arange(Hh, device=device),
        torch.arange(Wh, device=device),
        indexing='ij'
    )
    coords = torch.stack((x, y), 0).float()                # (2,H,W)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)        # (B,2,H,W)

    # ------------------------------------------------------------------------
    src_pts = coords + offset_map                      # (B,2,H,W)
    src_pts_f = torch.cat((src_pts.view(B, 2, -1),
                           torch.ones(B, 1, Hh*Wh, device=device)), 1)
    coords_f = torch.cat((coords.view(B, 2, -1),
                          torch.ones(B, 1, Hh*Wh, device=device)), 1)

    dst_pts = _apply_homography(src_pts_f, H)             # (B,2,N)
    dst_ctr = _apply_homography(coords_f,  H)             # (B,2,N)

    warped_off = (dst_pts - dst_ctr).view(B, 2, Hh, Wh)    # (B,2,H,W)
    return warped_off
