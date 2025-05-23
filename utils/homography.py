import torch
import random
import kornia as K


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
