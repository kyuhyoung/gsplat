import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_test_data(
    data_path: Optional[str] = None,
    device="cuda",
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),
    scene_grid: int = 1,
):
    """Load the test data."""
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz")
    data = np.load(data_path)
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    '''
    print(f'viewmats.shape : {viewmats.shape}');    # (3, 4, 4)
    print(f'viewmats[0] : \n{viewmats[0]}') #   T : (-0.0254, 0.2270, 1.1955) 
    print(f'viewmats[1] : \n{viewmats[1]}') #   T : (-0.0437, 0.3014, 1.1271)
    print(f'viewmats[2] : \n{viewmats[2]}');#   T : (0.0260,  0,1988, 1.0237)   exit(1) 
    '''
    Ks = torch.from_numpy(data["Ks"]).float().to(device)
    '''
    print(f'width : {width}, height : {height}')    #   648, 420
    print(f'Ks.shape : {Ks.shape}');  #exit(1)    #  (3, 3, 3)
    print(f'Ks[0] : \n{Ks[0]}') #   T : (-0.0254, 0.2270, 1.1955) 
    print(f'Ks[1] : \n{Ks[1]}') #   T : (-0.0254, 0.2270, 1.1955) 
    print(f'Ks[2] : \n{Ks[2]}') #   T : (-0.0254, 0.2270, 1.1955)
    #   all Ks are the same as following
    #   fx : 480.6123, fy : 481.5445, cx : 324.1875, cy : 210.0625    exit(1)
    '''
    means = torch.from_numpy(data["means3d"]).float().to(device)
    #print(f'means.shape : {means.shape}, means.min() : {means.min()}, means.max() : {means.max()}');  exit(1)    #  (138776, 3),    -12.039, 14.624
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)
    C = len(viewmats)

    # crop
    aabb = torch.tensor(scene_crop, device=device)
    edges = aabb[3:] - aabb[:3]
    #print(f'aabb : {aabb}, edges : {edges}'); exit(1)   #   [-2, -2, -2, 2, 2, 2], [4, 4, 4]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, colors = means[sel], colors[sel]

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    #print(f'means.shape : {means.shape}, means.min() : {means.min()}, means.max() : {means.max()}');  #exit(1)    #  (1, 111785, 3),    -1.9999, 1.9999
    means = means.reshape(-1, 3)
    #print(f'means.shape : {means.shape}');  #  (111785, 3)
    #print(f'colors.shape b4 : {colors.shape}')  #   (111785, 3)
    colors = colors.repeat(repeats**2, 1)
    #print(f'repeats : {repeats}, colors.shape after : {colors.shape}'); # 1, (111785, 3)    exit(1)

    # create gaussian attributes
    N = len(means)
    scales = torch.rand((N, 3), device=device) * 0.02
    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)
    opacities = torch.rand((N,), device=device)

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height
