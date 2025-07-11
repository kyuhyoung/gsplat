import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat.exporter import sh2rgb
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

import os
from plyfile import PlyData
import numpy as np
import torch



# Experimental
def construct_list_of_attributes(splats):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(splats["sh0"].shape[1] * splats["sh0"].shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(splats["shN"].shape[1] * splats["shN"].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(splats["scales"].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(splats["quats"].shape[1]):
        l.append('rot_{}'.format(i))
    return l

@torch.no_grad()
def load_ply(path: str, dev) -> None:
    plydata = PlyData.read(path)
    v = plydata['vertex'].data  # numpy structured array
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)            # (N,3)
    print(f'xyz.shape : {xyz.shape}, xyz.min() : {xyz.min()}, xyz.max() : {xyz.max()}');  exit(1)    #  (138776, 3),    -12.039, 14.624
    N = len(xyz)
    print(f'xyz.shape : {xyz.shape}');  exit(1)
    
    normals = np.stack([v['nx'], v['ny'], v['nz']], axis=1)     # (N,3)
    rgb_uint8 = np.stack([v['red'], v['green'], v['blue']], axis=1)  # (N,3) uint8
    # 3) 필요하다면 [0,255] → [0,1] 로 스케일
    rgb = rgb_uint8.astype(np.float32) / 255.0                  # (N,3) float32
    # 4) torch tensor 로 변환
    xyz_t = torch.tensor(xyz, dtype = torch.float, device = dev)       # float32
    #normals_t = torch.from_numpy(normals.astype(np.float32), device = dev)
    rgb_t = torch.tensor(rgb, dtype = torch.float, device = dev)
    scales_t = torch.rand((N, 3), device = dev) * 0.02
    quats_t = F.normalize(torch.randn((N, 4), device = dev), dim=-1)
    opacities_t = torch.rand((N,), device = dev)
    
    return xyz_t, quats_t, scales_t, opacities_t, rgb_t



# Within the same class as save_ply
@torch.no_grad()
def load_gs_ply(path: str, dev) -> None:
    plydata = PlyData.read(path)


    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    n_gauss = xyz.shape[0]; 
    features_dc = np.zeros((n_gauss, 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    #print(f'features_dc.min() : {features_dc.min()}, features_dc.max() : {features_dc.max()}');    exit(1) #   -1.918, 37.269

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    #print(f'opacities.shape : {opacities.shape}, opacities.min() : {opacities.min()}, opacities.max() : {opacities.max()}');    exit(1)
    #   (11237741, 1),  -5.3, 17.253
    #opacities *= 0.5 * 0.5 * 0.5
    opacities = rand((n_gauss,), device = dev)
    #opacities = opacities.clip(0, 1)

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    if extra_f_names:
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        Tfeatures_rest = torch.tensor(features_extra, dtype = torch.float, device = dev).transpose(1, 2).contiguous()
    else:
        Tfeatures_rest = None
    
    #skale_scale = 1.0 / 4.0
    skale_scale = 1.0
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name]) * skale_scale

    #print(f'scales.min() : {scales.min()}, scales.max() : {scales.max()}');    exit(1) #   -27.74, 15.6
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #print(f'rots.shape : {rots.shape}, rots.min() : {rots.min()}, rots.max() : {rots.max()}');    exit(1) 
    #   (11237741, 3), -10.47, 11.795

    Txyz = torch.tensor(xyz, dtype = torch.float, device = dev)
    #Tfeatures_dc = torch.tensor(features_dc, dtype = torch.float, device = dev).transpose(1, 2).contiguous()
    Tfeatures_dc = torch.tensor(features_dc, dtype = torch.float, device = dev).squeeze().contiguous()
    Topacity = torch.tensor(opacities, dtype = torch.float, device = dev).squeeze()
    Tscaling = torch.tensor(scales, dtype = torch.float, device = dev)
    Trotation = torch.tensor(rots, dtype = torch.float, device = dev)
    #Tactive_sh_degree = self.max_sh_degree
    #print(f'Txyz.shape : {Txyz.shape}');    exit(1) #   (11237741, 3)
    #print(f'Txyz.min() : {Txyz.min()}, Txyz.max() : {Txyz.max()}');    exit(1) #   -190981.2, 52918
    #print(f'Trotation.shape : {Trotation.shape}');    exit(1) #   (11237741, 4)
    #print(f'Tscaling.shape : {Tscaling.shape}');    exit(1) #   (11237741, 3)
    #print(f'Topacity.shape : {Topacity.shape}');    #exit(1) #   (112377413)
    #print(f'Tfeatures_dc.shape : {Tfeatures_dc.shape}');    exit(1) #   (112377413, 3)
    #print(f'Tfeatures_rest : {Tfeatures_rest}');    exit(1) #   None
    return Txyz, Trotation, Tscaling, Topacity, Tfeatures_dc, Tfeatures_rest, 

def gen_dummy_cam(dev):
    skale = 1.0 / 4.0
    #viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    width = int(skale * 11310);  height = int(skale * 17310);
    kay = np.zeros((3, 3), dtype = float);
    kay[0, 0] = skale * 16644.23;   kay[1, 1] = skale * 16644.23;
    kay[0, 2] = skale * 5655;       kay[1, 2] = skale * 8655;   kay[2, 2] = 1;   
    Ks = torch.from_numpy(kay[None, :, :]).float().to(dev)
    vm = np.zeros((4, 4), dtype = float);
    vm[0, 0] = 1.1024e-3;   vm[0, 1] = -9.9998e-1;  vm[0, 2] = 4.678e-3
    vm[1, 0] = -1.0000e+0;  vm[1, 1] = -1.1036e-3;  vm[1, 2] = 6.4805e-5
    vm[2, 0] = -5.9645e-5;  vm[2, 1] = -4.6782e-3;  vm[2, 2] = -9.9999e-1
    vm[0, 3] = -250.5565 + 300;   
    vm[1, 3] = 766.53515 - 1000;   
    vm[2, 3] = 1373.0609;
    vm[3, 3] = 1.0
    '''
    print(f'vm b4 : \n{vm}')
    vm = np.linalg.inv(vm)
    print(f'vm after : \n{vm}');    exit(1)
    '''
    #vm[0, 3] *= -1; vm[1, 3] *= -1;  vm[2, 3] *= -1
    vm[:3, :3] = vm[:3, :3].T
    viewmats = torch.from_numpy(vm[None, :, :]).float().to(dev)
    print(f'viewmats.shape : {viewmats.shape}');    #exit(1)    (1, 3, 3)
    print(f'viewmats[0] : \n{viewmats[0]}');    #exit(1)
    return viewmats, Ks, width, height 

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ply:
        #means, quats, scales, opacities, f_dc, f_rest = load_gs_ply(args.ply, device); colors = sh2rgb(f_dc).clip(0, 1)
        means, quats, scales, opacities, colors = load_ply(args.ply, device);
        viewmats, Ks, width, height = gen_dummy_cam(device)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()
        
        sh_degree = None
        #sh_degree = 1
        #sh_degree = 3
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)
        #for cam_model in ['ortho', 'pinhole', 'fisheye']:
        for cam_model in ['pinhole']:
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                #render_mode="RGB+D",
                render_mode="RGB",
                packed=False,
                distributed=world_size > 1,
                camera_model = cam_model,
                sh_degree = sh_degree
            )

            C = render_colors.shape[0]
            #assert render_colors.shape == (C, height, width, 4)
            assert render_colors.shape == (C, height, width, 3)
            assert render_alphas.shape == (C, height, width, 1)
            render_colors.sum().backward()

            render_rgbs = render_colors[..., 0:3]
            #render_depths = render_colors[..., 3:4]
            #render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(args.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{args.output_dir}/render_rank_{world_rank}_{cam_model}.png",
                (canvas * 255).astype(np.uint8),
            )







    elif args.gs_ply:
        #means, quats, scales, opacities, f_dc, f_rest = load_gs_ply(args.ply, device); colors = sh2rgb(f_dc).clip(0, 1)
        means, quats, scales, opacities, colors = load_gs_ply(args.gs_ply, device); colors = sh2rgb(f_dc).clip(0, 1)
        viewmats, Ks, width, height = gen_dummy_cam(device)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()
        
        sh_degree = None
        #sh_degree = 1
        #sh_degree = 3
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)
        #for cam_model in ['ortho', 'pinhole', 'fisheye']:
        for cam_model in ['pinhole']:
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                #render_mode="RGB+D",
                render_mode="RGB",
                packed=False,
                distributed=world_size > 1,
                camera_model = cam_model,
                sh_degree = sh_degree
            )

            C = render_colors.shape[0]
            #assert render_colors.shape == (C, height, width, 4)
            assert render_colors.shape == (C, height, width, 3)
            assert render_alphas.shape == (C, height, width, 1)
            render_colors.sum().backward()

            render_rgbs = render_colors[..., 0:3]
            #render_depths = render_colors[..., 3:4]
            #render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(args.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{args.output_dir}/render_rank_{world_rank}_{cam_model}.png",
                (canvas * 255).astype(np.uint8),
            )




    elif args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()
        
        sh_degree = None
        #sh_degree = 3
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)
        for cam_model in ['pinhole', 'fisheye', 'ortho']:
        #for cam_model in ['ortho', 'pinhole', 'fisheye']:
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                #render_mode="RGB+D",
                render_mode="RGB",
                packed=False,
                distributed=world_size > 1,
                camera_model = cam_model,
                sh_degree = sh_degree
            )

            C = render_colors.shape[0]
            #assert render_colors.shape == (C, height, width, 4)
            assert render_colors.shape == (C, height, width, 3)
            assert render_alphas.shape == (C, height, width, 1)
            #render_colors.sum().backward()

            render_rgbs = render_colors[..., 0:3]
            #render_depths = render_colors[..., 3:4]
            #render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(args.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{args.output_dir}/render_rank_{world_rank}_{cam_model}.png",
                (canvas * 255).astype(np.uint8),
            )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        
        #for ckpt_path in args.ckpt:
        t0 = torch.load(args.ckpt[0], map_location=device)
        '''
        print(f'type(t0) : {type(t0)}');    #exit(1)
        print(f't0.keys() : {t0.keys()}');    # step, splats, camtoworlds, Ks, pixels, near_plane, far_plane, sh_degree  
        print(f'type(t0["step"]) : {type(t0["step"])}');    #exit(1)
        print(f't0["step"] : {t0["step"]}');    exit(1)
        '''
        camtoworlds = t0["camtoworlds"]
        Ks = t0["Ks"]
        print(f'Ks : {Ks}');
        print(f'camtoworlds : {camtoworlds}');  #exit(1)
        pixels = t0["pixels"]
        height, width = pixels[0].shape[:2]
        print(f'width : {width}, height : {height}');   #exit(1)
        near_plane = t0["near_plane"]
        far_plane = t0["far_plane"]
        sh_degree = t0["sh_degree"]
        print(f'camtoworlds.shape : {camtoworlds.shape}');  #exit(1) #   1868859, 3
        print(f'Ks.shape : {Ks.shape}');  #exit(1) #   1868859, 3
        print(f'pixels.shape : {pixels.shape}');  #exit(1) #   1868859, 3
        #print(f'opacities.shape : {opacities.shape}');  #exit(1) #   1868859, 3



        #ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        ckpt = t0["splats"]
        print(f'type(ckpt) : {type(ckpt)}');    #exit(1)
        #print(f'ckpt.items() : {ckpt.items()}');    exit(1)
        #means.append(ckpt["means"])
        means = ckpt["means"]
        #quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)
        #scales.append(torch.exp(ckpt["scales"]))
        scales = torch.exp(ckpt["scales"])
        #opacities.append(torch.sigmoid(ckpt["opacities"]))
        opacities = torch.sigmoid(ckpt["opacities"])
        #sh0.append(ckpt["sh0"])
        sh0 = ckpt["sh0"]
        #shN.append(ckpt["shN"])
        shN = ckpt["shN"]
        print(f'means.shape : {means.shape}');  #exit(1) #   1868859, 3
        print(f'quats.shape : {quats.shape}');  #exit(1) #   1868859, 3
        print(f'scales.shape : {scales.shape}');  #exit(1) #   1868859, 3
        print(f'opacities.shape : {opacities.shape}');  #exit(1) #   1868859, 3

        #means = torch.cat(means, dim=0)
        #quats = torch.cat(quats, dim=0)
        #scales = torch.cat(scales, dim=0)
        #opacities = torch.cat(opacities, dim=0)
        opacities = opacities
        #sh0 = torch.cat(sh0, dim=0)
        sh0 = sh0
        #shN = torch.cat(shN, dim=0)
        shN = shN
        colors = torch.cat([sh0, shN], dim=-2)
        #sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

        c2w = camtoworlds.float().to(device) 
        K = Ks.float().to(device)
        #c2w = torch.from_numpy(c2w).float().to(device)
        #K = torch.from_numpy(K).float().to(device)
        viewmats = c2w.inverse()

        #for cam_model in ['ortho', 'pinhole', 'fisheye']:
        for cam_model in ['pinhole', 'fisheye', 'ortho']:
        #for cam_model in ['pinhole', 'fisheye']:
        #for cam_model in ['pinhole']:
        #for cam_model in ['ortho']:
            print(f'processing {cam_model} camera model')
            #print(f'means.shape : {means.shape}');  exit(1)
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                #render_mode="RGB+D",
                render_mode="RGB",
                packed=False,
                distributed=world_size > 1,
                camera_model = cam_model,
                sh_degree = sh_degree
            )

            C = render_colors.shape[0]
            #assert render_colors.shape == (C, height, width, 4)
            assert render_colors.shape == (C, height, width, 3)
            assert render_alphas.shape == (C, height, width, 1)
            #render_colors.sum().backward()

            render_rgbs = render_colors[..., 0:3]
            #render_depths = render_colors[..., 3:4]
            #render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(args.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    #render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    #render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{args.output_dir}/render_rank_{world_rank}_{cam_model}.png",
                (canvas * 255).astype(np.uint8),
            )
        exit(1)


        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders








    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gs_ply", type = str, help = "3DGS ply file trained from others"
    )
    parser.add_argument(
        "--ply", type = str, help = "Regular ply file trained from others"
    )

    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
