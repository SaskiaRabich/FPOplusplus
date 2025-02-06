# The code was adapted from Yu et al. (https://github.com/sxyu/plenoctree), published under the following license:

# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import svox
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm, trange
import yaml
from absl import flags
import imageio

import datasets

def to_dataformat(format, args):
    if format == "FC":
        if len(args) != 3:
            print("FC data format needs 3 arguments, in order fc_dim_sigma, fc_dim_rgb, sh_dim")
            return False, ""
        return True, "FC" + str(args[0]) + "_" + str(args[1]) + "_" + str(args[2])
    elif format == "LFC":
        if len(args) != 3:
            print("LFC data format needs 3 arguments, in order fc_dim_sigma, fc_dim_rgb, sh_dim")
            return False, ""
        return True, "LFC" + str(args[0]) + "_" + str(args[1]) + "_" + str(args[2])
    else:
        print("Data format unknown")
        return False, ""

def readPlenOctrees(time_steps, time_skip, time_offset, train_dir, tree_name):
    trees = []
    for ts in range(int(time_steps)):
        idx = ts * time_skip + time_offset
        print("Loading PlenOctree for time step "+str(ts+1)+ " of "+str(time_steps)+", index "+str(idx), end = '\r')
        if os.path.exists(os.path.join(train_dir, str(idx), "octrees", tree_name + ".npz")):
            t = svox.N3Tree.load(os.path.join(train_dir, str(idx), "octrees", tree_name + ".npz"), device="cpu")
            trees.append(t)
        else: 
            print()
            print("PlenOctree for time step "+str(idx)+" not found")
            return []
    print()
    if len(trees) == 0:
        print("No PlenOctrees loaded")
    print("Loaded "+str(len(trees))+" PlenOctrees")
    return trees

def createFPO(trees,data_format,time_steps=1,augment_time=False,device="cpu"):
    #get data from exemplary tree of the scene
    radius = 0.5/trees[0].invradius.clone().detach()
    center = (1.0 - 2.0*trees[0].offset.clone().detach())*radius
    #saving centers of all trees as extra data
    extra_data = torch.zeros((1,len(trees)*3))
    for ts in np.arange(time_steps):
        r = 0.5/trees[ts].invradius.clone().detach()
        c = (1.0 - 2.0*trees[ts].offset.clone().detach())*r
        extra_data[0][3*ts+0] = c[0]
        extra_data[0][3*ts+1] = c[1]
        extra_data[0][3*ts+2] = c[2]
    fpo = svox.N3DynamicTree(  N=trees[0].N,
                        #data_dim=data_dim, 
                        init_refine=0,
                        init_reserve=1100000,
                        geom_resize_fact=trees[0].geom_resize_fact,
                        depth_limit=trees[0].depth_limit,
                        radius=radius,
                        center=center,
                        data_format=data_format,
                        extra_data=extra_data,
                        device=device,
                        augmented_time=augment_time,
                        time_steps=time_steps)
    print('Created FPO: ',fpo)
    return fpo

def translateToTimestep(pts,bboxes,timestep):
    c = bboxes[0][timestep*3:timestep*3+3]
    c_ref = bboxes[0][0:3]
    pts_projected = pts - c_ref + c
    return pts_projected

def translateToRefBox(pts,bboxes,timestep):
    c = bboxes[0][timestep*3:timestep*3+3]
    c_ref = bboxes[0][0:3]
    pts_projected = pts - c + c_ref
    return pts_projected

def component_dependent_encoding(input,n):
    ts = input.shape[1]
    coeffs = (n+1)/2
    fact = ts/coeffs
    
    avg = torch.mean(input[...,-1:],dim=1,keepdim=True)
    shift = torch.where(torch.any(input[...,-1:] <= 0.,dim=1).unsqueeze(2),avg,torch.zeros(avg.shape).to(avg.device))
    
    p = ((input[...,-1:] - shift) * fact) + shift
    input[...,-1:] = p
    return input


def define_flags():
    """Define flags for both training and evaluation modes."""
    flags.DEFINE_string("train_dir", None, "where to store ckpts and logs")
    flags.DEFINE_string("data_dir", None, "input data directory.")
    flags.DEFINE_string("config", None, "using config files to set hyperparameters.")

    # Dataset Flags
    flags.DEFINE_enum(
        "dataset",
        "nhr",
        list(k for k in datasets.dataset_dict.keys()),
        "The type of dataset to feed to nerf.",
    )
    flags.DEFINE_bool(
        "white_bkgd",
        True,
        "using white color as default background.",
    )
    flags.DEFINE_bool(
        "black_bkgd",
        False,
        "using black color as default background." "(used in the blender dataset only)",
    )
    flags.DEFINE_integer(
        "factor", 0, "the downsample factor of images, 0 for no downsample."
    )

    # Octree flags
    flags.DEFINE_float(
        'renderer_step_size',
        1e-4,
        'step size epsilon in volume render.'
        '1e-3 = fast setting, 1e-4 = usual setting, 1e-5 = high setting, lower is better')
    flags.DEFINE_bool(
        'no_early_stop',
        False,
        'If set, does not use early stopping; slows down rendering slightly')
    flags.DEFINE_bool(
        'load_data_batchwise',
        False,
        'only load data necessary for next few iterations if else there would be too much data to load'
    )
    flags.DEFINE_integer(
        "time_steps",
        60,
        "number of time steps to create FPO from",
    )
    flags.DEFINE_integer(
        "time_skip",
        1,
        "Time steps to skip in the training dataset if dynamic",
    )
    flags.DEFINE_integer(
        "time_offset",
        0,
        "Time steps to skip in the beginning of the training dataset if dynamic",
    )


def update_flags(args):
    """Update the flags in `args` with the contents of the config YAML file."""
    if args.config is None:
        return
    pth = os.path.join(args.config + ".yaml")
    with open_file(pth, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Only allow args to be updated if they already exist.
    invalid_args = list(set(configs.keys()) - set(dir(args)))
    if invalid_args:
        raise ValueError(f"Invalid args {invalid_args} in {pth}.")
    args.__dict__.update(configs)

    if args.time_skip <1:
        args.time_skip = 1
    if args.time_offset <0:
        args.time_offset =0

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def open_file(pth, mode="r"):
    pth = os.path.expanduser(pth)
    return open(pth, mode=mode)

def compute_psnr(mse):
    """Compute psnr value given mse (we assume the maximum pixel value is 1).

    Args:
      mse: float, mean square error of pixels.

    Returns:
      psnr: float, the psnr value.
    """
    return -10.0 * torch.log(mse) / np.log(10.0)

def compute_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim

def eval_fpo(t, dataset, args, want_lpips=True, want_frames=False):
    import svox
    w, h, focal = dataset.w, dataset.h, dataset.focal
    ndc_config = None

    bg = 0.0 if args.black_bkgd else 1.0
    r = svox.VolumeRendererDynamic(
        t, step_size=args.renderer_step_size, ndc=ndc_config, time_steps=args.time_steps,background_brightness=bg)

    print('Evaluating Fourier PlenOctree')
    device = t.data.device
    if want_lpips:
        import lpips
        lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    avg_mae_sl = 0.0
    out_frames = []
    for ts in tqdm(range(dataset.time_steps), desc= 'Time steps'):
        psnr_pertimestep=0.
        for idx in tqdm(range(dataset.size), desc='Images',leave=False):
            if args.moving_cameras:
                c2w = torch.from_numpy(dataset.camtoworlds[ts][idx]).float().to(device)
            else:  
                c2w = torch.from_numpy(dataset.camtoworlds[idx]).float().to(device)
            if hasattr(dataset, 'intrinsics'):
                K = torch.from_numpy(dataset.intrinsics[idx]).float().to(device)
            if args.dataset == 'nhr':
                ref_idx, id = dataset.image_idx[idx]
                im_gt_ten = torch.from_numpy(dataset.images[ts][ref_idx][id]).float().to(device)
            else:
                im_gt_ten = torch.from_numpy(dataset.images[ts][idx]).float().to(device)

            if hasattr(dataset, 'intrinsics'):
                if args.dataset == 'nhr':
                    im = r.render_persp_from_K(
                        c2w, K, width=w[ref_idx], height=h[ref_idx], fast=not args.no_early_stop, timestep=ts)
                else:
                    im = r.render_persp_from_K(
                        c2w, K, width=w, height=h, fast=not args.no_early_stop, timestep=ts)
            else:
                im = r.render_persp(
                    c2w, width=w, height=h, fx=focal, fast=not args.no_early_stop, timestep=ts)
            im.clamp_(0.0, 1.0)

            im_sl = torch.where(torch.any(im < 1.,-1,keepdim=True),1.,0.)
            im_gt_sl = torch.where(torch.any(im_gt_ten < 1.,-1,keepdim=True),1.,0.)
            mae_sl = (torch.abs(im_sl - im_gt_sl)).mean()
            avg_mae_sl += mae_sl.item()

            mse = ((im - im_gt_ten) ** 2).mean()
            psnr = compute_psnr(mse).mean()
            ssim = compute_ssim(im, im_gt_ten, max_val=1.0).mean()

            psnr_pertimestep += psnr.item()

            avg_psnr += psnr.item()
            avg_ssim += ssim.item()
            if want_lpips:
                lpips_i = lpips_vgg(im_gt_ten.permute([2, 0, 1]).contiguous(),
                        im.permute([2, 0, 1]).contiguous(), normalize=True)
                avg_lpips += lpips_i.item()

            if want_frames:
                im = im.cpu()
                # vis = np.hstack((im_gt_ten.cpu().numpy(), im.cpu().numpy()))
                vis = im.cpu().numpy()  # for lpips calculation
                vis = (vis * 255).astype(np.uint8)
                out_frames.append(vis)

        psnr_pertimestep /= dataset.size
        #print("TS:",ts,": PSNR", psnr_pertimestep)

    avg_psnr /= dataset.size * dataset.time_steps
    avg_ssim /= dataset.size * dataset.time_steps
    avg_lpips /= dataset.size * dataset.time_steps
    avg_mae_sl /= dataset.size * dataset.time_steps
    print("MAE Silhouette: ", avg_mae_sl)
    return avg_psnr, avg_ssim, avg_lpips, out_frames

def eval_fpo_batchwise(t, dataset, args, want_lpips=True, want_frames=False):
    import svox
    w, h, focal = dataset.w, dataset.h, dataset.focal
    ndc_config = None

    bg = 0.0 if args.black_bkgd else 1.0
    r = svox.VolumeRendererDynamic(
        t, step_size=args.renderer_step_size, ndc=ndc_config, time_steps=args.time_steps,background_brightness=bg)

    print('Evaluating Fourier PlenOctree')
    device = t.data.device
    if want_lpips:
        import lpips
        lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    out_frames = []

    permut = np.arange(args.time_steps*dataset.size)

    batches = 1+(int)(args.time_steps*dataset.size / 1024)
    filler = 1024 - (int)((args.time_steps*dataset.size) % 1024)
    add_to = np.full(filler, -1, dtype=int)
    permut = np.concatenate((permut,add_to))

    if hasattr(dataset, 'intrinsics'):
        K0 = dataset.intrinsics
    else:
        K0 = None
    if hasattr(dataset,'image_idx'):
        img_idx0 = dataset.image_idx
    else:
        img_idx0 = None

    for b in range(batches):
        print(f"Batch {b+1}/{batches}")
        p = permut[b*1024:(b+1)*1024]
        images, h, w, extra = datasets.load_next_data_batch("test", p, dataset.size, args, intr= K0, idx_list = img_idx0)
        if args.dataset == 'nhr':
            img_idx = extra
        elif args.dataset == "synthetichuman" and args.moving_cameras:
            c2ws = extra
        
        print("Evaluate batch")
        for e in tqdm(range(len(p))):
            k = p[e] #index for timestep and offset/img_index in timestep
            if k == -1:
                continue
            ts = (int)(k / dataset.size)
            idx = k % dataset.size
            if args.moving_cameras:
                c2w = torch.from_numpy(c2ws[e]).float().to(device)
            else:  
                c2w = torch.from_numpy(dataset.camtoworlds[idx]).float().to(device)
            if hasattr(dataset, 'intrinsics'):
                K = torch.from_numpy(dataset.intrinsics[idx]).float().to(device)
            if args.dataset == 'nhr':
                ref_idx, id = img_idx[e]
                im_gt_ten = torch.from_numpy(images[ref_idx][id]).float().to(device)
            else:
                im_gt_ten = torch.from_numpy(images[e]).float().to(device)

            if hasattr(dataset, 'intrinsics'):
                if args.dataset == 'nhr':
                    im = r.render_persp_from_K(
                        c2w, K, width=w[ref_idx], height=h[ref_idx], fast=not args.no_early_stop, timestep=ts)
                else:
                    im = r.render_persp_from_K(
                        c2w, K, width=w, height=h, fast=not args.no_early_stop, timestep=ts)
            else:
                im = r.render_persp(
                    c2w, width=w, height=h, fx=focal, fast=not args.no_early_stop, timestep=ts)
            im.clamp_(0.0, 1.0)

            mse = ((im - im_gt_ten) ** 2).mean()
            psnr = compute_psnr(mse).mean()
            ssim = compute_ssim(im, im_gt_ten, max_val=1.0).mean()

            avg_psnr += psnr.item()
            avg_ssim += ssim.item()
            if want_lpips:
                lpips_i = lpips_vgg(im_gt_ten.permute([2, 0, 1]).contiguous(),
                        im.permute([2, 0, 1]).contiguous(), normalize=True)
                avg_lpips += lpips_i.item()

            if want_frames:
                im = im.cpu()
                # vis = np.hstack((im_gt_ten.cpu().numpy(), im.cpu().numpy()))
                vis = im.cpu().numpy()  # for lpips calculation
                vis = (vis * 255).astype(np.uint8)
                out_frames.append(vis)

    avg_psnr /= dataset.size * dataset.time_steps
    avg_ssim /= dataset.size * dataset.time_steps
    avg_lpips /= dataset.size * dataset.time_steps
    return avg_psnr, avg_ssim, avg_lpips, out_frames


def checkForInvalidEntries(tree, device="cpu"):
    print("*** Checking for invalid entries")
    print("Generating grid")
    # makes grid with centers of each cube
    reso = 2 ** (tree.depth_limit + 1)
    offset = tree.offset.cpu()
    scale = tree.invradius.cpu()
    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T.contiguous()
    print("grid.shape: ",grid.shape)
    # delete points outside of fpo
    print("Fitting grid to tree")
    chunk = int(grid.shape[0]/1024)
    for i in tqdm(range(0,grid.shape[0],chunk)):
        torch.cuda.empty_cache()
        grid_tmp = grid[i:i+chunk].to(device)
        mask = tree[grid_tmp].depths == tree.max_depth
        grid_tmp = grid_tmp[mask]
        if i == 0:
            grid_fitted = grid_tmp.cpu()
        else:
            grid_fitted = torch.cat([grid_fitted,grid_tmp.cpu()])
    del grid_tmp, grid
    print("grid.shape: ",grid_fitted.shape)
    grid = grid_fitted.to(device)
    del grid_fitted
    tree=tree.to(device)

    chunk = int(grid.shape[0]/1024)
    found_invalid_entries_nan=False
    found_invalid_entries_inf=False
    for i in tqdm(range(0,grid.shape[0],chunk)):
        torch.cuda.empty_cache()
        grid_tmp = grid[i:i+chunk]
        if tree.extra_data is not None:
            entries = tree[translateToTimestep(grid_tmp,tree.extra_data,0)].values
        else:
            entries = tree[grid_tmp].values
        if torch.any(torch.isnan(entries)):
            found_invalid_entries_nan = True
        if torch.any(torch.isinf(entries)):
            found_invalid_entries_inf = True
        del entries
    if found_invalid_entries_nan:
        print("*** Found nan entries!")
    if found_invalid_entries_inf:
        print("*** Found inf entries!")
    if not found_invalid_entries_inf and not found_invalid_entries_nan:
        print("*** No invalid entries found in tree")