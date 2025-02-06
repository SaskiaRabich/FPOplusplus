import svox
import torch
import numpy as np
import os
from tqdm import tqdm, trange

from absl import app
from absl import flags

from utils import * 

device = 'cuda:0'

args = flags.FLAGS

define_flags()

flags.DEFINE_string(
    "output",
    None,
    "directory to save the FPO",
)
flags.DEFINE_string(
    "tree_name",
    "tree",
    "name of octrees in subfolders for time steps",
)
flags.DEFINE_string(
    "save_name",
    "fpo",
    "name of Fourier PlenOctree",
)
flags.DEFINE_integer(
    "batch_num_refine",
    2048,
    "number of batches for refining the FPO structure",
)
flags.DEFINE_integer(
    "batch_num_fill",
    2048,
    "number of batches for filling the FPO structure",
)
flags.DEFINE_string(
    "data_format",
    "LFC",
    "data format of the dynamic PlenOctree: FC (use Fourier Coefficients), LFC (use Fourier Coefficients with logarithmic encoding)"
)
flags.DEFINE_integer(
    "sh_dim",
    9,
    "number of Spherical Harmonics coefficients",
)
flags.DEFINE_integer(
    "fc_dim_sigma",
    31,
    "number of Fourier coefficients for density",
)
flags.DEFINE_integer(
    "fc_dim_rgb",
    5,
    "number of Fourier coefficients for color",
)
flags.DEFINE_bool(
    "comp_enc",
    True,
    "enable component-dependent encoding on fourier coefficients",
)
flags.DEFINE_bool(
    "augment_time",
    True,
    "augment time steps to avoid artifacts at first and last time step (only FC and LFC)",
)
flags.DEFINE_bool(
    "checkpoint",
    False,
    "save a checkpoint of the refined octree structure before computing Fourier coefficients",
)
flags.DEFINE_bool(
    "no_move_trees",
    True,
    "do not move trees to GPU for Fourier coefficient computation (slower, use if runs out of memory)",
)

set_random_seed(0)

def refineStructure(fpo, trees):
    check = len(trees)
    print("Generating grid")
    # makes grid with centers of each cube
    reso = 2 ** (fpo.depth_limit + 1)
    offset = fpo.offset.cpu()
    scale = fpo.invradius.cpu()
    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T.contiguous()
    print("grid.shape: ",grid.shape)

    for t in range(len(trees)):
        torch.cuda.empty_cache()
        print("Processing tree "+str(t+1)+" of "+str(len(trees)), end = '')
        tree = trees[t]
        tree = tree.to(device)
        chunk = int(grid.shape[0]/args.batch_num_refine)
        print(", in "+ str(args.batch_num_refine)+ " chunks of size "+ str(chunk))
        for i in tqdm(range(0,grid.shape[0],chunk), leave=False):
            torch.cuda.empty_cache()
            grid_tmp = grid[i:i+chunk].to(device)
            grid_tmp_ts = translateToTimestep(grid_tmp,fpo.extra_data,t)
            fpo_depths = fpo[grid_tmp].depths
            tree_depths = tree[grid_tmp_ts].depths
            mask = fpo_depths < tree_depths
            del fpo_depths, tree_depths
            for i in range(fpo.depth_limit - 1):
                torch.cuda.empty_cache()
                grid_tmp = grid_tmp[mask]
                grid_tmp_ts = grid_tmp_ts[mask]
                del mask
                fpo[grid_tmp].refine()
                fpo_depths = fpo[grid_tmp].depths
                tree_depths = tree[grid_tmp_ts].depths
                mask = fpo_depths < tree_depths
                del fpo_depths, tree_depths
            refine_chunk = 2000000
            if grid_tmp[mask].shape[0] <= refine_chunk:
                fpo[grid_tmp[mask]].refine()
            else:
                # Do last layer separately
                grid_tmp = grid_tmp[mask].cpu()
                for j in range(0, grid_tmp.shape[0], refine_chunk):
                    fpo[grid_tmp[j:j+refine_chunk].to(device)].refine()
            del grid_tmp, grid_tmp_ts, mask
        tree = tree.to("cpu") # free memory on GPU
        torch.cuda.empty_cache()
        print("Current FPO: ", fpo)
    assert check == len(trees)
    print('finished')
    return

def fill(fpo, trees):
    print("Generating grid")
    # makes grid with centers of each cube
    reso = 2 ** (fpo.depth_limit + 1)
    offset = fpo.offset.cpu()
    scale = fpo.invradius.cpu()
    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T.contiguous()
    print("grid.shape: ",grid.shape)
    # delete points outside of fpo
    print("Fitting grid to FPO")
    chunk = int(grid.shape[0]/args.batch_num_fill)
    for i in tqdm(range(0,grid.shape[0],chunk)):
        torch.cuda.empty_cache()
        grid_tmp = grid[i:i+chunk].to(device)
        mask = fpo[grid_tmp].depths == fpo.max_depth
        grid_tmp = grid_tmp[mask]
        if i == 0:
            grid_fitted = grid_tmp.cpu()
        else:
            grid_fitted = torch.cat([grid_fitted,grid_tmp.cpu()])
    del grid_tmp, grid
    print("grid.shape: ",grid_fitted.shape)

    if not args.no_move_trees:
        grid = grid_fitted.to(device)
    else:
        grid = grid_fitted
    del grid_fitted
    print('Filling FPO')
    if not args.no_move_trees:
        trees = [trees[t].to(device) for t in range(len(trees))]

    if args.augment_time:
        ts = len(trees) + 2
    else:
        ts = len(trees)
    conv = svox.FourierConverter.from_DataFormat(fpo.data_format,ts, device=device)

    working_device = device if not args.no_move_trees else "cpu"
    chunk = int(grid.shape[0]/args.batch_num_fill)
    for i in tqdm(range(0,grid.shape[0],chunk)):
        torch.cuda.empty_cache()
        grid_tmp = grid[i:i+chunk].to(working_device)
        sh_coeff = trees[0][translateToTimestep(grid_tmp,fpo.extra_data.to(working_device),0)].values.to(device).unsqueeze(1)
        for t in range(1,len(trees)):
            sh_coeff = torch.cat([sh_coeff, trees[t][translateToTimestep(grid_tmp,fpo.extra_data.to(working_device),t)].values.to(device).unsqueeze(1)], dim=1)
        if args.augment_time:
            sh_coeff = torch.cat([sh_coeff[:,:1,:],sh_coeff,sh_coeff[:,-1:,:]], dim = 1)
        
        
        if args.data_format == "LFC":
            # transform density with ln(sigma + 1)
            sh_coeff[...,-1:] = torch.add(sh_coeff[...,-1:],1)
            sh_coeff[...,-1:] = torch.log(sh_coeff[...,-1:])
            if args.comp_enc:
                sh_coeff = component_dependent_encoding(sh_coeff,conv.n1)
            coeff = conv.sh2fourier(sh_coeff)
        else: #data_format == "FC"
            if args.comp_enc:
                sh_coeff = component_dependent_encoding(sh_coeff,conv.n1)
            coeff = conv.sh2fourier(sh_coeff)
        fpo[grid_tmp.to(device)] = coeff

    print('finished')
    return

def main(_argv):
    os.makedirs(args.output, exist_ok=True)

    f = os.path.join(args.output, 'args_{}_construction.txt'.format(args.save_name))
    with open(f, 'w') as file:
        for attr,flag_obj in args.__flags.items():
            file.write('{} = {}\n'.format(attr, flag_obj.value))
        file.write('\nupdated dict:\n')
        for key in list(args.__dict__.keys()):
            if key[:2] == "__": continue
            file.write('{} = {}\n'.format(key, args.__dict__[key]))

    print('* Read in SH-PlenOctrees')
    trees = readPlenOctrees(args.time_steps, args.time_skip, args.time_offset, args.train_dir, args.tree_name)
    if len(trees) == 0: 
        print("exiting")
        quit()
    
    #Step 1: Build structure to fill with Wavelet coefficients or read in existing structure
    if os.path.exists(os.path.join(args.output, args.save_name + "_ckpt.npz")):
        print('* Read in existing FPO, skipping refinement')
        fpo = svox.N3DynamicTree.load(os.path.join(args.output, args.save_name+ "_ckpt.npz"), device=device)
        print('Read refined FPO: ',fpo)
    else:
        print('* Create Dynamic PlenOctree')
        if args.data_format == "LFC":
            valid, format = to_dataformat("LFC", [args.fc_dim_sigma,args.fc_dim_rgb, args.sh_dim])
        else: #data_format == "FC"
            valid, format = to_dataformat("FC", [args.fc_dim_sigma,args.fc_dim_rgb, args.sh_dim])
        
        if valid:
            print("Data format: ",format)
            fpo = createFPO(trees, format, time_steps=args.time_steps,augment_time=args.augment_time, device=device)
        else:
            print("Data format invalid, exiting")
            quit()

        print('* Build FPO structure')
        with torch.no_grad():
            refineStructure(fpo, trees)
        if args.checkpoint:
            fpo.cpu().save(os.path.join(args.output, args.save_name + "_ckpt"), compress=False)
            print('Saved checkpoint of refined tree structure')
        fpo = fpo.to(device)

    #Step 2: Calculate Wavelet coefficients and fill FPO
    print('* Fill FPO values')
    with torch.no_grad():
        fill(fpo, trees)

    print('* Save FPO')
    print(fpo)
    os.makedirs(args.output, exist_ok=True)
    fpo.cpu().save(os.path.join(args.output, args.save_name), compress=False)
    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
