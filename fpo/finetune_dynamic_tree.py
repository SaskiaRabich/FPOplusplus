# The code was adapted from Yu et al. (https://github.com/sxyu/plenoctree), published under the following license:

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import svox
import torch
from torch.optim import SGD, Adam
import numpy as np
import os
import imageio
from tqdm import tqdm, trange

from absl import app
from absl import flags

import utils
import datasets

device = 'cuda:0'

args = flags.FLAGS
utils.define_flags()

flags.DEFINE_string(
    "tree_name",
    "fpo",
    "name of FPO to finetune",
)
flags.DEFINE_string(
    "save_name",
    "fpo_opt",
    "name of finetuned FPO",
)
flags.DEFINE_bool(
    "log_mse",
    True,
    "Whether to log mse (and psnr) values druing training and evaluation",
)
flags.DEFINE_string(
    "log_file_train",
    "mse_train.txt",
    "where to store mse training values",
)
flags.DEFINE_string(
    "log_file_test",
    "mse_test.txt",
    "where to store mse testing values",
)
flags.DEFINE_integer(
    "num_epochs",
    10,
    "number of epochs for FPO finetuning",
)
flags.DEFINE_bool(
    "randomize_trainset",
    True,
    "If set, randomizes training set per epoch",
)
flags.DEFINE_bool(
    'sgd',
    False,
    'use SGD optimizer instead of Adam'
)
flags.DEFINE_float(
    'sgd_momentum',
    0.0,
    'sgd momentum'
)
flags.DEFINE_bool(
    'sgd_nesterov',
    False,
    'sgd nesterov momentum?'
)
flags.DEFINE_float(
    'lr',
    1e-2,
    'optimizer step size'
)
flags.DEFINE_integer(
    'render_interval',
    0,
    'render interval'
)
flags.DEFINE_integer(
    'val_interval',
    2,
    'validation interval'
)
flags.DEFINE_bool(
    "continue_on_decrease",
    False,
    "If set, continues training even if validation PSNR decreases",
)
flags.DEFINE_bool(
    'moving_cameras',
    False,
    'if cameras of different time steps have different extrinsics'
)
flags.DEFINE_integer(
    'save_checkpoints',
    10,
    'Save refined tree every x epochs, 0 for disable'
)
flags.DEFINE_bool(
    "multi_tensor_data",
    True,
    "separate data and index data in different tensors for storage efficiency",
)



def main(_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(args)

    print('* Read in Dataset')
    def get_data(stage):
        assert stage in ["train", "val", "test"]
        dataset = datasets.get_dataset(stage, args)
        focal = dataset.focal
        all_c2w = dataset.camtoworlds
        all_gt = []
        if args.dataset == 'nhr':
            dataset.images[0][0]= dataset.images[0][0].reshape(-1, dataset.h[0], dataset.w[0], 3)
            dataset.images[0][1]= dataset.images[0][1].reshape(-1, dataset.h[1], dataset.w[1], 3)
            all_gt.append([torch.from_numpy(dataset.images[0][0]).float(),torch.from_numpy(dataset.images[0][1]).float()])
        else:
            dataset.images[0] = dataset.images[0].reshape(-1, dataset.h, dataset.w, 3)
            all_gt.append(torch.from_numpy(dataset.images[0]).float())
        if args.randomize_trainset:
            if args.moving_cameras:
                all_c2w_tmp = []
                all_c2w_tmp.append(torch.from_numpy(all_c2w[0]).float())
                all_c2w = all_c2w_tmp
            else:
                all_c2w = torch.from_numpy(all_c2w).float()
        else:
            if args.moving_cameras:
                all_c2w_tmp = []
                all_c2w_tmp.append(torch.from_numpy(all_c2w[0]).float().to(device))
                all_c2w = all_c2w_tmp
            else:
                all_c2w = torch.from_numpy(all_c2w).float().to(device)
        all_K = None
        if hasattr(dataset, 'intrinsics'):
            all_K = dataset.intrinsics
            all_K = torch.from_numpy(all_K).float().to(device)
        img_idx = None
        if hasattr(dataset, 'image_idx'):
            img_idx = dataset.image_idx
        return focal, all_c2w, all_gt, all_K, img_idx

    focal, train_c2w, train_gt, train_K, train_img_idx = get_data("train")
    
    test_focal, test_c2w, test_gt, test_K, test_img_idx = get_data("test")
    assert focal == test_focal

    if args.dataset == 'nhr':
        H = [train_gt[0][0][0].shape[0],train_gt[0][1][0].shape[0]]
        W = [train_gt[0][0][0].shape[1],train_gt[0][1][0].shape[1]]
    else:
        H, W = train_gt[0][0].shape[:2]

    if args.render_interval > 0:
        vis_dir = os.path.splitext(args.input)[0] + '_render'
        os.makedirs(vis_dir, exist_ok=True)

    if args.randomize_trainset:
        if args.moving_cameras:
            test_c2w_tmp = []
            test_c2w_tmp.append(test_c2w[0].to(device))
            test_c2w = test_c2w_tmp
        else:
            test_c2w = test_c2w.to(device)

    ndc_config = None

    print('* Read in Dynamic PlenOctree')
    fpo = svox.N3DynamicTree.load(os.path.join(args.train_dir, args.tree_name+".npz"), device=device)
    bg = 0.0 if args.black_bkgd else 1.0
    r = svox.VolumeRendererDynamic(fpo, step_size=args.renderer_step_size, ndc=ndc_config, time_steps=args.time_steps,background_brightness=bg)
    print('Loaded ',fpo)
    print(fpo.data.grad)

    print('* Start finetuning')
    
    if args.sgd:
        print('Using SGD, lr', args.lr)
        if args.lr < 1.0:
            print('For SGD please adjust LR to about 1e7')
        optimizer = SGD(fpo.parameters(), lr=args.lr, momentum=args.sgd_momentum,
                        nesterov=args.sgd_nesterov)
    else:
        adam_eps = 1e-4 if fpo.data.dtype is torch.float16 else 1e-8
        print('Using Adam, eps', adam_eps, 'lr', args.lr)
        optimizer = Adam(fpo.parameters(), lr=args.lr, eps=adam_eps)

    if args.moving_cameras:
        n_train_imgs = len(train_c2w[0])
        n_test_imgs = len(test_c2w[0])
    else:
        n_train_imgs = len(train_c2w)
        n_test_imgs = len(test_c2w)

    f = os.path.join(args.train_dir, 'args_{}_optimization.txt'.format(args.save_name))
    with open(f, 'w') as file:
        for attr,flag_obj in args.__flags.items():
            file.write('{} = {}\n'.format(attr, flag_obj.value))
        file.write('\nupdated dict:\n')
        for key in list(args.__dict__.keys()):
            if key[:2] == "__": continue
            file.write('{} = {}\n'.format(key, args.__dict__[key]))

    if args.log_mse:
        f1 = open(os.path.join(args.train_dir,args.log_file_train), 'w')
        f2 = open(os.path.join(args.train_dir,args.log_file_test), 'w')
        f1.write("# image mse psnr \n")
        f2.write("# image mse psnr \n")
        f2.close()

    def run_test_step(i):
        print('Evaluating')
        with torch.no_grad():
            if args.log_mse:
                f2 = open(os.path.join(args.train_dir,args.log_file_test), 'a')
                f2.write('# test step '+str(i)+'\n')
            tpsnr = 0.0
            tase = 0.0
            permut = np.arange(args.time_steps*n_test_imgs)
            batches = 1+(int)(args.time_steps*n_test_imgs / 512)
            filler = 512 - (int)((args.time_steps*n_test_imgs) % 512)
            add_to = np.full(filler, -1, dtype=int)
            permut = np.concatenate((permut,add_to))
            print()

            for b in range(batches):
                print(f"Batch {b+1}/{batches}")
                p = permut[b*512:(b+1)*512]
                test_gt, h, w, extra = datasets.load_next_data_batch("test", p, n_test_imgs, args, intr= test_K, idx_list = test_img_idx)
                if args.dataset == 'nhr':
                    test_gt0 = test_gt[0].reshape(-1, h[0], w[0], 3)
                    test_gt1 = test_gt[1].reshape(-1, h[1], w[1], 3)
                    test_gt = [torch.from_numpy(test_gt0).float(),torch.from_numpy(test_gt1).float()]
                else:
                    test_gt = test_gt.reshape(-1, h, w, 3)
                    test_gt = torch.from_numpy(test_gt).float()
                if args.dataset == 'nhr':
                    te_img_idx = extra
                if args.dataset == "synthetichuman" and args.moving_cameras:
                    test_poses = extra
                    test_poses = torch.from_numpy(test_poses).float()
                else:
                    test_poses = test_c2w

                print("Evaluate batch")
                for e in tqdm(range(len(p))):
                    k = p[e] #index for timestep and offset/img_index in timestep
                    if k == -1:
                        continue
                    torch.cuda.empty_cache()
                    ts = (int)(k / n_test_imgs)
                    j = k % n_test_imgs
                    if args.moving_cameras:
                        c2w = test_poses[e].clone().detach().to(device=device)
                    else:
                        c2w = test_poses[j].clone().detach().to(device=device)
                    if args.dataset == 'nhr':
                        idx,id = te_img_idx[e]
                        im_gt = test_gt[idx][id].clone().detach()
                    else:
                        im_gt = test_gt[e].clone().detach()
                    torch.cuda.empty_cache()

                    if test_K is not None:
                        if args.dataset == 'nhr':
                            im = r.render_persp_from_K(c2w, test_K[j],height=H[idx], width=W[idx], fast=False, timestep=ts)
                        else:
                            im = r.render_persp_from_K(c2w, test_K[j],height=H, width=W, fast=False, timestep=ts)
                    else:
                        im = r.render_persp(c2w, height=H, width=W, fx=focal, fast=False, timestep=ts)
                    im = im.cpu().clamp_(0.0, 1.0)
                    torch.cuda.empty_cache()

                    mse = ((im - im_gt) ** 2).mean()
                    psnr = -10.0 * np.log(mse) / np.log(10.0)
                    tpsnr += psnr.item()
                    if args.log_mse:
                        f2.write(str(ts*n_train_imgs + j) + ' ' + str(mse.item()) + ' ' + str(psnr.item()))
                        f2.write('\n')

                    if args.render_interval > 0 and j % args.render_interval == 0:
                        vis = torch.cat((im_gt, im), dim=1)
                        vis = (vis * 255).numpy().astype(np.uint8)
                        imageio.imwrite(f"{vis_dir}/{i:04}_{j:04}_{ts:02}.png", vis)
                    del im, im_gt, c2w
            tpsnr /= n_test_imgs*args.time_steps
            tase /= n_test_imgs*args.time_steps
            if args.log_mse:
                f2.close()
            return tpsnr

    best_validation_psnr = run_test_step(0)
    print('** initial val psnr ', best_validation_psnr)
    best_t = None
    overfit_counter = 0
    for i in range(args.num_epochs):
        print('epoch', i)
        tpsnr = 0.0
        tase = 0.0
        if args.log_mse:
            f1.write('# epoch '+str(i)+'\n')
        if args.randomize_trainset:
            permut = np.random.permutation(args.time_steps*n_train_imgs)
        else:
            permut = np.arange(args.time_steps*n_train_imgs)

        batches = 1+(int)(args.time_steps*n_train_imgs / 1024)
        filler = 1024 - (int)((args.time_steps*n_train_imgs) % 1024)
        add_to = np.full(filler, -1, dtype=int)
        permut = np.concatenate((permut,add_to))

        for b in range(batches):
            print(f"Batch {b+1}/{batches}")
            p = permut[b*1024:(b+1)*1024]
            train_gt, h, w, extra = datasets.load_next_data_batch("train", p, n_train_imgs, args, intr= train_K, idx_list = train_img_idx)
            if args.dataset == 'nhr':
                train_gt0 = train_gt[0].reshape(-1, h[0], w[0], 3)
                train_gt1 = train_gt[1].reshape(-1, h[1], w[1], 3)
                train_gt = [torch.from_numpy(train_gt0).float(),torch.from_numpy(train_gt1).float()]
            else:
                train_gt = train_gt.reshape(-1, h, w, 3)
                train_gt = torch.from_numpy(train_gt).float()
            if args.dataset == 'nhr':
                tr_img_idx = extra
            elif args.dataset == "synthetichuman" and args.moving_cameras:
                train_c2w = extra
                train_c2w = torch.from_numpy(train_c2w).float()

            print("Train batch")
            for e in tqdm(range(len(p))):
                k = p[e] #index for timestep and offset/img_index in timestep
                if k == -1:
                    continue
                torch.cuda.empty_cache()
                ts = (int)(k / n_train_imgs)
                j = k % n_train_imgs
                if args.moving_cameras:
                    c2w = train_c2w[e].clone().detach().to(device=device)
                else:
                    c2w = train_c2w[j].clone().detach().to(device=device)
                if args.dataset == 'nhr':
                    idx,id = tr_img_idx[e]
                    im_gt = train_gt[idx][id].clone().detach()
                else:
                    im_gt = train_gt[e].clone().detach()
                torch.cuda.empty_cache()
                if train_K is not None:
                    if args.dataset == 'nhr':
                        im = r.render_persp_from_K(c2w, train_K[j], height=H[idx], width=W[idx], cuda=True, timestep=ts)
                    else:
                        im = r.render_persp_from_K(c2w, train_K[j], height=H, width=W, cuda=True, timestep=ts)
                else:
                    im = r.render_persp(c2w, height=H, width=W, fx=focal, cuda=True, timestep=ts)
                del c2w
                im_gt_ten = im_gt.to(device=device)
                im = torch.clamp(im, 0.0, 1.0)
                mse = ((im - im_gt_ten) ** 2).mean()
                im_gt_ten = None
                del im_gt_ten, im_gt, im
                torch.cuda.empty_cache()

                optimizer.zero_grad()
                fpo.data.grad = None 
                torch.cuda.empty_cache()
                mse.backward()
                optimizer.step()
                psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)
                if args.log_mse:
                    f1.write(str(ts*n_train_imgs + j) + ' ' + str(mse.item()) + ' ' + str(psnr.item()))
                    f1.write('\n')
                tpsnr += psnr.item()
        tpsnr /= n_train_imgs * args.time_steps
        tase /= n_train_imgs * args.time_steps
        print('** train_psnr', tpsnr)

        if i % args.val_interval == args.val_interval - 1 or i == args.num_epochs - 1:
            validation_psnr = run_test_step(i + 1)
            print('** val psnr ', validation_psnr, 'best', best_validation_psnr)
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                best_t = fpo.clone(device='cpu')  # SVOX 0.2.22
                print('')
                overfit_counter = 0
            elif not args.continue_on_decrease:
                if overfit_counter > 2:
                    print('Stop since overfitting')
                    break
                else:
                    overfit_counter += 1
            if (args.save_checkpoints>0 and best_t is not None and i%args.save_checkpoints == args.save_checkpoints-1) and i < args.num_epochs-1:
                print('Saving checkpoint to', os.path.join(args.train_dir,args.save_name+'_ckpt_'+str(i+1)))
                best_t.save(os.path.join(args.train_dir,args.save_name+'_ckpt_'+str(i+1)), compress=False)
    if args.log_mse:
        f1.close()
    print('finished')

    print('* Save finetuned FPO')
    if best_t is not None:
        print('Saving best model to', os.path.join(args.train_dir,args.save_name))
        best_t.save(os.path.join(args.train_dir,args.save_name), compress=False)
    else:
        print('Did not improve upon initial model')
        #fpo.save(os.path.join(args.train_dir,args.save_name+"_overfitted"), compress=False)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
