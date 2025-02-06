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

import numpy as np
import os
import json
import cv2
from PIL import Image
from tqdm import tqdm

import utils

def get_dataset(split, args):
    return dataset_dict[args.dataset](split, args)

def load_next_data_batch(split, to_load, n_imgs, args, intr= None, idx_list = None):
    return dataset_dict[args.dataset].load_next_data(split, to_load, n_imgs, args, intr, idx_list)


class Dataset():
    """Dataset Base Class."""

    def __init__(self, split, args, prefetch=True):
        super(Dataset, self).__init__()
        self.split = split
        self._general_init(args)

    @property
    def size(self):
        return self.n_examples

    def _general_init(self, args):
        bbox_path = os.path.join(args.data_dir, 'bbox.txt')
        if os.path.isfile(bbox_path):
            self.bbox = np.loadtxt(bbox_path)[:-1]
        else:
            self.bbox = None
        self._load_renderings(args)


class BlenderDynamic(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        with utils.open_file(
            os.path.join(args.data_dir, "transforms_{}.json".format(self.split)), "r"
        ) as fp:
            meta = json.load(fp)
        print(' Load Blender', args.data_dir, 'split', self.split)

        # load cams
        cams = []
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            cams.append(frame["transform_matrix"])

        # load images for all timesteps
        images_dyn = []
        if args.load_data_batchwise:
            ts_to_load = [0]
        else:
            ts_to_load = range(args.time_steps)
        for ts in tqdm(ts_to_load):
            idx = args.time_offset + args.time_skip * ts
            images = []
            for i in range(len(meta["frames"])):
                frame = meta["frames"][i]
                if "file_name" in frame:
                    file_name = frame['file_name']
                else:
                    file_name = frame['file_path'].split('/')[1]
                fname = os.path.join(args.data_dir, str(idx), file_name + ".png")
                with utils.open_file(fname, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                    if args.factor == 2:
                        [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                        image = cv2.resize(
                            image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                        )
                    elif args.factor > 0:
                        raise ValueError(
                            "Blender dataset only supports factor=0 or 2, {} "
                            "set.".format(args.factor)
                        )
                if args.white_bkgd and image.shape[-1] == 4:
                    mask = image[..., -1:]
                    image = image[..., :3] * mask + (1.0 - mask)
                elif args.black_bkgd and image.shape[-1] == 4:
                    mask = image[..., -1:]
                    image = image[..., :3] * mask
                else:
                    image = image[..., :3]
                images.append(image)
            images_stckd = np.stack(images, axis=0)
            images_dyn.append(images_stckd)

        self.images = images_dyn
        self.h, self.w = self.images[0].shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
        self.n_examples = self.images[0].shape[0]
        self.time_steps = args.time_steps

    @classmethod
    def load_next_data(cls, split, to_load, n_imgs, args, intr = None, idx_list = None):
        """Load images from disk."""
        with utils.open_file(
            os.path.join(args.data_dir, "transforms_{}.json".format(split)), "r"
        ) as fp:
            meta = json.load(fp)
        print(' Load Blender', args.data_dir, 'split', split)

        # load images
        images = []
        for im in tqdm(to_load):
            if im == -1:
                continue
            ts = (int)(im/n_imgs)
            i = (int)(im%n_imgs)
            idx = args.time_offset + args.time_skip * ts
            
            frame = meta["frames"][i]
            if "file_name" in frame:
                file_name = frame['file_name']
            else:
                file_name = frame['file_path'].split('/')[1]
            fname = os.path.join(args.data_dir, str(idx), file_name + ".png")
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "Blender dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            if args.white_bkgd and image.shape[-1] == 4:
                mask = image[..., -1:]
                image = image[..., :3] * mask + (1.0 - mask)
            elif args.black_bkgd and image.shape[-1] == 4:
                mask = image[..., -1:]
                image = image[..., :3] * mask
            else:
                image = image[..., :3]
            images.append(image)
        h, w = images[0].shape[:2]
        return np.stack(images, axis=0), h, w, None

class SyntheticHuman(Dataset):
    """SyntheticHuman Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        print(' Load SyntheticHuman', args.data_dir, 'split', self.split)

        num_cams = len(next(os.walk(args.data_dir))[1])

        if self.split == "train":
            cams_to_load = np.delete(np.arange(num_cams),np.arange(0,num_cams,5))
        elif self.split == "test":
            cams_to_load = np.arange(0,num_cams,5)

        images_dyn = []
        cams_dyn = []

        if args.load_data_batchwise:
            ts_to_load = [0]
        else:
            ts_to_load = range(args.time_steps)
        for ts in tqdm(ts_to_load):
            images = []
            cams = []

            for cam in cams_to_load:
                meta={}
                with open(os.path.join(args.data_dir, "cam"+str(cam), 'transforms.json'), 'r') as fp:
                    meta = json.load(fp)
                frame = meta["frames"][args.time_offset + ts*args.time_skip]
                f_name = frame["file_path"].split('/')[-1]
                f_num = ""
                for c in f_name:
                    if not c.isalpha():
                        f_num += c
                f_name = "frame_"+f_num
                fname = os.path.join(args.data_dir, "cam"+str(cam), f_name + '.png')
                with utils.open_file(fname, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                    if args.factor == 2:
                        [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                        image = cv2.resize(
                            image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                        )
                    elif args.factor > 0:
                        raise ValueError(
                            "SyntheticHuman dataset only supports factor=0 or 2, {} "
                            "set.".format(args.factor)
                        )
                cams.append(frame["transform_matrix"])
                if args.white_bkgd and image.shape[-1] == 4:
                    mask = image[..., -1:]
                    image = image[..., :3] * mask + (1.0 - mask)
                else:
                    image = image[..., :3]
                images.append(image)
            images_dyn.append(np.stack(images,axis=0))
            cams_dyn.append(np.stack(cams,axis=0).astype(np.float32))
        
        self.images = images_dyn
        self.h, self.w = self.images[0].shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = cams_dyn
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
        self.n_examples = self.images[0].shape[0]
        self.time_steps = args.time_steps

        args.moving_cameras=True

    @classmethod
    def load_next_data(cls, split, to_load, n_imgs, args, intr = None, idx_list = None):
        """Load images from disk."""
        print(' Load SyntheticHuman', args.data_dir, 'split', split)

        num_cams = len(next(os.walk(args.data_dir))[1])

        if split == "train":
            cams_to_load = np.delete(np.arange(num_cams),np.arange(0,num_cams,5))
        elif split == "test":
            cams_to_load = np.arange(0,num_cams,5)

        images = []
        cams = []

        for im in tqdm(to_load):
            if im == -1:
                continue
            ts = (int)(im/n_imgs)
            i = (int)(im%n_imgs)
            cam = cams_to_load[i]

            meta={}
            with open(os.path.join(args.data_dir, "cam"+str(cam), 'transforms.json'), 'r') as fp:
                meta = json.load(fp)
            frame = meta["frames"][args.time_offset + ts*args.time_skip]
            f_name = frame["file_path"].split('/')[-1]
            f_num = ""
            for c in f_name:
                if not c.isalpha():
                    f_num += c
            f_name = "frame_"+f_num
            fname = os.path.join(args.data_dir, "cam"+str(cam), f_name + '.png')
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "SyntheticHuman dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            cams.append(frame["transform_matrix"])
            if args.white_bkgd and image.shape[-1] == 4:
                mask = image[..., -1:]
                image = image[..., :3] * mask + (1.0 - mask)
            else:
                image = image[..., :3]
            images.append(image)
        h,w = images[0].shape[:2]
        return np.stack(images,axis=0), h, w, np.stack(cams,axis=0).astype(np.float32)

class NHR(Dataset):
    """nhr Dataset."""

    def campose_to_extrinsic(self,camposes):
        if camposes.shape[1] != 12:
            raise Exception(" wrong campose data structure!")
        
        Ts = np.zeros((camposes.shape[0],4,4))
        
        Ts[:, :3, 2] = camposes[:, 0:3]
        Ts[:, :3, 0] = camposes[:, 3:6]
        Ts[:, :3, 1] = camposes[:, 6:9]
        Ts[:, :3, 3] = camposes[:, 9:12]
        Ts[:, 3, 3] = 1.0
        
        return Ts

    def read_intrinsics(self,fn_instrinsic):
        with open(fn_instrinsic, mode="r") as fp:
            data= fp.readlines()

        i = 0
        Ks = []
        while i<len(data):
            if len(data[i])>5:
                tmp = data[i].split()
                tmp = [float(i) for i in tmp]
                a = np.array(tmp)
                i = i+1
                tmp = data[i].split()
                tmp = [float(i) for i in tmp]
                b = np.array(tmp)
                i = i+1
                tmp = data[i].split()
                tmp = [float(i) for i in tmp]
                c = np.array(tmp)
                res = np.vstack([a,b,c])
                Ks.append(res)

            i = i+1
        Ks = np.stack(Ks)

        return Ks

    def load_extrinsic_intrinsics(self,data_path):

        rot = np.array([[1,0, 0,0],
                        [0,0,-1,0],
                        [0,1, 0,0],
                        [0,0, 0,1]],dtype=np.float)

        camposes = np.loadtxt(os.path.join(data_path, 'CamPose.inf'))
        Ts = np.matmul(rot,np.array(self.campose_to_extrinsic(camposes)))
        Ks = np.array(self.read_intrinsics(os.path.join(data_path, 'Intrinsic.inf')))

        return Ks, Ts

    def _load_renderings(self, args):
        """Load images from disk."""

        intrinsics, poses = self.load_extrinsic_intrinsics(args.data_dir)
        image_idx = []
        images_dyn = []
        ref_intr1 = None
        ref_intr2 = None

        #find ref intrinsics to distinguish different images
        for i in np.arange(intrinsics.shape[0]):
            if args.factor == 2:
                intrinsics[i] = intrinsics[i] / 2.
                intrinsics[i,2,2] = 1.
            if ref_intr1 is None:
                ref_intr1 = intrinsics[i]
            if ref_intr1 is not None and ref_intr2 is None:
                if (ref_intr1 != intrinsics[i]).any():
                    ref_intr2 = intrinsics[i]
                    break
        num_cams = len([name for name in os.listdir(os.path.join(args.data_dir,"img",str(0))) if os.path.isfile(os.path.join(args.data_dir,"img",str(0),name))])

        if num_cams > 60:
            i_test = np.array([2,7,14,21,37,43,54])
        else: 
            i_test = np.array([7,14,21,37,43,54])
        i_train = np.delete(np.arange(num_cams),i_test)

        print(' Load NHR', args.data_dir, 'split', self.split)

        if self.split == "train":
            cams_to_load = i_train
            intrinsics = intrinsics[cams_to_load]
            poses = poses[cams_to_load]
        elif self.split == "test":
            cams_to_load = i_test
            intrinsics = intrinsics[cams_to_load]
            poses = poses[cams_to_load]

        if args.load_data_batchwise:
            ts_to_load = [0]
        else:
            ts_to_load = range(args.time_steps)
        for ts in tqdm(ts_to_load):
            images = [[],[]]
            idx = args.time_offset + ts*args.time_skip
            for i in np.arange(len(cams_to_load)):
                cam = cams_to_load[i]

                if (intrinsics[i] == ref_intr1).all():
                    ref_ind = 0
                elif (intrinsics[i] == ref_intr2).all():
                    ref_ind = 1
                else:
                    print("unknown intrinsic found")
                    ref_ind=-1
                id = len(images[ref_ind])
                if ts == 0:
                    image_idx.append((ref_ind, id))

                n = "img_{:04d}.jpg".format(cam)
                fname = os.path.join(args.data_dir, "img",str(idx), n)
                with utils.open_file(fname, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                    if args.factor == 2:
                        [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                        image = cv2.resize(
                            image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                        )
                    elif args.factor > 0:
                        raise ValueError(
                            "NHR dataset only supports factor=0 or 2, {} "
                            "set.".format(args.factor)
                        )
                if args.white_bkgd:
                    fname = os.path.join(args.data_dir, "img",str(idx), "mask",n)
                    with utils.open_file(fname, "rb") as imgin:
                        mask = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                        if args.factor == 2:
                            [halfres_h, halfres_w] = [hw // 2 for hw in mask.shape[:2]]
                            mask = cv2.resize(
                                mask, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                            )
                        elif args.factor > 0:
                            raise ValueError(
                                "NHR dataset only supports factor=0 or 2, {} "
                                "set.".format(args.factor)
                            )
                    image = image[...,:3]*mask[...,:3]+(1. - mask[...,:3])
                else:
                    image = image[...,3]
                images[ref_ind].append(image)
            images_dyn.append([np.stack(images[0], axis=0),np.stack(images[1], axis=0)])

        self.images = images_dyn
        self.h = [self.images[0][0].shape[1],self.images[0][1].shape[1]]
        self.w = [self.images[0][0].shape[2],self.images[0][1].shape[2]]
        self.resolution = [self.h[0] * self.w[0],self.h[1] * self.w[1]]
        self.camtoworlds = poses
        self.focal = None
        self.image_idx = image_idx
        self.intrinsics = intrinsics
        self.n_examples = self.images[0][0].shape[0] + self.images[0][1].shape[0]
        self.time_steps = args.time_steps
    
    @classmethod
    def load_next_data(cls, split, to_load, n_imgs, args, intr = None, idx_list = None):
        """Load images from disk."""

        image_idx = []
        ref_intr1 = None
        ref_intr2 = None

        #find reference intrinsics to distinguish different images
        for i in np.arange(intr.shape[0]):
            if ref_intr1 is None and idx_list[i][0]==0:
                ref_intr1 = intr[i]
            elif ref_intr1 is None and idx_list[i][0]==1:
                ref_intr2 = intr[i]
            if ref_intr1 is not None and ref_intr2 is None:
                if (ref_intr1 != intr[i]).any():
                    ref_intr2 = intr[i]
                    break
            elif ref_intr2 is not None and ref_intr1 is None:
                if (ref_intr2 != intr[i]).any():
                    ref_intr1 = intr[i]
                    break
        num_cams = len([name for name in os.listdir(os.path.join(args.data_dir,"img",str(0))) if os.path.isfile(os.path.join(args.data_dir,"img",str(0),name))])

        if num_cams > 60:
            i_test = np.array([2,7,14,21,37,43,54])
        else: 
            i_test = np.array([7,14,21,37,43,54])
        i_train = np.delete(np.arange(num_cams),i_test)

        print(' Load NHR', args.data_dir, 'split', split)

        if split == "train":
            cams_to_load = i_train
        elif split == "test":
            cams_to_load = i_test

        images = [[],[]]
        for im in tqdm(to_load):
            if im == -1:
                continue
            ts = (int)(im/n_imgs)
            i = (int)(im%n_imgs)
            cam = cams_to_load[i]
            idx = args.time_offset + ts*args.time_skip

            if (intr[i] == ref_intr1).all():
                ref_ind = 0
            elif (intr[i] == ref_intr2).all():
                ref_ind = 1
            else:
                print("unknown intrinsic found")
                ref_ind=-1
            id = len(images[ref_ind])
            image_idx.append((ref_ind, id))

            n = "img_{:04d}.jpg".format(cam)
            fname = os.path.join(args.data_dir, "img",str(idx), n)
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "NHR dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            if args.white_bkgd:
                fname = os.path.join(args.data_dir, "img",str(idx), "mask",n)
                with utils.open_file(fname, "rb") as imgin:
                    mask = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                    if args.factor == 2:
                        [halfres_h, halfres_w] = [hw // 2 for hw in mask.shape[:2]]
                        mask = cv2.resize(
                            mask, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                        )
                    elif args.factor > 0:
                        raise ValueError(
                            "NHR dataset only supports factor=0 or 2, {} "
                            "set.".format(args.factor)
                        )
                image = image[...,:3]*mask[...,:3]+(1. - mask[...,:3])
            else:
                image = image[...,3]
            images[ref_ind].append(image)
        images[0] = np.stack(images[0], axis=0)
        images[1] = np.stack(images[1], axis=0)
        
        h = [images[0].shape[1],images[1].shape[1]]
        w = [images[0].shape[2],images[1].shape[2]]
        return [images[0],images[1]], h, w, image_idx


dataset_dict = {
    "blender_dynamic": BlenderDynamic,
    "synthetichuman": SyntheticHuman,
    "nhr": NHR,
}