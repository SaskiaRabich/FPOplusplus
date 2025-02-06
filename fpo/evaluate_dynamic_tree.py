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

import torch
import numpy as np
import os
from absl import app
from absl import flags
from tqdm import tqdm
import imageio

import utils
import datasets

import svox

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "input",
    "./tree_opt.npz",
    "Input octree npz from optimization.py",
)
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)
flags.DEFINE_string(
    "write_images",
    None,
    "If specified, writes images to given path (*.png)",
)
flags.DEFINE_bool(
    'moving_cameras',
    False,
    'if cameras of different time steps have different extrinsics'
)

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)

    print('N3DynamicTree load', FLAGS.input)
    fpo = svox.N3DynamicTree.load(FLAGS.input, device=device)
    print("Loaded: ",fpo)

    if FLAGS.load_data_batchwise:
        avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_fpo_batchwise(fpo, dataset, FLAGS,
                want_lpips=True,
                want_frames=FLAGS.write_vid is not None or FLAGS.write_images is not None)
    else:
        avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_fpo(fpo, dataset, FLAGS,
                want_lpips=True,
                want_frames=FLAGS.write_vid is not None or FLAGS.write_images is not None)
    print('Average PSNR', avg_psnr, 'SSIM', avg_ssim, 'LPIPS', avg_lpips)

    if FLAGS.write_vid is not None and len(out_frames):
        print('Writing to', FLAGS.write_vid)
        imageio.mimwrite(FLAGS.write_vid, out_frames)

    if FLAGS.write_images is not None and len(out_frames):
        print('Writing to', FLAGS.write_images)
        os.makedirs(FLAGS.write_images, exist_ok=True)
        for idx, frame in tqdm(enumerate(out_frames)):
            ts = (int)(idx / dataset.size)
            num = idx % dataset.size
            imageio.imwrite(os.path.join(FLAGS.write_images, f"{num:03d}_ts{ts}.png"), frame)

if __name__ == "__main__":
    app.run(main)
