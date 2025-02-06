# The code was adapted from Yu et al. (https://github.com/sxyu/svox), published under the following license:

#  Copyright 2021 PlenOctree Authors.
#
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
"""
Sparse voxel N^3 tree
"""

import os.path as osp
import torch
import numpy as np
import math
from torch import nn, autograd
from svox.helpers import N3TreeView, DataFormat, LocalIndex, _get_c_extension, clip2float16
from svox.svox import N3Tree, _QueryVerticalFunction
from warnings import warn

_C = _get_c_extension()


class N3DynamicTree(N3Tree):
    """
    PyTorch :math:`N^3`-tree library with CUDA acceleration.
    By :math:`N^3`-tree we mean a 3D tree with branching factor N at each interior node,
    where :math:`N=2` is the familiar octree.

    .. warning::
        `nn.Parameters` can change size, which
        makes current optimizers invalid. If any :code:`refine(): or
        :code:`shrink_to_fit()` call returns True,
        or :code:`expand(), shrink()` is used,
        please re-make any optimizers
    """
    def __init__(self, N=2, data_dim=None, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="LFC",
            extra_data=None,
            device="cpu",
            dtype=torch.float32,
            map_location=None,
            time_steps = 1,
            augmented_time = False):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf (NEW in 0.2.28: optional if data_format other than RGBA is given).
                        If data_format = "RGBA" or empty, this defaults to 4.
        :param depth_limit: int maximum depth  of tree to stop branching/refining
                            Note that the root is at depth -1.
                            Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                            are depth 9. :code:`max_depth` applies to the same
                            depth values.
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
                            inital resolution will be :code:`[N^(init_refine + 1)]^3`.
                            initial max_depth will be init_refine.
        :param geom_resize_fact: float geometric resizing factor
        :param radius: float or list, 1/2 side length of cube (possibly in each dim)
        :param center: list center of space
        :param data_format: a string to indicate the data format. :code:`RGBA | SH# | SG# | ASG#`
        :param extra_data: extra data to include with tree
        :param device: str device to put data
        :param dtype: str tree data type, torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device (will override device and warn)
        :param time_steps: int number of time steps the tree should be able to represent
        :param augmented_time: bool if time sequence should be augmented with 2 additional steps to reduce artifacts

        """
        super().__init__(N=N, data_dim=data_dim, depth_limit=depth_limit,
            init_reserve=init_reserve, init_refine=init_refine, geom_resize_fact=geom_resize_fact,
            radius=radius, center=center,
            data_format=data_format,
            extra_data=extra_data,
            device=device,
            dtype=dtype,
            map_location=map_location)

        self.time_steps = time_steps
        self.augmented_time = augmented_time


    # Special Features
    def partial(self, data_sel=None, data_format=None, dtype=None, device=None):
        """
        Get partial tree with some of the data dimensions (channels)
        E.g. :code:`tree.partial(-1)` to get tree with data_dim 1 of last channel only

        :param data_sel: data channel selector, default is all channels
        :param data_format: data format for new tree, default is current format
        :param dtype: new data type, torch.float32 | torch.float64
        :param device: where to put result tree

        :return: partial N3DynamicTree (copy)
        """
        if device is None:
            device = self.data.device
        if data_sel is None:
            new_data_dim = self.data_dim
            sel_indices = None
        else:
            sel_indices = torch.arange(self.data_dim)[data_sel]
            if sel_indices.ndim == 0:
                sel_indices = sel_indices.unsqueeze(0)
            new_data_dim = sel_indices.numel()
        if dtype is None:
            dtype = self.data.dtype
        t2 = N3DynamicTree(N=self.N, data_dim=new_data_dim,
                data_format=data_format or str(self.data_format),
                depth_limit=self.depth_limit,
                geom_resize_fact=self.geom_resize_fact,
                dtype=dtype,
                device=device, time_steps=self.time_steps, augmented_time=self.augmented_time)
        def copy_to_device(x):
            return torch.empty(x.shape, dtype=x.dtype, device=device).copy_(x)
        t2.invradius = copy_to_device(self.invradius)
        t2.offset = copy_to_device(self.offset)
        t2.child = copy_to_device(self.child)
        t2.parent_depth = copy_to_device(self.parent_depth)
        t2._n_internal = copy_to_device(self._n_internal)
        t2._n_free = copy_to_device(self._n_free)
        if self.extra_data is not None:
            t2.extra_data = copy_to_device(self.extra_data)
        else:
            t2.extra_data = None
        t2.data_format = self.data_format
        if data_sel is None:
            t2.data = nn.Parameter(copy_to_device(self.data))
        else:
            t2.data = nn.Parameter(copy_to_device(self.data[..., sel_indices].contiguous()))
        return t2

    # Persistence
    def save(self, path, shrink=True, compress=True):
        """
        Save to from npz file

        :param path: npz path
        :param shrink: if True (default), applies shrink_to_fit before saving
        :param compress: whether to compress the npz; may be slow

        """
        if shrink:
            self.shrink_to_fit()
        self.data.data = clip2float16(self.data.data)
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child.cpu(),
            "parent_depth" : self.parent_depth.cpu(),
            "n_internal" : self._n_internal.cpu().item(),
            "n_free" : self._n_free.cpu().item(),
            "invradius3" : self.invradius.cpu(),
            "offset" : self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "time_steps": self.time_steps,
            "data": self.data.data.half().cpu().numpy(),  # save CPU Memory
        }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if hasattr(self, "augmented_time"):
            data["augmented_time"] = self.augmented_time
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def load(cls, path, device='cpu', dtype=torch.float32, map_location=None):
        """
        Load from npz file

        :param path: npz path
        :param device: str device to put data
        :param dtype: str torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device

        """
        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        tree = cls(dtype=dtype, device=device)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.child = torch.from_numpy(z["child"]).to(device)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        tree._n_internal.fill_(z["n_internal"].item())
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                                np.float32)).to(device)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(device)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(device)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        tree.data_format = DataFormat(z['data_format'].item()) if \
                'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
                          'extra_data' in z.files else None
        if 'time_steps' in z:
            tree.time_steps = z['time_steps']
        else:
            tree.time_steps = -1
        if 'augmented_time' in z:
            tree.augmented_time = z['augmented_time']
        else:
            tree.augmented_time = False
        return tree

    # Magic
    def __repr__(self):
        return (f"svox.N3DynamicTree(N={self.N}, data_dim={self.data_dim}, " +
                f"depth_limit={self.depth_limit}, " +
                f"capacity:{self.n_internal - self._n_free.item()}/{self.capacity}, " +
                f"data_format:{self.data_format or 'RGBA'}, " +
                f"time_steps:{self.time_steps}"+("(+2)" if self.augmented_time else "")+")");

    def __getitem__(self, key):
        """
        Get N3TreeView
        """
        return N3TreeView(self, key)

    def __setitem__(self, key, val):
        N3TreeView(self, key).set(val)

    def _spec(self, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
        tree_spec.data = self.data
        tree_spec.child = self.child
        tree_spec.parent_depth = self.parent_depth
        if self.extra_data is not None:
            tree_spec.extra_data = self.extra_data  
            if self.data_format.format == DataFormat.FC or self.data_format.format == DataFormat.LFC:
                tree_spec.moving_cams = True
            else:
                tree_spec.moving_cams = False
        else:
            tree_spec.extra_data = torch.empty((0, 0), dtype=self.data.dtype, device=self.data.device)
            tree_spec.moving_cams=False
        tree_spec.offset = self.offset if world else torch.tensor(
                  [0.0, 0.0, 0.0], dtype=self.data.dtype, device=self.data.device)
        tree_spec.scaling = self.invradius if world else torch.tensor(
                  [1.0, 1.0, 1.0], dtype=self.data.dtype, device=self.data.device)
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                    self._weight_accum is not None else torch.empty(
                            0, dtype=self.data.dtype, device=self.data.device)
            tree_spec._weight_accum_max = (self._weight_accum_op == 'max')
        tree_spec.timesteps = self.time_steps
        tree_spec.augmented_time = self.augmented_time if hasattr(self, "augmented_time") else False
        return tree_spec

# Redirect functions to N3TreeView so you can do tree.depths instead of tree[:].depths
def _redirect_to_n3dynamictreeview():
    redir_props = ['depths', 'lengths', 'lengths_local', 'corners', 'corners_local',
                   'values', 'values_local']
    redir_funcs = ['sample', 'sample_local', 'aux',
            'normal_', 'clamp_', 'uniform_', 'relu_', 'sigmoid_', 'nan_to_num_']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_func)(*args, **kwargs)
        setattr(N3DynamicTree, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    def redirect_prop(redir_prop):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_prop)
        setattr(N3DynamicTree, redir_prop, property(redir_impl))
    for redir_prop in redir_props:
        redirect_prop(redir_prop)
_redirect_to_n3dynamictreeview()

