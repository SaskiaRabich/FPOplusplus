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
Volume rendering utilities
"""

import torch
import numpy as np
from torch import nn, autograd
from collections import namedtuple
from warnings import warn

from svox.helpers import _get_c_extension, LocalIndex, DataFormat

NDCConfig = namedtuple('NDCConfig', ["width", "height", "focal"])
Rays = namedtuple('Rays', ["origins", "dirs", "viewdirs"])

_C = _get_c_extension()

def _rays_spec_from_rays(rays):
    spec = _C.RaysSpec()
    spec.origins = rays.origins
    spec.dirs = rays.dirs
    spec.vdirs = rays.viewdirs
    return spec

def _make_camera_spec(c2w, width, height, fx, fy):
    spec = _C.CameraSpec()
    spec.c2w = c2w
    spec.width = width
    spec.height = height
    spec.fx = fx
    spec.fy = fy
    spec.K = torch.zeros(3,3)
    spec.K_specified = False
    return spec

def _make_camera_spec_from_K(c2w, width, height, K):
    spec = _C.CameraSpec()
    spec.c2w = c2w
    spec.width = width
    spec.height = height
    spec.fx = K[0][0].item()
    spec.fy = K[1][1].item()
    spec.K = K
    spec.K_specified = True
    return spec

class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree, rays, opt, timestep=-1.):
        out = _C.volume_render(tree, rays, opt, timestep)
        ctx.tree = tree
        ctx.rays = rays
        ctx.opt = opt
        ctx.timestep = timestep
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            return _C.volume_render_backward(
                ctx.tree, ctx.rays, ctx.opt, grad_out.contiguous(), ctx.timestep
            ), None, None, None, None
        return None, None, None, None, None

class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree, cam, opt, timestep=-1.):
        out = _C.volume_render_image(tree, cam, opt, timestep)
        ctx.tree = tree
        ctx.cam = cam
        ctx.opt = opt
        ctx.timestep = timestep
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            return _C.volume_render_image_backward(
                ctx.tree, ctx.cam, ctx.opt, grad_out.contiguous(), ctx.timestep
            ), None, None, None, None
        return None, None, None, None, None


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates. (only for grad check)"""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions

class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree,
            step_size : float=1e-3,
            background_brightness : float=1.0,
            ndc : NDCConfig=None,
            min_comp : int=0,
            max_comp : int=-1,
            density_softplus : bool=False,
            rgb_padding : float=0.0,
        ):
        """
        Construct volume renderer associated with given N^3 tree.
        You can construct multiple renderer instances for the same tree;
        the renderer class only stores configurations and no persistent tensors.

        The renderer traces rays with origins/dirs within the octree boundaries,
        detection ray-voxel intersections. The color and density within
        each voxel is assumed constant, and no interpolation is performed.

        For each intersection point, it queries the tree, assuming the last data dimension
        is density (sigma) and the rest of the dimensions are color,
        formatted according to tree.data_format.
        It then applies SH/SG/ASG basis functions, if any, according to viewdirs.
        Sigmoid will be applied to these colors to normalize them,
        and optionally a shifted softplus is applied to the density.

        :param tree: N3Tree instance for rendering
        :param step_size: float step size eps, added to each voxel aabb intersection step
        :param background_brightness: float background brightness, 1.0 = white
        :param ndc: NDCConfig, NDC coordinate configuration,
                    namedtuple(width, height, focal).
                    None = no NDC, use usual coordinates
        :param min_comp: minimum SH/SG component to render.
        :param max_comp: maximum SH/SG component to render, -1=last.
                         Set :code:`min_comp = max_comp` to render a particular
                         component. Default means all.
                         *Tip:* If you set :code:`min_comp > max_comp`,
                         the renderer will render all colors as 0.5 luminosity gray.
                         This is still differentiable and be used to implement a mask loss.
        :param density_softplus: if true, applies :math:`\\log(1 + \\exp(sigma - 1))`.
                                 **Mind the shift -1!** (from mip-NeRF).
                                 Please note softplus will NOT be compatible with volrend,
                                 please pre-apply it .
        :param rgb_padding: to avoid oversaturating the sigmoid,
                        applies :code:`* (1 + 2 * rgb_padding) - rgb_padding` to
                        colors after sigmoid (from mip-NeRF).
                        Please note the padding will NOT be compatible with volrend,
                        although most likely the effect is very small.
                        0.001 is a reasonable value to try.

        """
        super().__init__()
        self.tree = tree
        self.step_size = step_size
        self.background_brightness = background_brightness
        self.ndc_config = ndc
        self.min_comp = min_comp
        self.max_comp = max_comp
        self.density_softplus = density_softplus
        self.rgb_padding = rgb_padding
        if isinstance(tree.data_format, DataFormat):
            self._data_format = None
        else:
            warn("Legacy N3Tree (pre 0.2.18) without data_format, auto-infering SH deg")
            # Auto SH deg
            ddim = tree.data_dim
            if ddim == 4:
                self._data_format = DataFormat("")
            else:
                self._data_format = DataFormat(f"SH{(ddim - 1) // 3}")
        self.tree._weight_accum = None

    def forward(self, rays : Rays, cuda=True, fast=False):
        """
        Render a batch of rays. Differentiable.

        :param rays: namedtuple :code:`svox.Rays` of origins
                     :code:`(B, 3)`, dirs :code:`(B, 3):, viewdirs :code:`(B, 3)`
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(B, rgb_dim)`.
                Where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.
        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            assert self.data_format.format in [DataFormat.RGBA, DataFormat.SH], \
                 "Unsupported data format for slow volume rendering"
            warn("Using slow volume rendering, should only be used for debugging")
            def dda_unit(cen, invdir):
                """
                voxel aabb ray tracing step
                :param cen: jnp.ndarray [B, 3] center
                :param invdir: jnp.ndarray [B, 3] 1/dir
                :return: tmin jnp.ndarray [B] at least 0;
                         tmax jnp.ndarray [B]
                """
                B = invdir.shape[0]
                tmin = torch.zeros((B,), dtype=cen.dtype, device=cen.device)
                tmax = torch.full((B,), fill_value=1e9, dtype=cen.dtype, device=cen.device)
                for i in range(3):
                    t1 = -cen[..., i] * invdir[..., i]
                    t2 = t1 + invdir[..., i]
                    tmin = torch.max(tmin, torch.min(t1, t2))
                    tmax = torch.min(tmax, torch.max(t1, t2))
                return tmin, tmax

            origins, dirs, viewdirs = rays.origins, rays.dirs, rays.viewdirs
            origins = self.tree.world2tree(origins)
            B = dirs.size(0)
            assert viewdirs.size(0) == B and origins.size(0) == B
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)

            sh_mult = None
            if self.data_format.format == DataFormat.SH:
                from svox import sh
                sh_order = int(self.data_format.basis_dim ** 0.5) - 1
                sh_mult = sh.eval_sh_bases(sh_order, viewdirs)[:, None]

            invdirs = 1.0 / (dirs + 1e-9)
            t, tmax = dda_unit(origins, invdirs)
            light_intensity = torch.ones(B, device=origins.device)
            out_rgb = torch.zeros((B, 3), device=origins.device)

            good_indices = torch.arange(B, device=origins.device)
            delta_scale = (dirs / self.tree.invradius[None]).norm(dim=1)
            while good_indices.numel() > 0:
                pos = origins + t[:, None] * dirs
                treeview = self.tree[LocalIndex(pos)]
                rgba = treeview.values
                cube_sz = treeview.lengths_local
                pos_t = (pos - treeview.corners_local) / cube_sz[:, None]
                treeview = None

                subcube_tmin, subcube_tmax = dda_unit(pos_t, invdirs)

                delta_t = (subcube_tmax - subcube_tmin) * cube_sz + self.step_size
                att = torch.exp(- delta_t * torch.relu(rgba[..., -1]) * delta_scale[good_indices])
                weight = light_intensity[good_indices] * (1.0 - att)
                rgb = rgba[:, :-1]
                if self.data_format.format == DataFormat.SH:
                    # [B', 3, n_sh_coeffs]
                    rgb_sh = rgb.reshape(-1, 3, self.data_format.basis_dim)
                    rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))   # [B', 3]
                else:
                    rgb = torch.sigmoid(rgb)
                rgb = weight[:, None] * rgb[:, :3]

                out_rgb[good_indices] += rgb
                light_intensity[good_indices] *= att
                t += delta_t

                mask = t < tmax
                good_indices = good_indices[mask]
                origins = origins[mask]
                dirs = dirs[mask]
                invdirs = invdirs[mask]
                t = t[mask]
                if sh_mult is not None:
                    sh_mult = sh_mult[mask]
                tmax = tmax[mask]
            out_rgb += light_intensity * self.background_brightness
            return out_rgb
        return _VolumeRenderFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _rays_spec_from_rays(rays),
            self._get_options(fast)
        )

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None,
            cuda=True, fast=False):
        """
        Render a perspective image. Differentiable.

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(height, width, rgb_dim)`
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            return self(VolumeRenderer.persp_rays(c2w, width, height, fx, fy),
                        cuda=False, fast=fast)
        if fy is None:
            fy = fx
        return _VolumeRenderImageFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _make_camera_spec(c2w.to(dtype=self.tree.data.dtype),
                              width, height, fx, fy),
            self._get_options(fast),
        )

    def render_persp_from_K(self, c2w, K, width=800, height=800,
            cuda=True, fast=False):
        """
        Render a perspective image. Differentiable.

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(height, width, rgb_dim)`
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            return self(VolumeRenderer.persp_rays_from_K(c2w, K, width, height),
                        cuda=False, fast=fast)
        return _VolumeRenderImageFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _make_camera_spec_from_K(c2w.to(dtype=self.tree.data.dtype),
                              width, height, K.to(dtype=self.tree.data.dtype)),
            self._get_options(fast),
        )

    def se_grad(self, rays : Rays, colors, timestep = -1.):
        """
        Returns rendered color + gradient and Hessian diagonal of the total
        squared error:
        :math:`\\frac{1}{2} \\sum_{r \\in \\mathcal{R}} (\\hat{C}(r) - C(r))^2`
        where :math:`\\hat{C}(r)` is computed from the ray and
        :math:`C(r)` comes from the provided tensor :code:`colors`.
        This is the arbitrary ray-batch version of :code:`se_grad`.
        This is useful for diagonal NNLS methods for scaling step sizes.
        Note currently the Hessian is actually the squared norm of Jacobian rows
        as in Gauss-Newton algorithms.

        The tree's rendered output dimension (rgb_dim) cannot
        be greater than 4 (this is almost always true, don't need to worry).

        :param rays: namedtuple :code:`svox.Rays` of origins
                     :code:`(B, 3)`, dirs :code:`(B, 3):, viewdirs :code:`(B, 3)`
        :param colors: torch.Tensor :code:`(B, 3)` reference colors

        :return: :code:`colors (B, rgb_dim), grad (shape of tree.data),
                               diag_hessian (shape of tree.data)`
        """
        if _C is None or not self.tree.data.is_cuda:
            assert False, "Not supported in current version, use CUDA kernel"
        return _C.se_grad(self.tree._spec(), _rays_spec_from_rays(rays),
                          colors, self._get_options(False), timestep)

    def se_grad_persp(self, c2w, colors, width=800, height=800, fx=1111.111, fy=None, timestep = -1.):
        """
        Returns rendered color + gradient and Hessian diagonal of the total
        squared error:
        :math:`\\frac{1}{2} \\sum_{r \\in \\mathcal{R}} (\\hat{C}(r) - C(r))^2`
        where :math:`\\hat{C}(r)` is computed from the ray and
        :math:`C(r)` comes from the provided tensor :code:`colors`.
        This is the image-batch version of :code:`se_grad`.
        This is useful for diagonal NNLS methods for scaling step sizes.
        Note currently the Hessian is actually the squared norm of Jacobian rows
        as in Gauss-Newton algorithms.

        The tree's rendered output dimension (rgb_dim) cannot
        be greater than 4 (this is almost always true, don't need to worry).

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param colors: torch.Tensor :code:`(H, W, 3)` reference colors
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx

        :return: :code:`colors (H, W, rgb_dim), grad (shape of tree.data),
                               diag_hessian (shape of tree.data)`
        """
        if fy is None:
            fy = fx
        if _C is None or not self.tree.data.is_cuda:
            assert False, "Not supported in current version, use CUDA kernel"
        return _C.se_grad_persp(
            self.tree._spec(),
            _make_camera_spec(c2w.to(dtype=self.tree.data.dtype),
                              width, height, fx, fy),
            self._get_options(False),
            colors, timestep)

    @staticmethod
    def persp_rays(c2w, width=800, height=800, fx=1111.111, fy=None):
        """
        Generate perspective camera rays in row major order, then
        usable for renderer's forward method.
        *NDC is not supported currently.*

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx

        :return: rays namedtuple svox.Rays of origins
                     :code:`(H*W, 3)`, dirs :code:`(H*W, 3):, viewdirs :code:`(H*W, 3)`,
                     where H = W.

        """
        if fy is None:
            fy = fx
        origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=c2w.device),
            torch.arange(width, dtype=torch.float64, device=c2w.device),
        )
        xx = (xx - width * 0.5) / float(fx)
        yy = (yy - height * 0.5) / float(fy)
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, -yy, -zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3)
        del xx, yy, zz
        dirs = torch.matmul(c2w[None, :3, :3].double(), dirs[..., None])[..., 0].float()
        vdirs = dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    @staticmethod
    def persp_rays_from_K(c2w, K, width=800, height=800):
        """
        Generate perspective camera rays in row major order, then
        usable for renderer's forward method.
        *NDC is not supported currently.*

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param K: torch.Tensor (3, 3) camera intrinsics
        :param width: int output image width
        :param height: int output image height

        :return: rays namedtuple svox.Rays of origins
                     :code:`(H*W, 3)`, dirs :code:`(H*W, 3):, viewdirs :code:`(H*W, 3)`,
                     where H = W.

        """
        origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()

        batch_num=1
        # generate meshgrid
        xh, yw = torch.meshgrid([torch.arange(0, height, dtype=torch.float64, device=c2w.device), torch.arange(0, width, dtype=torch.float64, device=c2w.device)])
        coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],dim =0).float()
        coord_meshgrid = coord_meshgrid.view(1,3,-1)
        coord_meshgrid = coord_meshgrid #.cuda()
        # generate viewin directions
        Kinv = torch.inverse(K)[None,:]
        coord_meshgrids = coord_meshgrid.repeat(batch_num,1,1)
        dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
        #dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2))],dim = 1) ' This is used by the authors
        dir_in_camera = torch.cat([dir_in_camera, torch.zeros(batch_num,1,dir_in_camera.size(2))],dim = 1) # From me
        dir_in_world = torch.bmm(c2w[None,:], dir_in_camera)
        #dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
        dir_in_world = dir_in_world[:,0:3,:]
        dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
        dir_in_world = dir_in_world.reshape(batch_num,3, height, width)
        dirs = torch.permute(dir_in_world[0],(1,2,0)).reshape(height*width,3)

        vdirs=dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    @property
    def data_format(self):
        return self._data_format or self.tree.data_format

    def _get_options(self, fast=False):
        """
        Make RenderOptions struct to send to C++
        """
        opts = _C.RenderOptions()
        opts.step_size = self.step_size
        opts.background_brightness = self.background_brightness

        opts.format = self.data_format.format
        opts.basis_dim = self.data_format.basis_dim
        opts.min_comp = self.min_comp
        opts.max_comp = self.max_comp

        if self.max_comp < 0:
            opts.max_comp += opts.basis_dim

        opts.density_softplus = self.density_softplus
        opts.rgb_padding = self.rgb_padding

        if self.ndc_config is not None:
            opts.ndc_width = self.ndc_config.width
            opts.ndc_height = self.ndc_config.height
            opts.ndc_focal = self.ndc_config.focal
        else:
            opts.ndc_width = -1

        if fast:
            opts.sigma_thresh = 1e-2
            opts.stop_thresh = 1e-2
        else:
            opts.sigma_thresh = 0.0
            opts.stop_thresh = 0.0
        # Override
        if hasattr(self, "sigma_thresh"):
            opts.sigma_thresh = self.sigma_thresh
        if hasattr(self, "stop_thresh"):
            opts.stop_thresh = self.stop_thresh
        return opts

class VolumeRendererDynamic(VolumeRenderer):
    """
    Volume renderer
    """
    def __init__(self, tree,
            step_size : float=1e-3,
            background_brightness : float=1.0,
            ndc : NDCConfig=None,
            min_comp : int=0,
            max_comp : int=-1,
            density_softplus : bool=False,
            rgb_padding : float=0.0,
            time_steps : int = 1,
        ):
        """
        Construct volume renderer associated with given N^3 tree.
        You can construct multiple renderer instances for the same tree;
        the renderer class only stores configurations and no persistent tensors.

        The renderer traces rays with origins/dirs within the octree boundaries,
        detection ray-voxel intersections. The color and density within
        each voxel is assumed constant, and no interpolation is performed.

        For each intersection point, it queries the tree, assuming the last data dimension
        is density (sigma) and the rest of the dimensions are color,
        formatted according to tree.data_format.
        It then applies SH/SG/ASG basis functions, if any, according to viewdirs.
        Sigmoid will be applied to these colors to normalize them,
        and optionally a shifted softplus is applied to the density.

        :param tree: N3Tree instance for rendering
        :param step_size: float step size eps, added to each voxel aabb intersection step
        :param background_brightness: float background brightness, 1.0 = white
        :param ndc: NDCConfig, NDC coordinate configuration,
                    namedtuple(width, height, focal).
                    None = no NDC, use usual coordinates
        :param min_comp: minimum SH/SG component to render.
        :param max_comp: maximum SH/SG component to render, -1=last.
                         Set :code:`min_comp = max_comp` to render a particular
                         component. Default means all.
                         *Tip:* If you set :code:`min_comp > max_comp`,
                         the renderer will render all colors as 0.5 luminosity gray.
                         This is still differentiable and be used to implement a mask loss.
        :param density_softplus: if true, applies :math:`\\log(1 + \\exp(sigma - 1))`.
                                 **Mind the shift -1!** (from mip-NeRF).
                                 Please note softplus will NOT be compatible with volrend,
                                 please pre-apply it .
        :param rgb_padding: to avoid oversaturating the sigmoid,
                        applies :code:`* (1 + 2 * rgb_padding) - rgb_padding` to
                        colors after sigmoid (from mip-NeRF).
                        Please note the padding will NOT be compatible with volrend,
                        although most likely the effect is very small.
                        0.001 is a reasonable value to try.

        """
        super().__init__(tree,
            step_size=step_size,
            background_brightness=background_brightness,
            ndc=ndc,
            min_comp=min_comp,
            max_comp=max_comp,
            density_softplus=density_softplus,
            rgb_padding=rgb_padding)
        self.time_steps = time_steps

    def forward(self, rays : Rays, cuda=True, fast=False, timestep = -1.):
        """
        Render a batch of rays. Differentiable.

        :param rays: namedtuple :code:`svox.Rays` of origins
                     :code:`(B, 3)`, dirs :code:`(B, 3):, viewdirs :code:`(B, 3)`
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(B, rgb_dim)`.
                Where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.
        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            assert self.data_format.format in [DataFormat.RGBA, DataFormat.SH, DataFormat.FC,DataFormat.WA,DataFormat.WI,DataFormat.TEST,DataFormat.TESTB], \
                 "Unsupported data format for slow volume rendering"
            warn("Using slow volume rendering, should only be used for debugging")
            def dda_unit(cen, invdir):
                """
                voxel aabb ray tracing step
                :param cen: jnp.ndarray [B, 3] center
                :param invdir: jnp.ndarray [B, 3] 1/dir
                :return: tmin jnp.ndarray [B] at least 0;
                         tmax jnp.ndarray [B]
                """
                B = invdir.shape[0]
                tmin = torch.zeros((B,), dtype=cen.dtype, device=cen.device)
                tmax = torch.full((B,), fill_value=1e9, dtype=cen.dtype, device=cen.device)
                for i in range(3):
                    t1 = -cen[..., i] * invdir[..., i]
                    t2 = t1 + invdir[..., i]
                    tmin = torch.max(tmin, torch.min(t1, t2))
                    tmax = torch.min(tmax, torch.max(t1, t2))
                return tmin, tmax

            origins, dirs, viewdirs = rays.origins, rays.dirs, rays.viewdirs
            origins = self.tree.world2tree(origins)
            B = dirs.size(0)
            assert viewdirs.size(0) == B and origins.size(0) == B
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)

            sh_mult = None
            if self.data_format.format == DataFormat.SH:
                from svox import sh
                sh_order = int(self.data_format.basis_dim ** 0.5) - 1
                sh_mult = sh.eval_sh_bases(sh_order, viewdirs)[:, None]

            fc_conv = None
            if self.data_format.format == DataFormat.FC:
                from svox.fourier import FourierConverter
                from svox.svox import sh
                fc_conv = FourierConverter.from_DataFormat(self.data_format,self.time_steps, self.tree.data.device)
                sh_order = int(self.data_format.basis_dim ** 0.5) - 1
                sh_mult = sh.eval_sh_bases(sh_order, viewdirs)[:, None]
            elif self.data_format.format == DataFormat.WA:
                from svox.wavelet import WaveletConverter
                from svox.svox import sh
                fc_conv = WaveletConverter.from_DataFormat(self.data_format,self.time_steps, self.tree.data.device)
                sh_order = int(self.data_format.basis_dim ** 0.5) - 1
                sh_mult = sh.eval_sh_bases(sh_order, viewdirs)[:, None]

            invdirs = 1.0 / (dirs + 1e-9)
            t, tmax = dda_unit(origins, invdirs)
            light_intensity = torch.ones(B, device=origins.device)
            out_rgb = torch.zeros((B, 3), device=origins.device)

            good_indices = torch.arange(B, device=origins.device)
            delta_scale = (dirs / self.tree.invradius[None]).norm(dim=1)
            while good_indices.numel() > 0:
                pos = origins + t[:, None] * dirs
                pos = self.projectPointsToRefBox(pos,self.tree.extra_data,timestep)
                treeview = self.tree[LocalIndex(pos)]
                rgba = treeview.values
                cube_sz = treeview.lengths_local
                pos_t = (pos - treeview.corners_local) / cube_sz[:, None]
                treeview = None

                subcube_tmin, subcube_tmax = dda_unit(pos_t, invdirs)

                delta_t = (subcube_tmax - subcube_tmin) * cube_sz + self.step_size
                att = torch.exp(- delta_t * torch.relu(rgba[..., -1]) * delta_scale[good_indices])
                weight = light_intensity[good_indices] * (1.0 - att)
                rgb = rgba[:, :-1]
                if self.data_format.format != DataFormat.RGBA:
                    if self.data_format.format == DataFormat.FC:
                        rgb_sh = fc_conv.fourier2sh(rgb,timestep) #TODO check if correct
                        rgb_sh = rgb_sh.reshape(-1, 3, self.data_format.basis_dim)
                    elif self.data_format.format == DataFormat.WA:
                        rgb_sh = fc_conv.wavelet2sh(rgb,timestep) #TODO check if correct
                        rgb_sh = rgb_sh.reshape(-1, 3, self.data_format.basis_dim)
                    else:
                        # [B', 3, n_sh_coeffs]
                        rgb_sh = rgb.reshape(-1, 3, self.data_format.basis_dim)
                    rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))   # [B', 3]
                else:
                    rgb = torch.sigmoid(rgb)
                rgb = weight[:, None] * rgb[:, :3]

                out_rgb[good_indices] += rgb
                light_intensity[good_indices] *= att
                t += delta_t

                mask = t < tmax
                good_indices = good_indices[mask]
                origins = origins[mask]
                dirs = dirs[mask]
                invdirs = invdirs[mask]
                t = t[mask]
                if sh_mult is not None:
                    sh_mult = sh_mult[mask]
                tmax = tmax[mask]
            out_rgb += light_intensity * self.background_brightness
            return out_rgb
        return _VolumeRenderFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _rays_spec_from_rays(rays),
            self._get_options(fast),
            timestep
        )

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None,
            cuda=True, fast=False, timestep = -1.):
        """
        Render a perspective image. Differentiable.

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(height, width, rgb_dim)`
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            return self(VolumeRendererDynamic.persp_rays(c2w, width, height, fx, fy),
                        cuda=False, fast=fast, timestep=timestep)
        if fy is None:
            fy = fx
        return _VolumeRenderImageFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _make_camera_spec(c2w.to(dtype=self.tree.data.dtype),
                              width, height, fx, fy),
            self._get_options(fast), timestep
        )

    def render_persp_from_K(self, c2w, K, width=800, height=800,
            cuda=True, fast=False, timestep = -1.):
        """
        Render a perspective image. Differentiable.

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(height, width, rgb_dim)`
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            return self(VolumeRendererDynamic.persp_rays_from_K(c2w, K, width, height),
                        cuda=False, fast=fast, timestep=timestep)
        return _VolumeRenderImageFunction.apply(
            self.tree.data,
            self.tree._spec(),
            _make_camera_spec_from_K(c2w.to(dtype=self.tree.data.dtype),
                              width, height, K.to(dtype=self.tree.data.dtype)),
            self._get_options(fast), timestep
        )

    def se_grad(self, rays : Rays, colors, timestep=-1.):
        """
        Returns rendered color + gradient and Hessian diagonal of the total
        squared error:
        :math:`\\frac{1}{2} \\sum_{r \\in \\mathcal{R}} (\\hat{C}(r) - C(r))^2`
        where :math:`\\hat{C}(r)` is computed from the ray and
        :math:`C(r)` comes from the provided tensor :code:`colors`.
        This is the arbitrary ray-batch version of :code:`se_grad`.
        This is useful for diagonal NNLS methods for scaling step sizes.
        Note currently the Hessian is actually the squared norm of Jacobian rows
        as in Gauss-Newton algorithms.

        The tree's rendered output dimension (rgb_dim) cannot
        be greater than 4 (this is almost always true, don't need to worry).

        :param rays: namedtuple :code:`svox.Rays` of origins
                     :code:`(B, 3)`, dirs :code:`(B, 3):, viewdirs :code:`(B, 3)`
        :param colors: torch.Tensor :code:`(B, 3)` reference colors

        :return: :code:`colors (B, rgb_dim), grad (shape of tree.data),
                               diag_hessian (shape of tree.data)`
        """
        if _C is None or not self.tree.data.is_cuda:
            assert False, "Not supported in current version, use CUDA kernel"
        return _C.se_grad(self.tree._spec(), _rays_spec_from_rays(rays),
                          colors, self._get_options(False), timestep)

    def se_grad_persp(self, c2w, colors, width=800, height=800, fx=1111.111, fy=None, timestep = -1.):
        """
        Returns rendered color + gradient and Hessian diagonal of the total
        squared error:
        :math:`\\frac{1}{2} \\sum_{r \\in \\mathcal{R}} (\\hat{C}(r) - C(r))^2`
        where :math:`\\hat{C}(r)` is computed from the ray and
        :math:`C(r)` comes from the provided tensor :code:`colors`.
        This is the image-batch version of :code:`se_grad`.
        This is useful for diagonal NNLS methods for scaling step sizes.
        Note currently the Hessian is actually the squared norm of Jacobian rows
        as in Gauss-Newton algorithms.

        The tree's rendered output dimension (rgb_dim) cannot
        be greater than 4 (this is almost always true, don't need to worry).

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param colors: torch.Tensor :code:`(H, W, 3)` reference colors
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx

        :return: :code:`colors (H, W, rgb_dim), grad (shape of tree.data),
                               diag_hessian (shape of tree.data)`
        """
        if fy is None:
            fy = fx
        if _C is None or not self.tree.data.is_cuda:
            assert False, "Not supported in current version, use CUDA kernel"
        return _C.se_grad_persp(
            self.tree._spec(),
            _make_camera_spec(c2w.to(dtype=self.tree.data.dtype),
                              width, height, fx, fy),
            self._get_options(False),
            colors, timestep)

    @staticmethod
    def persp_rays(c2w, width=800, height=800, fx=1111.111, fy=None):
        """
        Generate perspective camera rays in row major order, then
        usable for renderer's forward method.
        *NDC is not supported currently.*

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx

        :return: rays namedtuple svox.Rays of origins
                     :code:`(H*W, 3)`, dirs :code:`(H*W, 3):, viewdirs :code:`(H*W, 3)`,
                     where H = W.

        """
        if fy is None:
            fy = fx
        origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=c2w.device),
            torch.arange(width, dtype=torch.float64, device=c2w.device),
        )
        xx = (xx - width * 0.5) / float(fx)
        yy = (yy - height * 0.5) / float(fy)
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, -yy, -zz), dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3)
        del xx, yy, zz
        dirs = torch.matmul(c2w[None, :3, :3].double(), dirs[..., None])[..., 0].float()
        vdirs = dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    @staticmethod
    def persp_rays_from_K(c2w, K, width=800, height=800):
        """
        Generate perspective camera rays in row major order, then
        usable for renderer's forward method.
        *NDC is not supported currently.*

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param K: torch.Tensor (3, 3) camera intrinsics
        :param width: int output image width
        :param height: int output image height

        :return: rays namedtuple svox.Rays of origins
                     :code:`(H*W, 3)`, dirs :code:`(H*W, 3):, viewdirs :code:`(H*W, 3)`,
                     where H = W.

        """
        origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()

        batch_num=1
        # generate meshgrid
        xh, yw = torch.meshgrid([torch.arange(0, height, dtype=torch.float64, device=c2w.device), torch.arange(0, width, dtype=torch.float64, device=c2w.device)])
        coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],dim =0).float()
        coord_meshgrid = coord_meshgrid.view(1,3,-1)
        coord_meshgrid = coord_meshgrid #.cuda()
        # generate viewin directions
        Kinv = torch.inverse(K)[None,:]
        coord_meshgrids = coord_meshgrid.repeat(batch_num,1,1)
        dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
        #dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2))],dim = 1) ' This is used by the authors
        dir_in_camera = torch.cat([dir_in_camera, torch.zeros(batch_num,1,dir_in_camera.size(2))],dim = 1) # From me
        dir_in_world = torch.bmm(c2w[None,:], dir_in_camera)
        #dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
        dir_in_world = dir_in_world[:,0:3,:]
        dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
        dir_in_world = dir_in_world.reshape(batch_num,3, height, width)
        dirs = torch.permute(dir_in_world[0],(1,2,0)).reshape(height*width,3)

        vdirs=dirs

        return Rays(
            origins=origins,
            dirs=dirs,
            viewdirs=vdirs
        )

    def _get_options(self, fast=False):
        """
        Make RenderOptions struct to send to C++
        """
        opts = _C.RenderOptions()
        opts.step_size = self.step_size
        opts.background_brightness = self.background_brightness

        opts.format = self.data_format.format
        opts.basis_dim = self.data_format.basis_dim
        opts.min_comp = self.min_comp
        opts.max_comp = self.max_comp

        if self.max_comp < 0:
            opts.max_comp += opts.basis_dim

        opts.density_softplus = self.density_softplus
        opts.rgb_padding = self.rgb_padding

        if self.ndc_config is not None:
            opts.ndc_width = self.ndc_config.width
            opts.ndc_height = self.ndc_config.height
            opts.ndc_focal = self.ndc_config.focal
        else:
            opts.ndc_width = -1

        if fast:
            opts.sigma_thresh = 1e-2
            opts.stop_thresh = 1e-2
        else:
            opts.sigma_thresh = 0.0
            opts.stop_thresh = 0.0
        # Override
        if hasattr(self, "sigma_thresh"):
            opts.sigma_thresh = self.sigma_thresh
        if hasattr(self, "stop_thresh"):
            opts.stop_thresh = self.stop_thresh

        opts.time_steps = self.time_steps
        opts.fc_dim1 = self.data_format.fc_dim1
        opts.fc_dim2 = self.data_format.fc_dim2

        return opts

    def projectPointsToTimestep(pts,bboxes,timestep):
        c = bboxes[2*timestep]
        r = bboxes[2*timestep+1]
        c_ref = bboxes[0]
        r_ref = bboxes[1]
        pts_rel = (pts - c_ref)/r_ref
        pts_projected = (pts_rel * r) + c
        return pts_projected

    def projectPointsToRefBox(pts,bboxes,timestep):
        c = bboxes[2*timestep]
        r = bboxes[2*timestep+1]
        c_ref = bboxes[0]
        r_ref = bboxes[1]
        pts_rel = (pts - c)/r
        pts_projected = (pts_rel * r_ref) + c_ref
        return pts_projected
