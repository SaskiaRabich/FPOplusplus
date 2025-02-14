#pragma once
#include "volrend/internal/n3tree_query.hpp"
#include <cmath>
#include <cuda_fp16.h>
#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/render_options.hpp"
#include "volrend/cuda/common.cuh"
#include "volrend/internal/data_spec.hpp"
#include "volrend/internal/lumisphere.hpp"
#include "volrend/internal/morton.hpp"

namespace volrend {
namespace device {
namespace {

template<typename scalar_t>
__device__ __inline__ void _dda_world(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax,
        const float* __restrict__ render_bbox) {
    scalar_t t1, t2;
    *tmin = 0.0;
    *tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = (render_bbox[i] + 1e-6 - cen[i]) * _invdir[i];
        t2 = (render_bbox[i + 3] - 1e-6 - cen[i]) * _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

template<typename scalar_t>
__device__ __inline__ scalar_t _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir) {
    scalar_t t1, t2;
    scalar_t tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template<typename scalar_t>
__device__ __inline__ void trace_ray(
        const internal::FTreeSpec& __restrict__ tree,
        scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        const scalar_t* __restrict__ cen,
        RenderOptions opt,
        float tmax_bg,
        scalar_t* __restrict__ out) {

    const float delta_scale = _get_delta_scale(
            tree.scale, /*modifies*/ dir);
    tmax_bg /= delta_scale;

    scalar_t tmin, tmax;
    scalar_t invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    _dda_world(cen, invdir, &tmin, &tmax, opt.render_bbox);
    tmax = min(tmax, tmax_bg);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        if (opt.render_depth)
            out[3] = 1.f;
        return;
    } else {
        scalar_t pos[3], tmp;
        const half* tree_val;
        scalar_t basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        scalar_t fc_basis[VOLREND_GLOBAL_FC_MAX];
        internal::maybe_precalc_basis(tree, vdir, basis_fn);
        for (int i = 0; i < opt.basis_minmax[0]; ++i) {
            basis_fn[i] = 0.f;
        }
        for (int i = opt.basis_minmax[1] + 1; i < VOLREND_GLOBAL_BASIS_MAX; ++i) {
            basis_fn[i] = 0.f;
        }
        float ts;
        if(tree.data_format.format == DataFormat::FC || tree.data_format.format == DataFormat::LFC) {
            ts = opt.timestep;
            if (opt.round_timestep) {
                ts = floor(ts+0.5);
            }
            if (tree.augmented_time) {
                ts += 1.f;
                internal::precalc_fourier_basis(VOLREND_MAX(tree.data_format.fc_dim_rgb,tree.data_format.fc_dim_sigma), ts, tree.timesteps+2, fc_basis);
            }
            else {
                internal::precalc_fourier_basis(VOLREND_MAX(tree.data_format.fc_dim_rgb,tree.data_format.fc_dim_sigma), ts, tree.timesteps, fc_basis);
            }
        }

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            pos[0] = cen[0] + t * dir[0];
            pos[1] = cen[1] + t * dir[1];
            pos[2] = cen[2] + t * dir[2];

            internal::query_single_from_root(tree, pos, &tree_val, &cube_sz);
            
            scalar_t att;
            const scalar_t t_subcube = _dda_unit(pos, invdir) /  cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;

            scalar_t sigma;
            if (tree.data_format.format == DataFormat::FC) {
                sigma = 0.;
                for (int i = 0; i < tree.data_format.fc_dim_sigma; ++i) {
                    sigma += fc_basis[i] * __half2float(tree_val[tree.data_dim - tree.data_format.fc_dim_sigma+i]);
                }
            } else if (tree.data_format.format == DataFormat::LFC) {
                sigma = 0.;
                for (int i = 0; i < tree.data_format.fc_dim_sigma; ++i) {
                    sigma += fc_basis[i] * __half2float(tree_val[tree.data_dim - tree.data_format.fc_dim_sigma+i]);
                }
                sigma = expf(sigma) -1;
            } else {
                sigma = __half2float(tree_val[tree.data_dim - 1]);
            }

            if (sigma > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                if (opt.render_depth) {
                    out[0] += weight * t;
                } else {
                    if (tree.data_format.basis_dim >= 0 
                    && tree.data_format.format != DataFormat::FC
                    && tree.data_format.format != DataFormat::LFC) {
                        int off = 0;
#define MUL_BASIS_I(t) basis_fn[t] * __half2float(tree_val[off + t])
#pragma unroll 3
                        for (int t = 0; t < 3; ++ t) {
                            tmp = basis_fn[0] * __half2float(tree_val[off]);
                            switch(tree.data_format.basis_dim) {
                                case 25:
                                    tmp += MUL_BASIS_I(16) +
                                        MUL_BASIS_I(17) +
                                        MUL_BASIS_I(18) +
                                        MUL_BASIS_I(19) +
                                        MUL_BASIS_I(20) +
                                        MUL_BASIS_I(21) +
                                        MUL_BASIS_I(22) +
                                        MUL_BASIS_I(23) +
                                        MUL_BASIS_I(24);
                                case 16:
                                    tmp += MUL_BASIS_I(9) +
                                          MUL_BASIS_I(10) +
                                          MUL_BASIS_I(11) +
                                          MUL_BASIS_I(12) +
                                          MUL_BASIS_I(13) +
                                          MUL_BASIS_I(14) +
                                          MUL_BASIS_I(15);

                                case 9:
                                    tmp += MUL_BASIS_I(4) +
                                        MUL_BASIS_I(5) +
                                        MUL_BASIS_I(6) +
                                        MUL_BASIS_I(7) +
                                        MUL_BASIS_I(8);

                                case 4:
                                    tmp += MUL_BASIS_I(1) +
                                        MUL_BASIS_I(2) +
                                        MUL_BASIS_I(3);
                            }
                            out[t] += weight / (1.f + expf(-tmp));
                            off += tree.data_format.basis_dim;
                        }
#undef MUL_BASIS_I
                    } else if (tree.data_format.format == DataFormat::FC || tree.data_format.format == DataFormat::LFC) {
                        int off = 0;
                        scalar_t sh_tmp[3*VOLREND_GLOBAL_BASIS_MAX];
                        for (int t = 0; t < 3; ++t) {
                            for (int j = 0; j < tree.data_format.basis_dim; ++j){
                                float tmp = 0.0;
                                int off_fc = t * tree.data_format.basis_dim * tree.data_format.fc_dim_rgb + j*tree.data_format.fc_dim_rgb;
                                for (int i = 0; i < tree.data_format.fc_dim_rgb; ++i){
                                    tmp += fc_basis[i] * __half2float(tree_val[off_fc +i]);
                                }
                                sh_tmp[t * tree.data_format.basis_dim + j] = tmp;
                            }
                        }
#define MUL_BASIS_I(t) basis_fn[t] * sh_tmp[off + t]
#pragma unroll 3
                        for (int t = 0; t < 3; ++ t) {
                            tmp = basis_fn[0] * sh_tmp[off];
                            switch(tree.data_format.basis_dim) {
                                case 25:
                                    tmp += MUL_BASIS_I(16) +
                                        MUL_BASIS_I(17) +
                                        MUL_BASIS_I(18) +
                                        MUL_BASIS_I(19) +
                                        MUL_BASIS_I(20) +
                                        MUL_BASIS_I(21) +
                                        MUL_BASIS_I(22) +
                                        MUL_BASIS_I(23) +
                                        MUL_BASIS_I(24);
                                case 16:
                                    tmp += MUL_BASIS_I(9) +
                                          MUL_BASIS_I(10) +
                                          MUL_BASIS_I(11) +
                                          MUL_BASIS_I(12) +
                                          MUL_BASIS_I(13) +
                                          MUL_BASIS_I(14) +
                                          MUL_BASIS_I(15);

                                case 9:
                                    tmp += MUL_BASIS_I(4) +
                                        MUL_BASIS_I(5) +
                                        MUL_BASIS_I(6) +
                                        MUL_BASIS_I(7) +
                                        MUL_BASIS_I(8);

                                case 4:
                                    tmp += MUL_BASIS_I(1) +
                                        MUL_BASIS_I(2) +
                                        MUL_BASIS_I(3);
                            }
                            out[t] += weight / (1.f + expf(-tmp));
                            off += tree.data_format.basis_dim;
                        }
                    } else {
                        for (int j = 0; j < 3; ++j) {
                            out[j] += __half2float(tree_val[j]) * weight;
                        }
                    }
                }

                light_intensity *= att;

                if (light_intensity < opt.stop_thresh) {
                    // Almost full opacity, stop
                    if (opt.render_depth) {
                        out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
                    }
                    scalar_t scale = 1.f / (1.f - light_intensity);
                    out[0] *= scale; out[1] *= scale; out[2] *= scale;
                    out[3] = 1.f;
                    return;
                }
            }
            t += delta_t;
        }
        if (opt.render_depth) {
            out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
            out[3] = 1.f;
        } else {
            out[3] = 1.f - light_intensity;
        }
    }
}

}  // namespace
}  // namespace device
}  // namespace volrend
