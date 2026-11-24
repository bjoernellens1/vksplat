#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN

import strategy_utils;


// Input buffers
layout(binding=0) RWStructuredBuffer<float> xyz_ws;  // 3
layout(binding=1) RWStructuredBuffer<float4> sh_coeffs;  // 16*3
layout(binding=2) RWStructuredBuffer<float4> rotations;  // 4
layout(binding=3) RWStructuredBuffer<float4> scales_opacs;  // 4

layout(binding=4) StructuredBuffer<int> tiles_touched;

layout(binding=5) StructuredBuffer<float2> v_output_xy_vs;
layout(binding=6) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding=7) StructuredBuffer<float> v_output_rgb;

// Optimizer buffers
layout(binding=8) RWStructuredBuffer<float> g_xyz_ws;  // 2*3
layout(binding=9) RWStructuredBuffer<float4> g_sh_coeffs_1;  // 16*3
layout(binding=10) RWStructuredBuffer<float4> g_sh_coeffs_2;  // 16*3
layout(binding=11) RWStructuredBuffer<float4> g_rotations;  // 2*4
layout(binding=12) RWStructuredBuffer<float4> g_scales_opacs;  // 2*4
layout(binding=13) StructuredBuffer<int32_t> group_ids;
layout(binding=14) StructuredBuffer<uint32_t> group_state;
layout(binding=15) StructuredBuffer<uint32_t> splat_state;
layout(binding=16) StructuredBuffer<float4> splat_anchor;  // (N, 2): [0]=(p_meas_xyz, w_depth), [1]=(normal_xyz, w_plane)
layout(binding=17) StructuredBuffer<uint32_t> active_indices;  // (active_splat_count,) compact non-frozen splat indices


// Adam optimizer

typedef vector<float,1> float1;

// "standard" Adam with bias correction (precomputed g1_sc, g2_sc)
__generic<let dim: int>
void optimizer_update(
    inout vector<float,dim> x, in vector<float,dim> v,
    inout vector<float,dim> g1, inout vector<float,dim> g2,
    float g1_sc, float g2_sc, vector<float,dim> lr
) {
    static const float beta_1 = 0.9;
    static const float beta_2 = 0.999;
    static const float eps = 1e-15;

    g1 = lerp(g1, v, 1.0-beta_1);
    g2 = lerp(g2, v*v, 1.0-beta_2);
    vector<float,dim> g1_t = g1 * g1_sc;
    vector<float,dim> g2_t = g2 * g2_sc;
    x = x - lr * g1_t / (sqrt(g2_t) + eps);
}


struct ShaderUniforms {
    uint32_t step;
    uint32_t active_sh;
    uint32_t num_splats;
    uint32_t image_size;  // upper 16 bit for height, lower for width
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float lr_means;
    float lr_quats;
    float lr_scales;
    float lr_opacities;
    float reg_scale;
    float reg_opacity;
    float lr_sh_dc;
    float lr_sh_rest;
    float4x4 world_view_transform;
    uint32_t num_groups;
    uint32_t protect_locked_groups;
    uint32_t num_anchors;
    float normal_scale_cap;
    uint32_t active_splat_count;
    uint32_t use_active_index_dispatch;
};


#define GROUP_SIZE SUBGROUP_SIZE

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform ShaderUniforms uniforms
) {
    const uint active_sh = uniforms.active_sh & 0xff;
    const uint camera_model = uniforms.active_sh >> 8;
    const uint step = uniforms.step;
    const uint image_height = uniforms.image_size >> 16;
    const uint image_width = uniforms.image_size & ((1 << 16) - 1);
    const float4x4 world_view_transform = (uniforms.world_view_transform);

    const uint tid = dispatchThreadID.x;
    uint g_idx = tid;
    if (uniforms.use_active_index_dispatch != 0u) {
        if (tid >= uniforms.active_splat_count)
            return;
        g_idx = active_indices[tid];
    }

    if (g_idx >= uniforms.num_splats)
        return;
    // Per-splat evidence policy (Sub-Plan V17.7):
    // 0: PREDICTED_UNOBSERVED (freeze geom/opac), 1: OBSERVED_UNCERTAIN (trainable w/ strong geom + cautious opac),
    // 2: OBSERVED_TRAINABLE (full optimization), 3: METRIC_VERIFIED (full optimization, NOT frozen),
    // 4: CONTRADICTED (skip), 5: MEASURED_FREE (full optimization), 0xFFFFFFFF: fallback to group_state.
    bool imagined = false;
    bool uncertain_obs = false;
    bool is_surfel_2d = false;
    {
        const uint32_t raw_st = splat_state[g_idx];
        const uint32_t st = splat_base_state(raw_st);
        is_surfel_2d = is_splat_surfel_2d(raw_st);
        const int32_t gid = group_ids[g_idx];
        if (gid >= 0 && uint32_t(gid) < uniforms.num_groups && group_state[gid] == 3u)
            return;
        if (st == SPLAT_STATE_CONTRADICTED)
            return;
        if (st == SPLAT_STATE_PREDICTED_UNOBSERVED) {
            imagined = true;
        } else if (st == SPLAT_STATE_OBSERVED_UNCERTAIN) {
            uncertain_obs = true;
        } else if (st == SPLAT_STATE_LEGACY_GROUP && gid >= 0 && uint32_t(gid) < uniforms.num_groups) {
            const uint32_t gstate = group_state[gid];
            if (gstate == 2u && uniforms.protect_locked_groups != 0u)
                return;
            imagined = (gstate == 1u);
        }
    }

    bool is_zero_grad = (tiles_touched[g_idx] == 0);

    // backward
    float2 v_out_xy_vs = is_zero_grad ? float2(0) : v_output_xy_vs[g_idx];
    float4 v_out_inv_cov_vs_opac = is_zero_grad ? float4(0) :
      v_output_inv_cov_vs[g_idx];
    float v_out_depths = 0.0;
    float3 v_out_rgb = is_zero_grad ? float3(0) :
      float3(read_t3_float3(g_idx, v_output_rgb));

    if (!is_zero_grad) {
        is_zero_grad = all(v_out_xy_vs == 0.0f) && all(v_out_inv_cov_vs_opac == 0.0f) && all(v_out_rgb == 0.0f);
    }

    // Freeze Adam momentum and regularization on unobserved steps (zero pixel differential)
    if (is_zero_grad)
        return;

    // load stuff
    float3 loaded_scales = scales_opacs[g_idx].xyz;
    if (is_surfel_2d) {
        loaded_scales.z = 0.0f;
    }
    Gaussian_3D gauss = {
        read_mean(g_idx, xyz_ws),
        read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh),
        rotations[g_idx],
        loaded_scales,
        scales_opacs[g_idx].w
    };

    DifferentialPair<Gaussian_3D> v_gauss = diffPair(gauss);
    float3 v_xyz_ws = float3(0);
    float4 v_rotation = float4(0);
    float4 v_scale_opac = float4(0);

    {
        Camera cam = load_camera(world_view_transform,
            uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy, camera_model, uniforms.dist_coeffs,
            image_height, image_width);
        Splat_2D_Vertex v_splat = {
            float3(v_out_xy_vs, v_out_depths),  // xyz_vs
            v_out_rgb,  // rgb
            float3(0.0),  // cov_vs
            v_out_inv_cov_vs_opac,  // inv_cov_vs
            0.0  // det
        };
        DifferentialPair<Camera> v_cam = diffPair(cam);
        bwd_diff(project_gaussian_to_camera)(v_gauss, v_cam, active_sh, v_splat);
        v_xyz_ws = v_gauss.d.xyz_ws;
        v_rotation = v_gauss.d.rotations;
        v_scale_opac = float4(v_gauss.d.scales, v_gauss.d.opacities);
    }

    // Robust measured-surface geometric guidance: L_depth + L_normal + L_plane
    if (!imagined && g_idx < uniforms.num_anchors) {
        float4 anc0 = splat_anchor[2*g_idx + 0];  // (p_meas_xyz, w_depth)
        float4 anc1 = splat_anchor[2*g_idx + 1];  // (normal_xyz, w_plane)
        if (anc0.w > 0.0f) {
            float3 dp = gauss.xyz_ws - anc0.xyz;
            float dlen = length(dp);
            float huber = (dlen > 0.025f) ? (0.025f / dlen) : 1.0f;
            v_xyz_ws += (anc0.w * huber) * dp;
        }
        if (anc1.w > 0.0f) {
            float3 nrm = anc1.xyz;
            float nlen = length(nrm);
            if (nlen > 1e-5f) {
                nrm /= nlen;
                float dn = dot(nrm, gauss.xyz_ws - anc0.xyz);
                v_xyz_ws += (2.0f * anc1.w * dn) * nrm;
                if (is_surfel_2d) {
                    DifferentialPair<float4> dp_q = diffPair(gauss.rotations);
                    bwd_diff(surfel_normal_alignment_loss)(dp_q, nrm, 0.5f * anc1.w);
                    v_rotation += dp_q.d;
                } else if (uniforms.normal_scale_cap > 0.0f && gauss.scales.z > uniforms.normal_scale_cap) {
                    v_scale_opac.z += anc1.w * (gauss.scales.z - uniforms.normal_scale_cap);
                }
            }
        }
    }

    // optimizers
    const float f_step = float(uniforms.step);
    const float g1_sc = 1.0f / (1.0f - pow(0.9f, f_step));
    const float g2_sc = 1.0f / (1.0f - pow(0.999f, f_step));

    uint group_offset = g_idx % GROUP_SIZE;
    uint group_base = g_idx - group_offset;

    uint32_t g1_idx = 2*g_idx;
    uint32_t g2_idx = 2*g_idx+1;

    // means
    if (!imagined) {
        float3 x = gauss.xyz_ws;
        float3 v = v_xyz_ws;
        float3 g1 = read_mean(g_idx, g_xyz_ws, 2, 0);
        float3 g2 = read_mean(g_idx, g_xyz_ws, 2, 1);
        optimizer_update(x, v, g1, g2, g1_sc, g2_sc, uniforms.lr_means);
        write_mean(x, g_idx, xyz_ws);
        write_mean(g1, g_idx, g_xyz_ws, 2, 0);
        write_mean(g2, g_idx, g_xyz_ws, 2, 1);
    }

    // rotations (normalized)
    if (!imagined) {
        float4 x = gauss.rotations;
        float4 v = v_rotation; // TODO: should ideally project when NORMALIZE_QUATS_IN_OPTIMIZER is True
        float4 g1 = g_rotations[g1_idx];
        float4 g2 = g_rotations[g2_idx];
        optimizer_update(x, v, g1, g2, g1_sc, g2_sc, uniforms.lr_quats);
        rotations[g_idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;
        g_rotations[g1_idx] = g1;
        g_rotations[g2_idx] = g2;
    }

    // scales (exp) and opac (sigmoid)
    if (!imagined) {
        // load
        float4 x = float4(gauss.scales, gauss.opacities);
        float4 v = float4(
            v_scale_opac.xyz + uniforms.reg_scale,
            v_scale_opac.w + uniforms.reg_opacity
        );
        float4 g1 = g_scales_opacs[g1_idx];
        float4 g2 = g_scales_opacs[g2_idx];

        // inverse map
        float4 u = float4(
            log(max(x.xyz, 1e-8)),  // scale: u = ln(x)
            log(max(x.w,1e-8) / max(1.0-x.w,1e-8))  // opacity: u = logit(x)
        );
        float4 dLdu = float4(
            v.xyz * x.xyz,  // scale: dL/du = dL/dx * dx/du = v * x
            v.w * x.w*(1.0-x.w)  // opacity: dL/du = dL/dx * dx/du = v * x(1-x)
        );

        // update
        float opac_lr = uncertain_obs ? (0.25f * uniforms.lr_opacities) : uniforms.lr_opacities;
        float lr_sz = is_surfel_2d ? 0.0f : uniforms.lr_scales;
        float4 lr = float4(uniforms.lr_scales, uniforms.lr_scales, lr_sz, opac_lr);
        optimizer_update(u, dLdu, g1, g2, g1_sc, g2_sc, lr);

        // map (no blanket opacity floor: false priors remain retractable!)
        x.xyz = exp(u.xyz);  // scale: x = exp(u)
        if (is_surfel_2d) {
            x.z = 0.0f;
        }
        x.w = 1.0 / (1.0+exp(-u.w));  // opacity: x = sigmoid(u)

        // save
        scales_opacs[g_idx] = x;
        g_scales_opacs[g1_idx] = g1;
        g_scales_opacs[g2_idx] = g2;
    }

    // SH

    uint reorder_offset = g_idx % SH_REORDER_SIZE;
    uint reorder_base = g_idx - reorder_offset;

    float lr0 = uniforms.lr_sh_dc;
    float lr1 = uniforms.lr_sh_rest;
    uint num_sh = get_num_sh(uniforms.active_sh);
    for (int di = 0; di < 12; di++) {
        uint cid = 12 * g_idx + di;
        uint bid = 12 * reorder_base + di * SH_REORDER_SIZE + reorder_offset;
        if (is_sh_skipped(uniforms.step, cid, num_sh))
            continue;

        float4 lr = (cid%12 == 0) ? float4(lr0, lr0, lr0, lr1) : float4(lr1);

        float4 x = get_sh_coeff_4(gauss.sh_coeffs, di);
        float4 v = is_zero_grad ? float4(0) :
            get_sh_coeff_4(v_gauss.d.sh_coeffs, di);
        // default optimizer (Adam)
        float4 g1 = g_sh_coeffs_1[bid];
        float4 g2 = g_sh_coeffs_2[bid];
        optimizer_update(x, v, g1, g2, g1_sc, g2_sc, lr);
        sh_coeffs[bid] = x;
        g_sh_coeffs_1[bid] = g1;
        g_sh_coeffs_2[bid] = g2;
    }
}
