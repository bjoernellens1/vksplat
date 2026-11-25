// GPU Temporal Depth Reprojection Shader.
//
// Reprojects a batch of H_hist historical reliable sensor depth maps into a target
// camera viewpoint using an atomic uint32 z-buffer (asuint(z) InterlockedMin),
// view-angle cosine weighting, and per-pixel valid-support counts.
//
// ENTRY == 0 (temporal_depth_reproject_scatter):
//   1D dispatch over num_views * src_h * src_w source pixels.
//   Unprojects valid source depth -> transforms via T_src_to_dst -> projects into dst ->
//   updates dst_zbuf_u32 (InterlockedMin), dst_support_u32 (InterlockedAdd), and
//   dst_weight_buf (InterlockedAddF32 of view-angle cosine).
//
// ENTRY == 1 (temporal_depth_reproject_finalize):
//   1D dispatch over dst_h * dst_w destination pixels.
//   Converts dst_zbuf_u32 (+inf sentinel 0x7F800000u) to float32 depth (0.0f where invalid).

#include "config.slang"

#ifndef ENTRY
#define ENTRY 0
#endif

static const uint32_t BLOCK_SIZE = 256u;
static const uint32_t ZBUF_INF_BITS = 0x7F800000u;

#if ENTRY == 0

struct TemporalReprojectUniforms {
    uint32_t num_views;
    uint32_t src_h;
    uint32_t src_w;
    uint32_t dst_h;
    uint32_t dst_w;
    float dst_fx;
    float dst_fy;
    float dst_cx;
    float dst_cy;
    float min_depth_m;
    float max_depth_m;
    float min_cos_angle;
};

layout(binding=0) StructuredBuffer<float> src_depths;
layout(binding=1) StructuredBuffer<float4> view_params;
layout(binding=2) RWStructuredBuffer<uint32_t> dst_zbuf_u32;
layout(binding=3) RWByteAddressBuffer dst_weight_buf;
layout(binding=4) RWStructuredBuffer<uint32_t> dst_support_u32;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform TemporalReprojectUniforms u
) {
    const uint32_t tid = dispatchThreadID.x;
    const uint32_t src_hw = u.src_h * u.src_w;
    const uint32_t total_src = u.num_views * src_hw;
    if (tid >= total_src) {
        return;
    }

    const float z_s = src_depths[tid];
    if (!isfinite(z_s) || z_s < u.min_depth_m || z_s > u.max_depth_m) {
        return;
    }

    const uint32_t v = tid / src_hw;
    const uint32_t rem = tid - v * src_hw;
    const uint32_t v_s = rem / u.src_w;
    const uint32_t u_s = rem - v_s * u.src_w;

    const uint32_t vp = 4u * v;
    const float4 r0 = view_params[vp + 0u];
    const float4 r1 = view_params[vp + 1u];
    const float4 r2 = view_params[vp + 2u];
    const float4 ks = view_params[vp + 3u];  // (inv_fx_s, inv_fy_s, cx_s, cy_s)

    const float x_s = (float(u_s) - ks.z) * ks.x * z_s;
    const float y_s = (float(v_s) - ks.w) * ks.y * z_s;

    const float x_d = r0.x * x_s + r0.y * y_s + r0.z * z_s + r0.w;
    const float y_d = r1.x * x_s + r1.y * y_s + r1.z * z_s + r1.w;
    const float z_d = r2.x * x_s + r2.y * y_s + r2.z * z_s + r2.w;

    if (!isfinite(z_d) || z_d < u.min_depth_m || z_d > u.max_depth_m) {
        return;
    }

    const float3 p_d = float3(x_d, y_d, z_d);
    const float3 t_s = float3(r0.w, r1.w, r2.w);
    const float3 to_dst = -p_d;
    const float3 to_src = t_s - p_d;
    const float len_dst = length(to_dst);
    const float len_src = length(to_src);
    if (len_dst <= 1e-8f || len_src <= 1e-8f) {
        return;
    }
    const float cos_w = dot(to_src, to_dst) / (len_src * len_dst);
    if (!isfinite(cos_w) || cos_w < u.min_cos_angle || cos_w <= 0.0f) {
        return;
    }

    const float inv_z = 1.0f / z_d;
    const int32_t u_d = int32_t(floor(u.dst_fx * (x_d * inv_z) + u.dst_cx + 0.5f));
    const int32_t v_d = int32_t(floor(u.dst_fy * (y_d * inv_z) + u.dst_cy + 0.5f));
    if (u_d < 0 || u_d >= int32_t(u.dst_w) || v_d < 0 || v_d >= int32_t(u.dst_h)) {
        return;
    }

    const uint32_t pix = uint32_t(v_d) * u.dst_w + uint32_t(u_d);
    InterlockedMin(dst_zbuf_u32[pix], asuint(z_d));
    InterlockedAdd(dst_support_u32[pix], 1u);
#if USE_EMULATED_F32_ATOMIC
    AtomicAddFloat(dst_weight_buf, 4u * pix, cos_w);
#else
    dst_weight_buf.InterlockedAddF32(4u * pix, cos_w);
#endif
}

#elif ENTRY == 1

struct TemporalFinalizeUniforms {
    uint32_t num_dst_pixels;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

layout(binding=0) StructuredBuffer<uint32_t> dst_zbuf_u32;
layout(binding=1) StructuredBuffer<uint32_t> dst_support_u32;
layout(binding=2) RWStructuredBuffer<float> dst_depth_f32;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform TemporalFinalizeUniforms u
) {
    const uint32_t pix = dispatchThreadID.x;
    if (pix >= u.num_dst_pixels) {
        return;
    }
    const uint32_t cnt = dst_support_u32[pix];
    const uint32_t z_bits = dst_zbuf_u32[pix];
    dst_depth_f32[pix] = (cnt > 0u && z_bits != ZBUF_INF_BITS) ? asfloat(z_bits) : 0.0f;
}

#endif
