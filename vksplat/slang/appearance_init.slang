#include "config.slang"

static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float MIN_DEPTH_M = 0.2f;
static const float MIN_DIST_SQ_M2 = 0.25f;

// Input buffers
layout(binding=0) StructuredBuffer<float> means_world;      // (N, 3)
layout(binding=1) StructuredBuffer<float> normals_world;    // (N, 3) or dummy
layout(binding=2) StructuredBuffer<float> fallback_rgb;     // (N, 3)
layout(binding=3) StructuredBuffer<float4> view_params;     // (V, 6): w2c_r0, w2c_r1, w2c_r2, cam_o, k_rgb, k_depth
layout(binding=4) ByteAddressBuffer rgb_views_u8;           // (V, H_rgb, W_rgb, rgb_channels) u8 + padding
layout(binding=5) StructuredBuffer<float> depth_views_f32;  // (V, H_d, W_d) f32

// Output buffers
layout(binding=6) RWStructuredBuffer<float> out_sh_coeffs;  // (N, 16, 3)
layout(binding=7) RWStructuredBuffer<float> out_rgb_init;   // (N, 3)
layout(binding=8) RWStructuredBuffer<int32_t> out_obs_count;// (N,)
layout(binding=9) RWStructuredBuffer<float> out_weight_sum; // (N,)
layout(binding=10) RWStructuredBuffer<float> out_se_num;    // (N,)

struct AppearanceInitUniforms {
    uint32_t num_points;
    uint32_t num_views;
    uint32_t h_rgb;
    uint32_t w_rgb;
    uint32_t h_d;
    uint32_t w_d;
    uint32_t rgb_channels;
    uint32_t has_normals;
    uint32_t sh_degree;
    float reg_lambda;
    float depth_tol_m;
    float min_cos_angle;
};

[ForceInline]
float3 load_rgb_u8(uint32_t byte_off) {
    uint32_t aligned = byte_off & ~3u;
    uint32_t shift = (byte_off & 3u) * 8u;
    uint32_t w0 = rgb_views_u8.Load(aligned);
    uint32_t w1 = (shift > 8u) ? rgb_views_u8.Load(aligned + 4u) : 0u;
    uint32_t packed = (shift == 0u) ? w0 : ((w0 >> shift) | (w1 << (32u - shift)));
    return float3(
        float(packed & 0xFFu),
        float((packed >> 8u) & 0xFFu),
        float((packed >> 16u) & 0xFFu)
    ) * (1.0f / 255.0f);
}

[ForceInline]
float3 sample_rgb_bilinear(uint32_t v, float u_rgb, float v_rgb, uint32_t h_rgb, uint32_t w_rgb, uint32_t rgb_ch) {
    float u_c = clamp(u_rgb, 0.0f, float(w_rgb - 1u));
    float v_c = clamp(v_rgb, 0.0f, float(h_rgb - 1u));
    uint32_t x0 = uint32_t(floor(u_c));
    uint32_t y0 = uint32_t(floor(v_c));
    uint32_t x1 = min(x0 + 1u, w_rgb - 1u);
    uint32_t y1 = min(y0 + 1u, h_rgb - 1u);
    float wx = u_c - float(x0);
    float wy = v_c - float(y0);
    uint32_t base_pix = v * (h_rgb * w_rgb);
    float3 c00 = load_rgb_u8((base_pix + y0 * w_rgb + x0) * rgb_ch);
    float3 c10 = load_rgb_u8((base_pix + y0 * w_rgb + x1) * rgb_ch);
    float3 c01 = load_rgb_u8((base_pix + y1 * w_rgb + x0) * rgb_ch);
    float3 c11 = load_rgb_u8((base_pix + y1 * w_rgb + x1) * rgb_ch);
    return lerp(lerp(c00, c10, wx), lerp(c01, c11, wx), wy);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform AppearanceInitUniforms uniforms
) {
    uint32_t i = dispatchThreadID.x;
    if (i >= uniforms.num_points)
        return;

    float3 xyz = float3(means_world[3u * i + 0u], means_world[3u * i + 1u], means_world[3u * i + 2u]);
    float3 fallback = clamp(
        float3(fallback_rgb[3u * i + 0u], fallback_rgb[3u * i + 1u], fallback_rgb[3u * i + 2u]),
        0.0f, 1.0f
    );

    bool has_norm = false;
    float3 unit_n = float3(0.0f);
    if (uniforms.has_normals != 0u) {
        float3 n_raw = float3(normals_world[3u * i + 0u], normals_world[3u * i + 1u], normals_world[3u * i + 2u]);
        float n_len = length(n_raw);
        if (n_len > 1e-8f) {
            has_norm = true;
            unit_n = n_raw / n_len;
        }
    }

    const uint32_t V = uniforms.num_views;
    const uint32_t h_rgb = uniforms.h_rgb;
    const uint32_t w_rgb = uniforms.w_rgb;
    const uint32_t h_d = uniforms.h_d;
    const uint32_t w_d = uniforms.w_d;
    const uint32_t rgb_ch = uniforms.rgb_channels;
    const bool use_deg1 = (uniforms.sh_degree >= 1u);

    // Symmetric 4x4 normal matrix A = sum w * b * b^T
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    float a22 = 0.0f, a23 = 0.0f;
    float a33 = 0.0f;

    // 4x3 RHS B = sum w * b * (rgb - 0.5)
    float3 b0 = float3(0.0f);
    float3 b1 = float3(0.0f);
    float3 b2 = float3(0.0f);
    float3 b3 = float3(0.0f);

    float3 rgb_wsum = float3(0.0f);
    float weight_sum = 0.0f;
    int32_t obs_count = 0;
    uint32_t valid_mask_lo = 0u;
    uint32_t valid_mask_hi = 0u;

    for (uint32_t v = 0u; v < V; ++v) {
        uint32_t vp = 6u * v;
        float4 r0 = view_params[vp + 0u];
        float4 r1 = view_params[vp + 1u];
        float4 r2 = view_params[vp + 2u];

        float x_cam = dot(r0.xyz, xyz) + r0.w;
        float y_cam = dot(r1.xyz, xyz) + r1.w;
        float z_cam = dot(r2.xyz, xyz) + r2.w;

        if (!isfinite(z_cam) || z_cam <= MIN_DEPTH_M)
            continue;

        float inv_z = 1.0f / z_cam;
        float xn = x_cam * inv_z;
        float yn = y_cam * inv_z;

        float4 k_rgb = view_params[vp + 4u];
        float4 k_depth = view_params[vp + 5u];

        float u_d = k_depth.x * xn + k_depth.z;
        float v_d = k_depth.y * yn + k_depth.w;
        int32_t u_di = int32_t(round(u_d));
        int32_t v_di = int32_t(round(v_d));

        float u_rgb = k_rgb.x * xn + k_rgb.z;
        float v_rgb = k_rgb.y * yn + k_rgb.w;

        if (u_di < 0 || u_di >= int32_t(w_d) || v_di < 0 || v_di >= int32_t(h_d) ||
            u_rgb < 0.0f || u_rgb > float(w_rgb - 1u) || v_rgb < 0.0f || v_rgb > float(h_rgb - 1u))
            continue;

        float d_obs = depth_views_f32[v * (h_d * w_d) + uint32_t(v_di) * w_d + uint32_t(u_di)];
        if (!isfinite(d_obs) || d_obs <= 0.0f || abs(z_cam - d_obs) > uniforms.depth_tol_m)
            continue;

        float3 cam_o = view_params[vp + 3u].xyz;
        float3 diff = cam_o - xyz;
        float dist_sq = dot(diff, diff);
        float inv_dist = rsqrt(max(dist_sq, 1e-12f));
        float3 view_dir = diff * inv_dist;
        float inv_dist_sq = 1.0f / max(dist_sq, MIN_DIST_SQ_M2);

        float weight = inv_dist_sq;
        if (has_norm) {
            float cos_angle = dot(unit_n, view_dir);
            if (cos_angle < uniforms.min_cos_angle)
                continue;
            weight = cos_angle * inv_dist_sq;
        }

        float3 rgb_obs = sample_rgb_bilinear(v, u_rgb, v_rgb, h_rgb, w_rgb, rgb_ch);
        float3 y_obs = rgb_obs - 0.5f;

        float f0 = SH_C0;
        float wf0 = weight * f0;
        a00 += wf0 * f0;
        b0 += wf0 * y_obs;

        if (use_deg1) {
            float f1 = -SH_C1 * view_dir.y;
            float f2 =  SH_C1 * view_dir.z;
            float f3 = -SH_C1 * view_dir.x;

            float wf1 = weight * f1;
            float wf2 = weight * f2;
            float wf3 = weight * f3;

            a01 += wf0 * f1;
            a02 += wf0 * f2;
            a03 += wf0 * f3;
            a11 += wf1 * f1;
            a12 += wf1 * f2;
            a13 += wf1 * f3;
            a22 += wf2 * f2;
            a23 += wf2 * f3;
            a33 += wf3 * f3;

            b1 += wf1 * y_obs;
            b2 += wf2 * y_obs;
            b3 += wf3 * y_obs;
        }

        rgb_wsum += weight * rgb_obs;
        weight_sum += weight;
        obs_count += 1;

        if (v < 32u) {
            valid_mask_lo |= (1u << v);
        } else if (v < 64u) {
            valid_mask_hi |= (1u << (v - 32u));
        }
    }

    // Weighted RGB initialization & DC prior
    bool observed = (obs_count > 0) && (weight_sum > 0.0f);
    float3 mean_rgb = clamp(rgb_wsum / max(weight_sum, 1e-12f), 0.0f, 1.0f);
    float3 rgb_init = observed ? mean_rgb : fallback;
    float3 c_prior0 = (rgb_init - 0.5f) / SH_C0;

    float3 x0 = c_prior0;
    float3 x1 = float3(0.0f);
    float3 x2 = float3(0.0f);
    float3 x3 = float3(0.0f);

    if (obs_count >= 2) {
        float reg_lam = uniforms.reg_lambda;
        float reg_eff = max(reg_lam, 1e-8f);
        if (!use_deg1) {
            x0 = (b0 + reg_lam * c_prior0) / max(a00 + reg_eff, 1e-12f);
        } else {
            // 4x4 Cholesky factorization of M = A + reg_eff * I
            float m00 = a00 + reg_eff;
            float m11 = a11 + reg_eff;
            float m22 = a22 + reg_eff;
            float m33 = a33 + reg_eff;

            float l00 = sqrt(max(m00, 1e-12f));
            float inv_l00 = 1.0f / l00;
            float l10 = a01 * inv_l00;
            float l20 = a02 * inv_l00;
            float l30 = a03 * inv_l00;

            float l11 = sqrt(max(m11 - l10 * l10, 1e-12f));
            float inv_l11 = 1.0f / l11;
            float l21 = (a12 - l20 * l10) * inv_l11;
            float l31 = (a13 - l30 * l10) * inv_l11;

            float l22 = sqrt(max(m22 - l20 * l20 - l21 * l21, 1e-12f));
            float inv_l22 = 1.0f / l22;
            float l32 = (a23 - l30 * l20 - l31 * l21) * inv_l22;

            float l33 = sqrt(max(m33 - l30 * l30 - l31 * l31 - l32 * l32, 1e-12f));
            float inv_l33 = 1.0f / l33;

            // Forward substitution: L * y = rhs
            float3 rhs0 = b0 + reg_lam * c_prior0;
            float3 y0 = rhs0 * inv_l00;
            float3 y1 = (b1 - l10 * y0) * inv_l11;
            float3 y2 = (b2 - l20 * y0 - l21 * y1) * inv_l22;
            float3 y3 = (b3 - l30 * y0 - l31 * y1 - l32 * y2) * inv_l33;

            // Back substitution: L^T * x = y
            x3 = y3 * inv_l33;
            x2 = (y2 - l32 * x3) * inv_l22;
            x1 = (y1 - l21 * x2 - l31 * x3) * inv_l11;
            x0 = (y0 - l10 * x1 - l20 * x2 - l30 * x3) * inv_l00;
        }
    }

    // Pass 2: exact weighted squared RGB residual for wls_sh_fit_rmse
    float se_num = 0.0f;
    if (obs_count > 0) {
        for (uint32_t v = 0u; v < V; ++v) {
            if (v < 32u) {
                if ((valid_mask_lo & (1u << v)) == 0u)
                    continue;
            } else if (v < 64u) {
                if ((valid_mask_hi & (1u << (v - 32u))) == 0u)
                    continue;
            }

            uint32_t vp = 6u * v;
            float4 r0 = view_params[vp + 0u];
            float4 r1 = view_params[vp + 1u];
            float4 r2 = view_params[vp + 2u];

            float x_cam = dot(r0.xyz, xyz) + r0.w;
            float y_cam = dot(r1.xyz, xyz) + r1.w;
            float z_cam = dot(r2.xyz, xyz) + r2.w;

            float inv_z = 1.0f / z_cam;
            float xn = x_cam * inv_z;
            float yn = y_cam * inv_z;

            float4 k_rgb = view_params[vp + 4u];
            float u_rgb = k_rgb.x * xn + k_rgb.z;
            float v_rgb = k_rgb.y * yn + k_rgb.w;

            if (v >= 64u) {
                if (!isfinite(z_cam) || z_cam <= MIN_DEPTH_M)
                    continue;
                float4 k_depth = view_params[vp + 5u];
                int32_t u_di = int32_t(round(k_depth.x * xn + k_depth.z));
                int32_t v_di = int32_t(round(k_depth.y * yn + k_depth.w));
                if (u_di < 0 || u_di >= int32_t(w_d) || v_di < 0 || v_di >= int32_t(h_d) ||
                    u_rgb < 0.0f || u_rgb > float(w_rgb - 1u) || v_rgb < 0.0f || v_rgb > float(h_rgb - 1u))
                    continue;
                float d_obs = depth_views_f32[v * (h_d * w_d) + uint32_t(v_di) * w_d + uint32_t(u_di)];
                if (!isfinite(d_obs) || d_obs <= 0.0f || abs(z_cam - d_obs) > uniforms.depth_tol_m)
                    continue;
            }

            float3 cam_o = view_params[vp + 3u].xyz;
            float3 diff = cam_o - xyz;
            float dist_sq = dot(diff, diff);
            float inv_dist = rsqrt(max(dist_sq, 1e-12f));
            float3 view_dir = diff * inv_dist;
            float inv_dist_sq = 1.0f / max(dist_sq, MIN_DIST_SQ_M2);

            float weight = inv_dist_sq;
            if (has_norm) {
                float cos_angle = dot(unit_n, view_dir);
                if (v >= 64u && cos_angle < uniforms.min_cos_angle)
                    continue;
                weight = cos_angle * inv_dist_sq;
            }

            float3 rgb_obs = sample_rgb_bilinear(v, u_rgb, v_rgb, h_rgb, w_rgb, rgb_ch);
            float3 pred = SH_C0 * x0;
            if (use_deg1) {
                pred += (-SH_C1 * view_dir.y) * x1
                      + ( SH_C1 * view_dir.z) * x2
                      + (-SH_C1 * view_dir.x) * x3;
            }
            pred = clamp(0.5f + pred, 0.0f, 1.0f);
            float3 dc = pred - rgb_obs;
            se_num += weight * (dot(dc, dc) * (1.0f / 3.0f));
        }
    }

    // Write outputs
    out_rgb_init[3u * i + 0u] = rgb_init.x;
    out_rgb_init[3u * i + 1u] = rgb_init.y;
    out_rgb_init[3u * i + 2u] = rgb_init.z;
    out_obs_count[i] = obs_count;
    out_weight_sum[i] = weight_sum;
    out_se_num[i] = se_num;

    uint32_t sh_base = 48u * i;
    out_sh_coeffs[sh_base + 0u] = x0.x;
    out_sh_coeffs[sh_base + 1u] = x0.y;
    out_sh_coeffs[sh_base + 2u] = x0.z;
    out_sh_coeffs[sh_base + 3u] = x1.x;
    out_sh_coeffs[sh_base + 4u] = x1.y;
    out_sh_coeffs[sh_base + 5u] = x1.z;
    out_sh_coeffs[sh_base + 6u] = x2.x;
    out_sh_coeffs[sh_base + 7u] = x2.y;
    out_sh_coeffs[sh_base + 8u] = x2.z;
    out_sh_coeffs[sh_base + 9u] = x3.x;
    out_sh_coeffs[sh_base + 10u] = x3.y;
    out_sh_coeffs[sh_base + 11u] = x3.z;
    for (uint32_t k = 12u; k < 48u; ++k) {
        out_sh_coeffs[sh_base + k] = 0.0f;
    }
}
