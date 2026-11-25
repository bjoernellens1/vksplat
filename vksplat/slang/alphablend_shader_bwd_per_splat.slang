#undef SUBTILE_SIZE
#define SUBTILE_SIZE 0

// Time: (T/P) * (N+S)/S * (P+S)
// P: larger is better, up to TILE_SIZE
// S: closer to sqrt(N*P) is better
// S should be a multiple of SUBGROUP_SIZE, round up

#define PIXEL_BATCH_SIZE TILE_SIZE

static const uint MAX_SPLAT_BATCH_SIZE = 128;


groupshared float pixel_T[PIXEL_BATCH_SIZE];
groupshared float3 v_pixel_rgb[PIXEL_BATCH_SIZE];
groupshared float v_pixel_T[PIXEL_BATCH_SIZE];
groupshared int n_contrib_fwd[PIXEL_BATCH_SIZE];
groupshared float pixel_d_exp[PIXEL_BATCH_SIZE];
groupshared float pixel_g_D_eff[PIXEL_BATCH_SIZE];
groupshared int sg_max_contrib[MAX_SPLAT_BATCH_SIZE / SUBGROUP_SIZE];
groupshared uint tile_has_active_splat;

layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) StructuredBuffer<float4> final_pixel_state;
layout(binding=6) StructuredBuffer<int32_t> n_contributors;
layout(binding=7) StructuredBuffer<float4> v_current_pixel_state;

layout(binding=8) RWByteAddressBuffer out_v_xy_vs;
layout(binding=9) RWByteAddressBuffer out_v_inv_cov_vs_opacity;
layout(binding=10) RWByteAddressBuffer out_v_rgb;
layout(binding=11) StructuredBuffer<uint32_t> active_tiles;  // compacted tile list when uniforms.use_tile_list
layout(binding=12) StructuredBuffer<int32_t> group_ids;     // read only when uniforms.freeze_flags != 0
layout(binding=13) StructuredBuffer<uint32_t> group_state;
layout(binding=14) StructuredBuffer<uint32_t> splat_state;
layout(binding=15) RWByteAddressBuffer out_ago_contrib;
layout(binding=16) RWByteAddressBuffer out_ago_abs_grad;
layout(binding=17) StructuredBuffer<float> depths;
layout(binding=18) StructuredBuffer<float4> final_depth_state;
layout(binding=19) StructuredBuffer<float2> train_depth_weight;
layout(binding=20) RWByteAddressBuffer out_v_depths;
layout(binding=21) RWStructuredBuffer<uint32_t> out_ago_gpu_counters; // S104: written only when (uniforms.render_exclude_flags & 4u) != 0


[shader("compute")]
[numthreads(MAX_SPLAT_BATCH_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint thread_rank = groupThreadID.x;
    uint tile_idx = uniforms.use_tile_list != 0 ? active_tiles[groupID.x] : groupID.y * grid_width + groupID.x;
    uint2 tbound0 = uint2(tile_idx%grid_width, tile_idx/grid_width) * uint2(TILE_WIDTH, TILE_HEIGHT);

    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];
    if (tile_idx_start >= tile_idx_end)
        return;

    // S104: Active-tile backward early-exit — skip frozen-only tiles before reading pixel buffers.
    if (uniforms.freeze_flags != 0u) {
        if (thread_rank == 0u) {
            tile_has_active_splat = 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        bool local_has_active = false;
        for (uint32_t s = tile_idx_start + thread_rank; s < tile_idx_end; s += MAX_SPLAT_BATCH_SIZE) {
            int32_t g_id = sorted_gauss_idx[s];
            if (splat_freeze_mode(g_id, uniforms.freeze_flags, uniforms.num_groups, group_ids, group_state, splat_state) != 2u) {
                local_has_active = true;
                break;
            }
        }
        if (WaveActiveAnyTrue(local_has_active) && WaveIsFirstLane()) {
            InterlockedOr(tile_has_active_splat, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (tile_has_active_splat == 0u) {
            if ((uniforms.render_exclude_flags & 4u) != 0u && thread_rank == 0u) {
                InterlockedAdd(out_ago_gpu_counters[3], 1u);
            }
            return;
        }
    }
    if ((uniforms.render_exclude_flags & 4u) != 0u && thread_rank == 0u) {
        InterlockedAdd(out_ago_gpu_counters[2], 1u);
    }

    const bool ago_enabled = (uniforms.ago_telemetry_enabled != 0u);
    const bool depth_enabled = (uniforms.depth_loss_scale > 0.0f);

    int local_max_contrib = 0;
    [ForceUnroll]
    for (int i = 0; i < TILE_SIZE; i += MAX_SPLAT_BATCH_SIZE) {
        int pix_rank_batch = i + thread_rank;

        int pix_rank = pix_rank_batch;
        uint2 pix_coord = tbound0 + uint2(pix_rank%TILE_WIDTH, pix_rank/TILE_WIDTH);
        bool is_inside = (pix_coord.x < W && pix_coord.y < H);

        if (is_inside) {
            uint32_t pix_id = pix_coord.y * W + pix_coord.x;
            pixel_T[pix_rank_batch] = final_pixel_state[pix_id].a;
            float4 vp = v_current_pixel_state[pix_id];
            v_pixel_rgb[pix_rank_batch] = vp.rgb;
            v_pixel_T[pix_rank_batch] = vp.w;
            int nc = n_contributors[pix_id];
            n_contrib_fwd[pix_rank_batch] = nc;
            local_max_contrib = max(local_max_contrib, nc);
            if (depth_enabled) {
                float4 pd = final_depth_state[pix_id];
                float d_exp_norm = pd.x;
                float acc_alpha = pd.z;
                float2 dw = train_depth_weight[pix_id];
                if (dw.x > 0.0f && dw.y > 0.0f && acc_alpha > 0.1f) {
                    float d_exp_m = d_exp_norm * uniforms.depth_inv_scale;
                    float r = d_exp_m - dw.x;
                    float delta = uniforms.depth_huber_delta;
                    float d_huber = clamp(r, -delta, delta) / delta;
                    float g_D_norm = (uniforms.depth_loss_scale * dw.y * d_huber) * uniforms.depth_inv_scale;
                    pixel_g_D_eff[pix_rank_batch] = g_D_norm / acc_alpha;
                    pixel_d_exp[pix_rank_batch] = d_exp_norm;
                } else {
                    pixel_g_D_eff[pix_rank_batch] = 0.0f;
                    pixel_d_exp[pix_rank_batch] = 0.0f;
                }
            }
        }
        else {
            n_contrib_fwd[pix_rank_batch] = 0;
            if (depth_enabled) {
                pixel_g_D_eff[pix_rank_batch] = 0.0f;
                pixel_d_exp[pix_rank_batch] = 0.0f;
            }
        }

    }
    int sg_max = WaveActiveMax(local_max_contrib);
    if (WaveIsFirstLane()) {
        sg_max_contrib[thread_rank / SUBGROUP_SIZE] = sg_max;
    }
    GroupMemoryBarrierWithGroupSync();
    uint lane_id = thread_rank % SUBGROUP_SIZE;
    int tile_max_n_contrib = WaveActiveMax(
        (lane_id < (MAX_SPLAT_BATCH_SIZE / SUBGROUP_SIZE)) ? sg_max_contrib[lane_id] : 0
    );
    GroupMemoryBarrierWithGroupSync();

    tile_idx_end = tile_idx_start + min(tile_idx_end - tile_idx_start, uint32_t(max(tile_max_n_contrib, 0)));
    uint32_t num_splats_in_tile = tile_idx_end - tile_idx_start;
    if (num_splats_in_tile == 0)
        return;

    #ifdef SPLAT_BATCH_SIZE
      #undef SPLAT_BATCH_SIZE
    #endif
    int SPLAT_BATCH_SIZE = int(sqrt(num_splats_in_tile * PIXEL_BATCH_SIZE));
    SPLAT_BATCH_SIZE = ((SPLAT_BATCH_SIZE + SUBGROUP_SIZE) / SUBGROUP_SIZE) * SUBGROUP_SIZE;
    SPLAT_BATCH_SIZE = min(SPLAT_BATCH_SIZE, MAX_SPLAT_BATCH_SIZE);
    bool active = (thread_rank < SPLAT_BATCH_SIZE);

    const int rounds = ((num_splats_in_tile + SPLAT_BATCH_SIZE - 1) / SPLAT_BATCH_SIZE);

    [ForceUnroll]
    for (int pixel_batch_base = 0; pixel_batch_base < TILE_SIZE; pixel_batch_base += PIXEL_BATCH_SIZE) {

    for (int batch_idx = 0; batch_idx < rounds; batch_idx++) {

        // load splat
        // thread 0 loads last splat, thread 1 loads second last, ...
        int32_t splat_id = -1;
        Splat_2D_AlphaBlend splat = { float4(0, 0, 0, 1), float2(0), float3(0) };
        float splat_depth = 0.0f;
        int32_t progress = batch_idx * SPLAT_BATCH_SIZE + thread_rank;
        int32_t splat_gid = tile_idx_end - progress - 1;
        int32_t splat_tid = splat_gid - tile_idx_start;
        if (active & (splat_tid >= 0)) {
            splat_id = sorted_gauss_idx[splat_gid];
            splat = load_splat_alphablend(splat_id, xy_vs, inv_cov_vs_opacity, rgb);
            if (depth_enabled) {
                splat_depth = depths[splat_id];
            }
        }

        // compute subtile culling masks
        #if SUBTILE_SIZE > 0
        uint32_t subtile_overlap_bitmask = 0;
        if (active) subtile_overlap_bitmask = computeTileOverlapBitmask(
            false, splat.xy_vs - tbound0, splat.inv_cov_vs_opacity);
        #endif

        float2 v_xy_vs = float2(0.0);
        float4 v_inv_cov_vs_opacity = float4(0.0);
        float3 v_rgb = float3(0.0);
        float v_depth = 0.0;
        float ago_contrib = 0.0;
        float ago_abs_grad = 0.0;

        int num_pixels = PIXEL_BATCH_SIZE;
        int num_splats = min(SPLAT_BATCH_SIZE, num_splats_in_tile - batch_idx * SPLAT_BATCH_SIZE);

        for (int i = 0; i < num_splats + num_pixels - 1; ++i, GroupMemoryBarrierWithGroupSync()) {

            int splat_rank = thread_rank;
            int pixel_rank = i - thread_rank;
            if (!active || splat_rank >= num_splats || pixel_rank >= num_pixels || pixel_rank < 0)
                continue;

            int pixel_idx = pixel_batch_base + pixel_rank;
            uint2 pix_coord = tbound0 + uint2(pixel_idx%TILE_WIDTH, pixel_idx/TILE_WIDTH);
            uint pix_rank = pixel_rank;
          #if SUBTILE_SIZE > 0
            uint subgrid_idx;
            decodeThreadToSubtiles(pixel_rank, pix_coord, subgrid_idx, pix_rank);
            int overlap = (subtile_overlap_bitmask >> subgrid_idx) & 1;
            if (overlap == 0)
                continue;
          #endif

            uint32_t n_contrib = n_contrib_fwd[pix_rank];
            if (splat_tid < 0 || splat_tid >= n_contrib)
                continue;

            float2 d = float2(pix_coord) - splat.xy_vs;
            float3 g_inv_cov = splat.inv_cov_vs_opacity.xyz;
            float g_opacity = splat.inv_cov_vs_opacity.w;
            float power = -0.5f * (g_inv_cov.x * d.x * d.x + g_inv_cov.z * d.y * d.y + 2.0f * g_inv_cov.y * d.x * d.y);
            float val = exp(power);
            float raw_alpha = g_opacity * val;
            float a = min(ALPHA_BLEND_MAX_OPACITY, raw_alpha);
            if (a <= ALPHA_THRESHOLD)
                continue;

            float T = pixel_T[pix_rank] / (1.0f - a);
            pixel_T[pix_rank] = T;

            float3 v_rgb_pix = v_pixel_rgb[pix_rank];
            float v_T = v_pixel_T[pix_rank];
            float aT = a * T;
            float rgbd = dot(v_rgb_pix, splat.rgb);
            float total_d = rgbd;
            if (depth_enabled) {
                float g_D_eff = pixel_g_D_eff[pix_rank];
                if (g_D_eff != 0.0f) {
                    total_d += g_D_eff * (splat_depth - pixel_d_exp[pix_rank]);
                    v_depth += aT * g_D_eff;
                }
            }
            v_pixel_T[pix_rank] = v_T * (1.0f - a) + a * total_d;

            v_rgb += aT * v_rgb_pix;
            float v_a = (raw_alpha < ALPHA_BLEND_MAX_OPACITY) ? (T * (total_d - v_T)) : 0.0f;
            float v_power = v_a * a;
            v_inv_cov_vs_opacity += float4(
                -0.5f * d.x * d.x * v_power,
                -d.x * d.y * v_power,
                -0.5f * d.y * d.y * v_power,
                v_a * val
            );
            float2 d_xy_pix = float2(
                (g_inv_cov.x * d.x + g_inv_cov.y * d.y) * v_power,
                (g_inv_cov.y * d.x + g_inv_cov.z * d.y) * v_power
            );
            v_xy_vs += d_xy_pix;
            if (ago_enabled) {
                ago_contrib += aT;
                ago_abs_grad += length(d_xy_pix);
            }
        }

      #if USE_EMULATED_F32_ATOMIC
        #define _ATOMIC_ADD(buffer, address, value) \
            if (value != 0.0) AtomicAddFloat(buffer, (address)*sizeof(float), value);
      #else
        #define _ATOMIC_ADD(buffer, address, value) \
            if (value != 0.0) buffer.InterlockedAddF32((address)*sizeof(float), value);
      #endif
        if (active && splat_id >= 0) {
            const uint freeze = splat_freeze_mode(splat_id, uniforms.freeze_flags, uniforms.num_groups, group_ids, group_state, splat_state);
            if (freeze == 0) {
                _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+0, v_xy_vs.x);
                _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+1, v_xy_vs.y);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+0, v_inv_cov_vs_opacity.x);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+1, v_inv_cov_vs_opacity.y);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+2, v_inv_cov_vs_opacity.z);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+3, v_inv_cov_vs_opacity.w);
                if (depth_enabled) {
                    _ATOMIC_ADD(out_v_depths, splat_id, v_depth);
                }
            }
            if (freeze != 2) {
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+0, float(v_rgb.x));
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+1, float(v_rgb.y));
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+2, float(v_rgb.z));
            }
            if (ago_enabled) {
                _ATOMIC_ADD(out_ago_contrib, splat_id, ago_contrib);
                if (freeze == 0) {
                    _ATOMIC_ADD(out_ago_abs_grad, splat_id, ago_abs_grad);
                }
            }
        }

    }  // for (int i = 0; i < rounds; i++)

    }  // pixel batch
}
