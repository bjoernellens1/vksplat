// GPU-resident AGO-2 active-set compaction, 256-bin log-gain histogram, & step-gain summary shader.
//
// ENTRY == 0 (ago_select_active_set):
//   Reads per-Gaussian AGO telemetry (2 float4s = 8 floats/Gaussian):
//     ago_telemetry[2*g + 0] = (grad_norm, update_norm, predicted_gain, abs_proj_grad)
//     ago_telemetry[2*g + 1] = (update_count, contrib_sum, last_step_updated, last_predicted_gain)
//   and per-splat / group freeze state, computes utility score:
//     base_gain = (last_predicted_gain != 0.0) ? last_predicted_gain : predicted_gain
//     score     = base_gain + 0.25 * abs_proj_grad
//   Supports three selection modes:
//     mode == 0: legacy threshold (is_newborn || score >= gain_threshold)
//     mode == 1: exact-K (all newborns + mature with bin > cutoff_bin + atomic boundary_quota in bin == cutoff_bin)
//     mode == 2: exact-K newborns-only (newborns with bin > cutoff_bin + atomic boundary_quota in bin == cutoff_bin)
//   Atomically compacts selected indices into `active_indices`, and writes a 20-byte
//   summary into `summary_buf`:
//     byte  0 (uint32): active_count
//     byte  4 (float):  total_predicted_gain (sum of max(score, 0.0) over eligible splats)
//     byte  8 (uint32): eligible_count (non-frozen splats)
//     byte 12 (float):  max_predicted_gain (max of max(score, 0.0) over eligible splats)
//     byte 16 (uint32): boundary_claimed (atomic counter for exact-K boundary bin quota)
//
// ENTRY == 1 (ago_gain_histogram):
//   Computes 256-bin log2-gain histograms for mature (bins 0..255) and newborn (bins 256..511)
//   eligible splats in shared memory + global atomic merge, plus 16-byte header:
//     byte  0 (uint32): eligible_count
//     byte  4 (float):  total_predicted_gain
//     byte  8 (uint32): newborn_count
//     byte 12 (float):  max_predicted_gain
//     byte 16 + 4*b (uint32, b in 0..511): bin counts

#include "config.slang"

#ifndef ENTRY
#define ENTRY 0
#endif

static const uint32_t BLOCK_SIZE = 256u;
static const uint32_t SPLAT_STATE_CONTRADICTED = 4u;
static const uint32_t SPLAT_STATE_LEGACY_GROUP = 0xFFFFFFFFu;
static const float AGO_HIST_LOG2_MIN = -30.0f;
static const float AGO_HIST_LOG2_MAX = 10.0f;
static const float AGO_HIST_INV_RANGE = 1.0f / (AGO_HIST_LOG2_MAX - AGO_HIST_LOG2_MIN);

struct AgoSelectUniforms {
    uint32_t num_splats;
    uint32_t num_groups;
    uint32_t protect_locked_groups;
    uint32_t exclude_frozen;
    float gain_threshold;
    float newborn_warmup_steps;
    uint32_t mode;
    uint32_t cutoff_bin;
    uint32_t boundary_quota;
    uint32_t _pad0;
};

layout(binding=0) StructuredBuffer<float4> ago_telemetry;
layout(binding=1) StructuredBuffer<uint32_t> splat_state;
layout(binding=2) StructuredBuffer<int32_t> group_ids;
layout(binding=3) StructuredBuffer<uint32_t> group_state;

bool ago_is_frozen(uint32_t g_idx, AgoSelectUniforms u) {
    if (u.exclude_frozen == 0u) {
        return false;
    }
    const uint32_t raw_st = splat_state[g_idx];
    const uint32_t st = splat_base_state(raw_st);
    const int32_t gid = group_ids[g_idx];
    const uint32_t gs = (gid >= 0 && uint32_t(gid) < u.num_groups) ? group_state[gid] : 0u;
    if (gs == 3u) {
        return true;  // RETRACTED group
    }
    if (st == SPLAT_STATE_CONTRADICTED) {
        return true;  // CONTRADICTED splat
    }
    if (st == SPLAT_STATE_LEGACY_GROUP && gs == 2u && u.protect_locked_groups != 0u) {
        return true;  // protected LOCKED structural group
    }
    return false;
}

uint32_t ago_gain_to_bin(float gain_pos) {
    if (gain_pos <= 0.0f || !isfinite(gain_pos)) {
        return 0u;
    }
    float t = (log2(gain_pos) - AGO_HIST_LOG2_MIN) * AGO_HIST_INV_RANGE;
    t = clamp(t, 0.0f, 1.0f);
    return min(1u + uint32_t(t * 254.0f), 255u);
}

#if ENTRY == 0

layout(binding=4) RWStructuredBuffer<uint32_t> active_indices;
layout(binding=5) RWByteAddressBuffer summary_buf;

groupshared uint32_t s_wg_count;
groupshared uint32_t s_wg_base;
groupshared uint32_t s_elig[BLOCK_SIZE];
groupshared float s_gain[BLOCK_SIZE];
groupshared float s_max_gain[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform AgoSelectUniforms u
) {
    const uint32_t lid = groupThreadID.x;
    const uint32_t g_idx = dispatchThreadID.x;

    bool selected = false;
    uint32_t eligible = 0u;
    float gain_pos = 0.0f;

    if (g_idx < u.num_splats) {
        if (!ago_is_frozen(g_idx, u)) {
            eligible = 1u;
            const uint32_t base = 2u * g_idx;
            const float4 t0 = ago_telemetry[base + 0u];
            const float4 t1 = ago_telemetry[base + 1u];
            const float base_gain = (t1.w != 0.0f) ? t1.w : t0.z;
            float score = base_gain + 0.25f * t0.w;
            if (!isfinite(score)) {
                score = 0.0f;
            }
            gain_pos = max(score, 0.0f);
            const bool is_newborn = (t1.x < u.newborn_warmup_steps);

            if (u.mode == 0u) {
                selected = is_newborn || (score >= u.gain_threshold);
            } else if (u.mode == 1u) {
                if (is_newborn) {
                    selected = true;
                } else {
                    const uint32_t bin = ago_gain_to_bin(gain_pos);
                    if (bin > u.cutoff_bin) {
                        selected = true;
                    } else if (bin == u.cutoff_bin && u.boundary_quota > 0u) {
                        uint32_t slot = 0u;
                        summary_buf.InterlockedAdd(16u, 1u, slot);
                        selected = (slot < u.boundary_quota);
                    }
                }
            } else {
                // mode == 2u: exact-K newborns-only (newborn_count > K)
                if (is_newborn) {
                    const uint32_t bin = ago_gain_to_bin(gain_pos);
                    if (bin > u.cutoff_bin) {
                        selected = true;
                    } else if (bin == u.cutoff_bin && u.boundary_quota > 0u) {
                        uint32_t slot = 0u;
                        summary_buf.InterlockedAdd(16u, 1u, slot);
                        selected = (slot < u.boundary_quota);
                    }
                }
            }
        }
    }

    // Workgroup-level compaction into active_indices (1 global atomic per workgroup)
    if (lid == 0u) {
        s_wg_count = 0u;
        s_wg_base = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint32_t local_slot = 0u;
    if (selected) {
        InterlockedAdd(s_wg_count, 1u, local_slot);
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u && s_wg_count > 0u) {
        summary_buf.InterlockedAdd(0u, s_wg_count, s_wg_base);
    }
    GroupMemoryBarrierWithGroupSync();

    if (selected) {
        active_indices[s_wg_base + local_slot] = g_idx;
    }

    // Workgroup-level reduction for eligible_count, total_predicted_gain, max_predicted_gain
    s_elig[lid] = eligible;
    s_gain[lid] = gain_pos;
    s_max_gain[lid] = gain_pos;
    GroupMemoryBarrierWithGroupSync();

    for (uint32_t stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_elig[lid] += s_elig[lid + stride];
            s_gain[lid] += s_gain[lid + stride];
            s_max_gain[lid] = max(s_max_gain[lid], s_max_gain[lid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0u) {
        if (s_elig[0] > 0u) {
            summary_buf.InterlockedAdd(8u, s_elig[0]);
        }
        if (s_gain[0] > 0.0f) {
#if USE_EMULATED_F32_ATOMIC
            AtomicAddFloat(summary_buf, 4u, s_gain[0]);
#else
            summary_buf.InterlockedAddF32(4u, s_gain[0]);
#endif
        }
        if (s_max_gain[0] > 0.0f) {
            summary_buf.InterlockedMax(12u, asuint(s_max_gain[0]));
        }
    }
}

#elif ENTRY == 1

layout(binding=4) RWByteAddressBuffer hist_buf;

groupshared uint32_t s_hist[512];
groupshared uint32_t s_elig[BLOCK_SIZE];
groupshared uint32_t s_nb[BLOCK_SIZE];
groupshared float s_gain[BLOCK_SIZE];
groupshared float s_max_gain[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform AgoSelectUniforms u
) {
    const uint32_t lid = groupThreadID.x;
    const uint32_t g_idx = dispatchThreadID.x;

    s_hist[lid] = 0u;
    s_hist[lid + 256u] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint32_t eligible = 0u;
    uint32_t newborn = 0u;
    float gain_pos = 0.0f;

    if (g_idx < u.num_splats) {
        if (!ago_is_frozen(g_idx, u)) {
            eligible = 1u;
            const uint32_t base = 2u * g_idx;
            const float4 t0 = ago_telemetry[base + 0u];
            const float4 t1 = ago_telemetry[base + 1u];
            const float base_gain = (t1.w != 0.0f) ? t1.w : t0.z;
            float score = base_gain + 0.25f * t0.w;
            if (!isfinite(score)) {
                score = 0.0f;
            }
            gain_pos = max(score, 0.0f);
            const bool is_newborn = (t1.x < u.newborn_warmup_steps);
            newborn = is_newborn ? 1u : 0u;
            const uint32_t bin = ago_gain_to_bin(gain_pos);
            const uint32_t h_idx = (is_newborn ? 256u : 0u) + bin;
            InterlockedAdd(s_hist[h_idx], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint32_t c0 = s_hist[lid];
    if (c0 > 0u) {
        hist_buf.InterlockedAdd(16u + 4u * lid, c0);
    }
    const uint32_t c1 = s_hist[lid + 256u];
    if (c1 > 0u) {
        hist_buf.InterlockedAdd(16u + 4u * (lid + 256u), c1);
    }

    s_elig[lid] = eligible;
    s_nb[lid] = newborn;
    s_gain[lid] = gain_pos;
    s_max_gain[lid] = gain_pos;
    GroupMemoryBarrierWithGroupSync();

    for (uint32_t stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            s_elig[lid] += s_elig[lid + stride];
            s_nb[lid] += s_nb[lid + stride];
            s_gain[lid] += s_gain[lid + stride];
            s_max_gain[lid] = max(s_max_gain[lid], s_max_gain[lid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0u) {
        if (s_elig[0] > 0u) {
            hist_buf.InterlockedAdd(0u, s_elig[0]);
        }
        if (s_gain[0] > 0.0f) {
#if USE_EMULATED_F32_ATOMIC
            AtomicAddFloat(hist_buf, 4u, s_gain[0]);
#else
            hist_buf.InterlockedAddF32(4u, s_gain[0]);
#endif
        }
        if (s_nb[0] > 0u) {
            hist_buf.InterlockedAdd(8u, s_nb[0]);
        }
        if (s_max_gain[0] > 0.0f) {
            hist_buf.InterlockedMax(12u, asuint(s_max_gain[0]));
        }
    }
}

#endif
