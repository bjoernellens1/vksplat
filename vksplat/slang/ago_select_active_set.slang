// GPU-resident AGO-2 active-set compaction & step-gain summary shader.
//
// Reads per-Gaussian AGO telemetry (2 float4s = 8 floats/Gaussian):
//   ago_telemetry[2*g + 0] = (grad_norm, update_norm, predicted_gain, abs_proj_grad)
//   ago_telemetry[2*g + 1] = (update_count, contrib_sum, last_step_updated, last_predicted_gain)
// and per-splat / group freeze state, computes utility score:
//   base_gain = (last_predicted_gain != 0.0) ? last_predicted_gain : predicted_gain
//   score     = base_gain + 0.25 * abs_proj_grad
// excludes frozen splats (when exclude_frozen != 0), always includes newborns
// (update_count < newborn_warmup_steps) or splats with score >= gain_threshold,
// atomically compacts selected indices into `active_indices`, and writes a 16-byte
// summary into `summary_buf`:
//   byte  0 (uint32): active_count
//   byte  4 (float):  total_predicted_gain (sum of max(score, 0.0) over eligible splats)
//   byte  8 (uint32): eligible_count (non-frozen splats)
//   byte 12 (float):  max_predicted_gain (max of max(score, 0.0) over eligible splats)

#include "config.slang"

static const uint32_t BLOCK_SIZE = 256u;
static const uint32_t SPLAT_STATE_CONTRADICTED = 4u;
static const uint32_t SPLAT_STATE_LEGACY_GROUP = 0xFFFFFFFFu;

struct AgoSelectUniforms {
    uint32_t num_splats;
    uint32_t num_groups;
    uint32_t protect_locked_groups;
    uint32_t exclude_frozen;
    float gain_threshold;
    float newborn_warmup_steps;
    uint32_t _pad0;
    uint32_t _pad1;
};

layout(binding=0) StructuredBuffer<float4> ago_telemetry;
layout(binding=1) StructuredBuffer<uint32_t> splat_state;
layout(binding=2) StructuredBuffer<int32_t> group_ids;
layout(binding=3) StructuredBuffer<uint32_t> group_state;
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
        bool is_frozen = false;
        if (u.exclude_frozen != 0u) {
            const uint32_t raw_st = splat_state[g_idx];
            const uint32_t st = splat_base_state(raw_st);
            const int32_t gid = group_ids[g_idx];
            const uint32_t gs = (gid >= 0 && uint32_t(gid) < u.num_groups) ? group_state[gid] : 0u;
            if (gs == 3u) {
                is_frozen = true;  // RETRACTED group
            } else if (st == SPLAT_STATE_CONTRADICTED) {
                is_frozen = true;  // CONTRADICTED splat
            } else if (st == SPLAT_STATE_LEGACY_GROUP && gs == 2u && u.protect_locked_groups != 0u) {
                is_frozen = true;  // protected LOCKED structural group
            }
        }

        if (!is_frozen) {
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
            selected = is_newborn || (score >= u.gain_threshold);
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
