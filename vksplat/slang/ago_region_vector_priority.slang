// GPU-resident 6-component RegionPriorityVector accumulator and bounded
// structural (split / prune) candidate selector for AGO Phase 11-B.
//
// Reads live GPU telemetry, scales/opacities, splat_state, group_ids, and group_state:
//   ago_telemetry[2*g + 0] = (grad_norm, update_norm, predicted_gain, abs_proj_grad)
//   ago_telemetry[2*g + 1] = (update_count, contrib_sum, last_step_updated, last_predicted_gain)
//   scales_opacs[g]        = (log_sx, log_sy, log_sz, opacity_logit)
//
// Writes:
//   1. `region_accum_buf` (8 uint32 words = 32 bytes per region slot r in [0, num_regions)):
//        word 0: splat_count
//        word 1: eligible_count
//        word 2: u_geom_q   (fixed-point 1e4)
//        word 3: u_app_q    (fixed-point 1e4)
//        word 4: u_split_q  (fixed-point 1e4)
//        word 5: u_birth_q  (fixed-point 1e4)
//        word 6: u_merge_q  (fixed-point 1e4)
//        word 7: u_corr_q   (fixed-point 1e4)
//   2. `candidate_buf`:
//        word 0: split_candidates_total (atomic count; indices stored up to max_split)
//        word 1: prune_candidates_total (atomic count; indices stored up to max_prune)
//        words [2 .. 2 + max_split): split_candidate_indices
//        words [2 + max_split .. 2 + max_split + max_prune): prune_candidate_indices

#include "config.slang"

static const uint32_t BLOCK_SIZE = 256u;
static const uint32_t SPLAT_STATE_CONTRADICTED = 4u;
static const uint32_t SPLAT_STATE_LEGACY_GROUP = 0xFFFFFFFFu;
static const float FIXED_SCALE = 10000.0f;

struct AgoRegionPriorityUniforms {
    uint32_t num_splats;
    uint32_t num_regions;          // >= 1; region_id = clamp(group_id, 0, num_regions - 1)
    uint32_t num_groups;
    uint32_t protect_locked_groups;
    uint32_t max_split;
    uint32_t max_prune;
    float split_scale_thresh;
    float split_grad_thresh;
    float prune_opacity_thresh;
    float merge_scale_thresh;
    float newborn_warmup_steps;
    uint32_t _pad0;
};

layout(binding=0) StructuredBuffer<float4> ago_telemetry;
layout(binding=1) StructuredBuffer<float4> scales_opacs;
layout(binding=2) StructuredBuffer<uint32_t> splat_state;
layout(binding=3) StructuredBuffer<int32_t> group_ids;
layout(binding=4) StructuredBuffer<uint32_t> group_state;
layout(binding=5) RWByteAddressBuffer region_accum_buf;
layout(binding=6) RWByteAddressBuffer candidate_buf;

uint32_t quantize_u32(float v) {
    if (!isfinite(v) || v <= 0.0f) {
        return 0u;
    }
    const float clamped = min(v * FIXED_SCALE, 1000000.0f);
    return uint32_t(clamped + 0.5f);
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform AgoRegionPriorityUniforms u
) {
    const uint32_t g_idx = dispatchThreadID.x;
    if (g_idx >= u.num_splats || u.num_regions == 0u) {
        return;
    }

    const int32_t raw_gid = group_ids[g_idx];
    const uint32_t region_idx = (raw_gid >= 0 && uint32_t(raw_gid) < u.num_regions)
        ? uint32_t(raw_gid)
        : 0u;

    const uint32_t raw_st = splat_state[g_idx];
    const uint32_t st = splat_base_state(raw_st);
    const uint32_t gs = (raw_gid >= 0 && uint32_t(raw_gid) < u.num_groups)
        ? group_state[raw_gid]
        : 0u;

    const bool is_retracted = (gs == 3u);
    const bool is_contradicted = (st == SPLAT_STATE_CONTRADICTED);
    const bool is_locked = (st == SPLAT_STATE_LEGACY_GROUP && gs == 2u && u.protect_locked_groups != 0u);
    const bool is_eligible = !(is_retracted || is_contradicted || is_locked);

    const uint32_t base = g_idx * 2u;
    const float4 t0 = ago_telemetry[base + 0u];
    const float4 t1 = ago_telemetry[base + 1u];
    const float4 so = scales_opacs[g_idx];

    const float grad_norm = isfinite(t0.x) ? max(t0.x, 0.0f) : 0.0f;
    const float update_norm = isfinite(t0.y) ? max(t0.y, 0.0f) : 0.0f;
    const float pred_gain = isfinite(t0.z) ? max(t0.z, 0.0f) : 0.0f;
    const float abs_proj_grad = isfinite(t0.w) ? max(t0.w, 0.0f) : 0.0f;
    const float update_count = isfinite(t1.x) ? max(t1.x, 0.0f) : 0.0f;
    const float contrib_sum = isfinite(t1.y) ? max(t1.y, 0.0f) : 0.0f;
    const float last_pred_gain = isfinite(t1.w) ? max(t1.w, 0.0f) : 0.0f;
    const float eff_gain = (last_pred_gain > 0.0f) ? last_pred_gain : pred_gain;

    const float logit = isfinite(so.w) ? clamp(so.w, -15.0f, 15.0f) : 0.0f;
    const float opacity = 1.0f / (1.0f + exp(-logit));

    const float log_max_s = max(so.x, max(so.y, so.z));
    const float max_scale = isfinite(log_max_s) ? exp(clamp(log_max_s, -10.0f, 4.0f)) : 0.0f;

    // 6-component priority vector U_p
    const float u_geom = abs_proj_grad + 0.5f * grad_norm;
    const float u_app = eff_gain + 0.25f * update_norm;
    const bool split_cond = (max_scale >= u.split_scale_thresh) &&
                            (abs_proj_grad >= u.split_grad_thresh || grad_norm >= u.split_grad_thresh);
    const float u_split = split_cond ? (max(abs_proj_grad, grad_norm) * max_scale) : 0.0f;
    const float u_birth = (update_count < u.newborn_warmup_steps) ? (eff_gain + 0.1f) : 0.0f;
    const bool prune_cond = (opacity <= u.prune_opacity_thresh);
    const float u_merge = (prune_cond || max_scale <= u.merge_scale_thresh) ? (1.0f - opacity) : 0.0f;
    const float u_corr = is_contradicted
        ? (grad_norm + 1.0f)
        : ((grad_norm > 0.0f && contrib_sum < 1e-4f) ? grad_norm : 0.0f);

    const uint32_t byte_off = region_idx * 32u;
    uint32_t prev = 0u;
    region_accum_buf.InterlockedAdd(byte_off + 0u, 1u, prev);
    if (is_eligible) {
        region_accum_buf.InterlockedAdd(byte_off + 4u, 1u, prev);
        const uint32_t q_geom = quantize_u32(u_geom);
        const uint32_t q_app = quantize_u32(u_app);
        const uint32_t q_split = quantize_u32(u_split);
        const uint32_t q_birth = quantize_u32(u_birth);
        const uint32_t q_merge = quantize_u32(u_merge);
        const uint32_t q_corr = quantize_u32(u_corr);
        if (q_geom > 0u) region_accum_buf.InterlockedAdd(byte_off + 8u, q_geom, prev);
        if (q_app > 0u) region_accum_buf.InterlockedAdd(byte_off + 12u, q_app, prev);
        if (q_split > 0u) region_accum_buf.InterlockedAdd(byte_off + 16u, q_split, prev);
        if (q_birth > 0u) region_accum_buf.InterlockedAdd(byte_off + 20u, q_birth, prev);
        if (q_merge > 0u) region_accum_buf.InterlockedAdd(byte_off + 24u, q_merge, prev);
        if (q_corr > 0u) region_accum_buf.InterlockedAdd(byte_off + 28u, q_corr, prev);

        // Bounded GPU structural candidate compaction
        if (split_cond && u.max_split > 0u) {
            uint32_t slot = 0u;
            candidate_buf.InterlockedAdd(0u, 1u, slot);
            if (slot < u.max_split) {
                candidate_buf.Store((2u + slot) * 4u, g_idx);
            }
        }
        if (prune_cond && u.max_prune > 0u) {
            uint32_t slot = 0u;
            candidate_buf.InterlockedAdd(4u, 1u, slot);
            if (slot < u.max_prune) {
                candidate_buf.Store((2u + u.max_split + slot) * 4u, g_idx);
            }
        }
    } else if (is_contradicted) {
        const uint32_t q_corr = quantize_u32(u_corr);
        if (q_corr > 0u) region_accum_buf.InterlockedAdd(byte_off + 28u, q_corr, prev);
    }
}
