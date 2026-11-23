#define GROUP_SIZE 256
#define GROUP_SIZE_SPARSE 64

#define MCMC_PHASE_InjectNoise 0
#define MCMC_PHASE_Probabilities 1

#define MCMC_PHASE_RelocationIndexMap 10
#define MCMC_PHASE_ComputeRelocation 11
#define MCMC_PHASE_UpdateRelocation 12

#define MCMC_PHASE_AddIndexMap 20
#define MCMC_PHASE_ComputeAdd 21
#define MCMC_PHASE_UpdateAdd 22

#define MCMC_PHASE_MaskRetractedGroup 30
#define MCMC_PHASE_ClampScales 31
#define MCMC_PHASE_UpdateGroupRows 32
#define MCMC_PHASE_CompactGather 33
#define MCMC_PHASE_CompactGatherSH 34
#define MCMC_PHASE_DecayOpacityRows 35
#define MCMC_PHASE_RayGuardObserve 36
#define MCMC_PHASE_RayGuardApply 37
#define MCMC_PHASE_RayGuardRemap 38

#ifndef MCMC_PHASE
#define MCMC_PHASE -1
#endif

#include "strategy_utils.slang"


struct InjectNoiseUniforms {
    uint32_t array_size;
    uint32_t seed;
    uint32_t image_width;
    uint32_t image_height;
    float noise_lr;
    uint32_t protect_structural_groups;
    uint32_t num_groups;
};

struct Uniforms {
    uint32_t array_size;
    uint32_t num_add;
    uint32_t seed;
    float min_opacity;
    uint32_t protect_structural_groups;
    uint32_t num_groups;
    uint32_t use_cell_filter;
    uint32_t num_cells;
};

// Per-splat MCMC protection: protect PREDICTED_UNOBSERVED (0u) and OBSERVED_UNCERTAIN (1u);
// OBSERVED_TRAINABLE (2u) and METRIC_VERIFIED (3u) are eligible for refinement; CONTRADICTED (4u) is recycled.
bool mcmc_protected(uint32_t tid, int32_t gid, uint32_t protect_all, uint32_t num_groups,
                    StructuredBuffer<uint32_t> group_state, StructuredBuffer<uint32_t> splat_state) {
    const uint32_t st = splat_base_state(splat_state[tid]);
    if (st == 0u || st == 1u)
        return true;
    if (st == 2u || st == 3u || st == 4u || st == 5u)
        return false;
    if (gid < 0)
        return false;
    if (protect_all != 0)
        return true;
    return uint32_t(gid) < num_groups && group_state[gid] == 1u;
}

struct MaskGroupUniforms {
    uint32_t array_size;
    uint32_t num_groups;
};

struct UpdateGroupRowsUniforms {
    uint32_t num_rows;
};

struct DecayOpacityRowsUniforms {
    uint32_t num_rows;
    float delta_logit;
    float min_opacity;
    uint32_t reset_moments;
    uint32_t respect_protected_states;
    uint32_t num_groups;
};

struct RayGuardObserveUniforms {
    uint32_t num_splats;
    uint32_t width;
    uint32_t height;
    uint32_t num_groups;
    float fx;
    float fy;
    float cx;
    float cy;
    float surface_protect_m;
    float dp_inv_scale;
    // Camera origin in normalized coordinates (orthogonal dp_rot preserves ray angles & scaled baseline)
    float cam_on_x;
    float cam_on_y;
    float cam_on_z;
    // Combined n2c = w2c @ inv(dp_transform) (3x4 row-major: maps xyz_norm -> camera meters)
    float n2c_00, n2c_01, n2c_02, n2c_03;
    float n2c_10, n2c_11, n2c_12, n2c_13;
    float n2c_20, n2c_21, n2c_22, n2c_23;
};

struct RayGuardApplyUniforms {
    uint32_t num_splats;
    uint32_t num_groups;
    uint32_t mode; // 0=apply decay, 1=invalidate contradictions (pose revision)
    uint32_t reset_moments;
    uint32_t respect_protected_states;
    float min_rays;
    float min_baseline_m;
    float min_1_minus_cos;
    float delta_logit;
    float min_opacity;
    float current_step;
    float cooldown_steps;
};

struct RayGuardRemapUniforms {
    uint32_t new_n;
};

struct CompactUniforms {
    uint32_t new_n;  // surviving splats
    uint32_t words;  // 32-bit words per splat (plain gather) or padded splat count (SH gather)
};

struct ClampScalesUniforms {
    uint32_t array_size;
    float max_scale;  // every axis of an ungrouped splat; the normal axis (z) of a grouped one
    float tangent_cap;  // axes x, y of a grouped splat (group_id >= 0)
};




#if MCMC_PHASE == MCMC_PHASE_InjectNoise

layout(binding=0) StructuredBuffer<int32_t> radii;  // (N,)
layout(binding=1) StructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=2) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=3) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)
layout(binding=4) StructuredBuffer<int32_t> group_ids;  // (N,)
layout(binding=5) StructuredBuffer<uint32_t> group_state;  // (G,)
layout(binding=6) StructuredBuffer<uint32_t> splat_state;  // (N,)

#define injectNoise main

float op_sigmoid(float x) {
    static const float kK = 100.0;
    static const float kX0 = 0.995;
    return 1.0 / (1.0 + exp(-kK * (x - kX0)));
}

[numthreads(GROUP_SIZE, 1, 1)]
void injectNoise(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform InjectNoiseUniforms uniforms
) {
    const uint32_t num_splats = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= num_splats)
        return;
    if (splat_base_state(splat_state[tid]) == 4u || mcmc_protected(tid, group_ids[tid], uniforms.protect_structural_groups, uniforms.num_groups, group_state, splat_state))
        return;
    
    float4 quat = normalize(rotations[tid]);
    float4 scale_opac = scales_opacs[tid];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;

    // inject noise
    if (!all(isfinite(scale)) || !isfinite(opac) || !all(isfinite(quat)))
        return;
    float3x3 cov = sqrt_cov_from_quat_scales(quat, scale);
    cov = mul(cov, transpose(cov));
    float3 noise = randn3(tid, uniforms.seed);
    noise = mul(cov, noise);
    noise *= op_sigmoid(1.0-opac) * uniforms.noise_lr;
    if (all(isfinite(noise))) {
        write_mean(read_mean(tid, means) + noise, tid, means);
    }
}

#endif  // MCMC_PHASE == MCMC_PHASE_InjectNoise



#if MCMC_PHASE == MCMC_PHASE_Probabilities

layout(binding=0) StructuredBuffer<float4> scales_opacs;  // (N,)
layout(binding=1) RWStructuredBuffer<uint32_t> sample_probs;  // (N,)
layout(binding=2) StructuredBuffer<int32_t> group_ids;  // (N,)
layout(binding=3) StructuredBuffer<uint32_t> group_state;  // (G,)
layout(binding=4) StructuredBuffer<uint32_t> splat_state;  // (N,)
layout(binding=5) StructuredBuffer<int32_t> splat_cell_ids;  // (N,) or (1,)
layout(binding=6) StructuredBuffer<int32_t> cell_filter_table;  // (C,) or (1,)
#define computeProbabilities main

[numthreads(GROUP_SIZE, 1, 1)]
void computeProbabilities(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= num_splats)
        return;

    if (splat_base_state(splat_state[tid]) == 4u || mcmc_protected(tid, group_ids[tid], uniforms.protect_structural_groups, uniforms.num_groups, group_state, splat_state)) {
        sample_probs[tid] = 0;
        return;
    }

    if (uniforms.use_cell_filter != 0) {
        const int32_t cid = splat_cell_ids[tid];
        if (cid < 0 || uint32_t(cid) >= uniforms.num_cells || cell_filter_table[cid] <= 0) {
            sample_probs[tid] = 0;
            return;
        }
    }

    float raw_opac = scales_opacs[tid].w;
    if (!isfinite(raw_opac)) {
        sample_probs[tid] = 0;
        return;
    }
    float opac = clamp(raw_opac, 1e-6, 1-1e-6);

    // low opacity -> 0; high opacity -> nonzero, prevent overflow
    const float kMinOpacity = uniforms.min_opacity;
    float max_opac = float(0x70000000) / float(num_splats);
    sample_probs[tid] = (opac <= kMinOpacity) ? 0 :
        max(uint32_t(round(opac * max_opac)), 1u);
}

#endif  // MCMC_PHASE == MCMC_PHASE_Probabilities



/*
To relocate splats:

1) Compute relocation probabilities for all Gaussians
2) Prefix sum the probability array
3) Compute relocation index map
4) Compute attributes for Gaussians relocated to
5) Update attributes for Gaussians relocated from

*/

[ForceInline]
float log_factorial(float x) {
    // least squares for x = [0, 1... 50]
    // absolute error: max 4.6e-5, L1 9.3e-6
    // notebook: https://www.desmos.com/calculator/mik6cz7h5v
    static const float kA = 0.996763591291;
    static const float kH = 2.73507778369;
    static const float kB = 0.978531458767;
    static const float kC2 = -0.0400131099548;
    static const float kC1 = -0.596969060666;
    static const float kD3 = 0.00251628691783;
    static const float kD2 = 0.0959712063178;
    static const float kD1 = 0.803207449544;
    return (kA * log(x + kH) - kB +
        (kC2 * x + kC1) / (((kD3 * x + kD2) * x + kD1) * x + 1.0)
    ) * x;
}

[ForceInline]
float binom(float n, float k) {
    return exp(log_factorial(n) - log_factorial(k) - log_factorial(n-k));
}


void relocation(inout float opacity, inout float3 scale, int n_idx) {
    n_idx = clamp(n_idx, 1, 16);  // binom(15, 7) = 6435 avoids float32 catastrophic cancellation
    opacity = clamp(opacity, 1e-5f, 0.9999f);

    float new_opacity = 1.0f - pow(max(1.0f - opacity, 1e-8f), 1.0f / float(n_idx));
    new_opacity = clamp(new_opacity, 1e-5f, 0.9999f);

    float denom_sum = 0.0f;
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float sign_k = ((k & 1) != 0) ? -1.0f : 1.0f;
            denom_sum += binom(i - 1, k) *
                (sign_k / sqrt(float(k + 1))) *
                pow(new_opacity, float(k + 1));
        }
    }
    float coeff = (denom_sum > 1e-5f && isfinite(denom_sum))
        ? clamp(opacity / denom_sum, 0.05f, 1.0f)
        : (1.0f / sqrt(float(n_idx)));

    opacity = new_opacity;
    scale = clamp(coeff * scale, float3(1e-6f), float3(10.0f));
}



#if MCMC_PHASE == MCMC_PHASE_RelocationIndexMap

layout(binding=0) StructuredBuffer<uint32_t> sample_probs;  // (N,)
layout(binding=1) StructuredBuffer<uint32_t> sample_probs_cumsum;  // (N,)
layout(binding=2) RWStructuredBuffer<uint32_t> index_map;  // (N,)
layout(binding=3) RWByteAddressBuffer n_idx_buffer;  // (N,)
layout(binding=4) StructuredBuffer<int32_t> group_ids;  // (N,)
layout(binding=5) StructuredBuffer<uint32_t> group_state;  // (G,)
layout(binding=6) StructuredBuffer<uint32_t> splat_state;  // (N,)
layout(binding=7) StructuredBuffer<float4> scales_opacs;  // (N,)
#define computeRelocationIndexMap main

[numthreads(GROUP_SIZE, 1, 1)]
void computeRelocationIndexMap(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t array_size = uniforms.array_size;
    const uint32_t tid = thread_id.x;
    if (tid >= array_size)
        return;

    if (mcmc_protected(tid, group_ids[tid], uniforms.protect_structural_groups, uniforms.num_groups, group_state, splat_state)) {
        index_map[tid] = tid;
        return;
    }

    float raw_opac = scales_opacs[tid].w;
    bool is_dead = (splat_base_state(splat_state[tid]) == 4u) || (!isfinite(raw_opac)) ||
                   (clamp(raw_opac, 1e-6, 1-1e-6) <= uniforms.min_opacity);

    // Live splats (even in cells ineligible to donate) stay in place; only dead splats relocate.
    if (!is_dead) {
        index_map[tid] = tid;
        return;
    }

    // binary search for random index
    // find the min idx such that sample_probs_cumsum[idx] >= rand
    uint32_t rand_max = sample_probs_cumsum[array_size - 1];
  #if USE_EMULATED_INT64
    uint32_t rand = mod64(hash_u32_u322(uniforms.seed, tid), rand_max) + 1u;
  #else
    uint32_t rand = uint32_t(hash_u32_u64(uniforms.seed, tid) % rand_max + 1);
  #endif
    uint32_t lo = 0;
    uint32_t hi = array_size - 1;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (sample_probs_cumsum[mid] < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    index_map[tid] = lo;

    static const uint sizeof_int = 4;
    n_idx_buffer.InterlockedAdd(lo*sizeof_int, 1);
}

#endif  // MCMC_PHASE == MCMC_PHASE_RelocationIndexMap



#if MCMC_PHASE == MCMC_PHASE_ComputeRelocation

layout(binding=0) StructuredBuffer<uint32_t> n_idx_buffer;  // (N,)

layout(binding=1) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=2) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=3) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=4) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=5) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=6) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define computeRelocation main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void computeRelocation(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t cur_idx = thread_id.x;
    if (cur_idx >= num_splats)
        return;

    // only compute on indices being relocated to
    uint n_idx = n_idx_buffer[cur_idx] + 1;
    if (n_idx == 1)
        return;

    // compute relocation
    float4 scale_opac = scales_opacs[cur_idx];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;
    relocation(opac, scale, int(n_idx));
    opac = clamp(opac, uniforms.min_opacity, 0.999999);
    scales_opacs[cur_idx] = float4(scale, opac);

    // set grad to zero
    #if 1
    _FOR_LOOP(i, 2) write_mean(float3(0.0), cur_idx, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(cur_idx, i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(cur_idx, i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[cur_idx*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[cur_idx*2+i] = float4(0.0);
    #endif

}

#endif  // MCMC_PHASE == MCMC_PHASE_ComputeRelocation



#if MCMC_PHASE == MCMC_PHASE_UpdateRelocation

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (N,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N, 16, 3) = (N, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define updateRelocation main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void updateRelocation(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t id_dst = thread_id.x;
    if (id_dst >= num_splats)
        return;

    // only update on those relocated from
    uint32_t id_src = index_map[id_dst];
    if (id_src == id_dst)
        return;

    // copy components
    write_mean(read_mean(id_src, means), id_dst, means);
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];
    rotations[id_dst] = rotations[id_src];
    scales_opacs[id_dst] = scales_opacs[id_src];

    // set grad to zero (not in gsplat)
    #if 0
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);
    #endif
}

#endif  // MCMC_PHASE == MCMC_PHASE_UpdateRelocation



/*
To add more splats:

1) Compute relocation probabilities for all Gaussians
2) Prefix sum the probability array
3) [dN] Compute add index map
4) [N] Compute attributes for Gaussians added to
5) [dN] Update attributes for Gaussians added from

*/


#if MCMC_PHASE == MCMC_PHASE_AddIndexMap

layout(binding=0) StructuredBuffer<uint32_t> sample_probs_cumsum;  // (N,)
layout(binding=1) RWStructuredBuffer<uint32_t> index_map;  // (dN,)
layout(binding=2) RWByteAddressBuffer n_idx_buffer;  // (N,)
#define computeAddIndexMap main

[numthreads(GROUP_SIZE, 1, 1)]
void computeAddIndexMap(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.num_add)
        return;

    const uint32_t num_splats = uniforms.array_size;

    // binary search for random index
    // find the min idx such that sample_probs_cumsum[idx] >= rand
    uint32_t rand_max = sample_probs_cumsum[num_splats - 1];
  #if USE_EMULATED_INT64
    uint32_t rand = mod64(hash_u32_u322(uniforms.seed+1, tid), rand_max) + 1u;
  #else
    uint32_t rand = uint32_t(hash_u32_u64(uniforms.seed+1, tid) % rand_max + 1);
  #endif
    uint32_t lo = 0;
    uint32_t hi = num_splats - 1;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (sample_probs_cumsum[mid] < rand)
            lo = mid + 1;
        else
            hi = mid;
    }
    index_map[tid] = lo;

    static const uint sizeof_int = 4;
    n_idx_buffer.InterlockedAdd(lo*sizeof_int, 1);
}

#endif  // MCMC_PHASE == MCMC_PHASE_AddIndexMap



#if MCMC_PHASE == MCMC_PHASE_ComputeAdd

layout(binding=0) StructuredBuffer<uint32_t> n_idx_buffer;  // (N,)

layout(binding=1) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

#define computeAdd main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void computeAdd(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t num_splats = uniforms.array_size;
    const uint32_t cur_idx = thread_id.x;
    if (cur_idx >= num_splats)
        return;

    // only compute on indices being relocated to
    uint n_idx = n_idx_buffer[cur_idx] + 1;
    if (n_idx == 1)
        return;

    // compute relocation
    float4 scale_opac = scales_opacs[cur_idx];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;
    relocation(opac, scale, int(n_idx));
    opac = clamp(opac, uniforms.min_opacity, 0.999999);
    scales_opacs[cur_idx] = float4(scale, opac);

    // don't change grad
}

#endif  // MCMC_PHASE == MCMC_PHASE_ComputeAdd



#if MCMC_PHASE == MCMC_PHASE_UpdateAdd

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (dN,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N+dN, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N+dN, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N+dN, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N+dN, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N+dN, 16, 3) = (N+dN, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N+dN, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N+dN, 2, 4)

#define updateAdd main

[numthreads(GROUP_SIZE, 1, 1)]
void updateAdd(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.num_add)
        return;

    const uint32_t num_splats = uniforms.array_size;
    uint32_t id_dst = tid + num_splats;
    uint32_t id_src = index_map[tid];

    // copy components
    write_mean(read_mean(id_src, means), id_dst, means);
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];
    rotations[id_dst] = rotations[id_src];
    scales_opacs[id_dst] = scales_opacs[id_src];

    // set grad to zero
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);
}

#endif  // MCMC_PHASE == MCMC_PHASE_UpdateAdd


#if MCMC_PHASE == MCMC_PHASE_MaskRetractedGroup

layout(binding=0) StructuredBuffer<int32_t> group_ids;  // (N,)
layout(binding=1) StructuredBuffer<uint32_t> group_state;  // (G,) 0=normal, 1=IMAGINED, 2=LOCKED, 3=RETRACTED
layout(binding=2) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)
layout(binding=3) StructuredBuffer<uint32_t> splat_state;  // (N,)

#define maskRetractedGroup main

[numthreads(GROUP_SIZE, 1, 1)]
void maskRetractedGroup(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform MaskGroupUniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.array_size)
        return;

    int32_t gid = group_ids[tid];
    bool grp_ret = (gid >= 0 && uint32_t(gid) < uniforms.num_groups && group_state[gid] == 3u);
    if (grp_ret || splat_base_state(splat_state[tid]) == 4u) {
        scales_opacs[tid].w = 0.0;
    }
}

#endif  // MCMC_PHASE == MCMC_PHASE_MaskRetractedGroup

#if MCMC_PHASE == MCMC_PHASE_ClampScales

layout(binding=0) StructuredBuffer<int32_t> group_ids;  // (N,)
layout(binding=1) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

#define clampScales main

[numthreads(GROUP_SIZE, 1, 1)]
void clampScales(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform ClampScalesUniforms uniforms
) {
    const uint32_t tid = thread_id.x;
    if (tid >= uniforms.array_size)
        return;

    float4 so = scales_opacs[tid];
    const float xy_cap = group_ids[tid] >= 0 ? uniforms.tangent_cap : uniforms.max_scale;
    const float3 limit = float3(xy_cap, xy_cap, uniforms.max_scale);
    const float3 clamped = min(so.xyz, limit);
    if (any(clamped != so.xyz))
        scales_opacs[tid] = float4(clamped, so.w);
}

#endif  // MCMC_PHASE == MCMC_PHASE_ClampScales


#if MCMC_PHASE == MCMC_PHASE_UpdateGroupRows

// Rewrites geometry of listed splats (update_group_sparse) and resets their geometry Adam moments;
// opacity, its moments, SH and SH moments are untouched.
layout(binding=0) StructuredBuffer<uint32_t> rows;  // (R,) splat indices
layout(binding=1) StructuredBuffer<float> values;  // (R, 10): xyz(3), scales(3), quat(4)
layout(binding=2) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)
layout(binding=5) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=7) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define updateGroupRows main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void updateGroupRows(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform UpdateGroupRowsUniforms uniforms
) {
    const uint32_t t = thread_id.x;
    if (t >= uniforms.num_rows)
        return;
    const uint32_t r = rows[t];
    const uint32_t v = 10 * t;
    write_mean(float3(values[v+0], values[v+1], values[v+2]), r, means);
    rotations[r] = float4(values[v+6], values[v+7], values[v+8], values[v+9]);
    scales_opacs[r] = float4(values[v+3], values[v+4], values[v+5], scales_opacs[r].w);
    _FOR_LOOP(i, 2) write_mean(float3(0.0), r, g_means, 2, i);
    _FOR_LOOP(i, 2) g_rotations[r*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[r*2+i] = float4(0.0, 0.0, 0.0, g_scales_opacs[r*2+i].w);
}

#endif  // MCMC_PHASE == MCMC_PHASE_UpdateGroupRows


#if MCMC_PHASE == MCMC_PHASE_CompactGather

// dst[i] = src[keep[i]] over `words` 32-bit words per splat (src is a copy of dst's old contents)
layout(binding=0) StructuredBuffer<uint32_t> keep;  // (new_n,) old index of each surviving splat
layout(binding=1) StructuredBuffer<uint32_t> src;
layout(binding=2) RWStructuredBuffer<uint32_t> dst;

#define compactGather main

[numthreads(GROUP_SIZE, 1, 1)]
void compactGather(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform CompactUniforms uniforms
) {
    const uint32_t t = thread_id.x;
    if (t >= uniforms.new_n * uniforms.words)
        return;
    const uint32_t i = t / uniforms.words;
    dst[t] = src[keep[i] * uniforms.words + (t - i * uniforms.words)];
}

#endif  // MCMC_PHASE == MCMC_PHASE_CompactGather


#if MCMC_PHASE == MCMC_PHASE_CompactGatherSH

// SH-layout (reordered blocks) gather; lanes past new_n in the last block are zeroed.
layout(binding=0) StructuredBuffer<uint32_t> keep;
layout(binding=1) StructuredBuffer<float4> src;
layout(binding=2) RWStructuredBuffer<float4> dst;

#define compactGatherSH main

[numthreads(GROUP_SIZE, 1, 1)]
void compactGatherSH(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform CompactUniforms uniforms  // words = padded splat count
) {
    const uint32_t t = thread_id.x;
    if (t >= uniforms.words * 12)
        return;
    const uint32_t i = t / 12;
    const uint32_t sid = t % 12;
    dst[shAt(i, sid)] = (i < uniforms.new_n) ? src[shAt(keep[i], sid)] : float4(0.0);
}

#endif  // MCMC_PHASE == MCMC_PHASE_CompactGatherSH


#if MCMC_PHASE == MCMC_PHASE_DecayOpacityRows

// Sparse logit-space opacity decay for contradicted splats (RayGuard):
// decays scales_opacs[r].w in logit space without full-buffer readback/upload,
// optionally protects PREDICTED_UNOBSERVED (0), OBSERVED_UNCERTAIN (1), METRIC_VERIFIED (3),
// and IMAGINED (1) / LOCKED (2) structural groups, and optionally zeroes ONLY the opacity (.w)
// Adam moments of affected rows.
layout(binding=0) StructuredBuffer<uint32_t> rows;           // (R,) deduplicated splat indices
layout(binding=1) RWStructuredBuffer<float4> scales_opacs;   // (N, 4)
layout(binding=2) RWStructuredBuffer<float4> g_scales_opacs; // (N, 2, 4)
layout(binding=3) StructuredBuffer<int32_t> group_ids;       // (N,)
layout(binding=4) StructuredBuffer<uint32_t> group_state;    // (G,) 0=normal, 1=IMAGINED, 2=LOCKED, 3=RETRACTED
layout(binding=5) StructuredBuffer<uint32_t> splat_state;    // (N,)

#define decayOpacityRows main

[numthreads(GROUP_SIZE_SPARSE, 1, 1)]
void decayOpacityRows(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform DecayOpacityRowsUniforms uniforms
) {
    const uint32_t t = thread_id.x;
    if (t >= uniforms.num_rows)
        return;
    const uint32_t r = rows[t];
    if (uniforms.respect_protected_states != 0u) {
        const uint32_t st = splat_base_state(splat_state[r]);
        if (st == 0u || st == 1u || st == 3u)
            return;
        const int32_t gid = group_ids[r];
        if (gid >= 0 && uint32_t(gid) < uniforms.num_groups) {
            const uint32_t gst = group_state[gid];
            if (gst == 1u || gst == 2u)
                return;
        }
    }
    float4 so = scales_opacs[r];
    const float p = clamp(so.w, 1e-7f, 1.0f - 1e-7f);
    const float logit = log(p / (1.0f - p)) - uniforms.delta_logit;
    const float new_p = 1.0f / (1.0f + exp(-clamp(logit, -20.0f, 20.0f)));
    so.w = max(new_p, uniforms.min_opacity);
    scales_opacs[r] = so;
    if (uniforms.reset_moments != 0u) {
        g_scales_opacs[2u * r + 0u].w = 0.0f;
        g_scales_opacs[2u * r + 1u].w = 0.0f;
    }
}

#endif  // MCMC_PHASE == MCMC_PHASE_DecayOpacityRows


#if MCMC_PHASE == MCMC_PHASE_RayGuardObserve

// GPU-resident incremental RayGuard observation (zero xyz/opacity/state readback):
// Layout of rayguard_ev (3 * N float4s):
//   [3*i + 0] = (support_hits, contra_hits, max_baseline_m, max_1_minus_cos)
//   [3*i + 1] = (first_cam_x, first_cam_y, first_cam_z, cooldown_until_step)
//   [3*i + 2] = (first_dir_x, first_dir_y, first_dir_z, 0.0)
layout(binding=0) StructuredBuffer<float> means;              // (N, 3) normalized
layout(binding=1) StructuredBuffer<float4> scales_opacs;      // (N, 4) norm_scales + linear opac
layout(binding=2) StructuredBuffer<int32_t> group_ids;        // (N,)
layout(binding=3) StructuredBuffer<uint32_t> group_state;     // (G,)
layout(binding=4) StructuredBuffer<uint32_t> splat_state;     // (N,)
layout(binding=5) StructuredBuffer<float> depth_img;          // (H * W) meters
layout(binding=6) RWStructuredBuffer<float4> rayguard_ev;     // (3 * N, 4)

#define rayGuardObserve main

bool rg_valid_depth(float d) {
    return isfinite(d) && d > 0.25f && d < 5.5f;
}

bool rg_is_depth_edge(int32_t u, int32_t v, int32_t w, int32_t h, StructuredBuffer<float> dep) {
    // 3x3 local jump + 1px dilation equivalent: check 5x5 neighborhood around (u,v) for discontinuity
    // or invalid depth hole within 1 px of the 3x3 stencil.
    float d0 = dep[v * w + u];
    if (!rg_valid_depth(d0))
        return true;
    float d_min = d0;
    float d_max = d0;
    for (int32_t dv = -1; dv <= 1; ++dv) {
        int32_t vv = clamp(v + dv, 0, h - 1);
        for (int32_t du = -1; du <= 1; ++du) {
            int32_t uu = clamp(u + du, 0, w - 1);
            float dn = dep[vv * w + uu];
            if (!rg_valid_depth(dn))
                return true;
            d_min = min(d_min, dn);
            d_max = max(d_max, dn);
        }
    }
    float thresh = max(0.08f, 0.04f * d0);
    return (d_max - d_min) > thresh;
}

[numthreads(GROUP_SIZE, 1, 1)]
void rayGuardObserve(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform RayGuardObserveUniforms U
) {
    const uint32_t tid = thread_id.x;
    if (tid >= U.num_splats)
        return;

    float4 so = scales_opacs[tid];
    if (!isfinite(so.w) || so.w < 0.08f)
        return;

    const uint32_t st = splat_base_state(splat_state[tid]);
    if (st == 0u || st == 1u || st == 3u)
        return;
    const int32_t gid = group_ids[tid];
    if (gid >= 0 && uint32_t(gid) < U.num_groups) {
        const uint32_t gst = group_state[gid];
        if (gst == 1u || gst == 2u)
            return;
    }

    float3 pn = read_mean(tid, means);
    if (!all(isfinite(pn)))
        return;

    float3 pc = float3(
        U.n2c_00 * pn.x + U.n2c_01 * pn.y + U.n2c_02 * pn.z + U.n2c_03,
        U.n2c_10 * pn.x + U.n2c_11 * pn.y + U.n2c_12 * pn.z + U.n2c_13,
        U.n2c_20 * pn.x + U.n2c_21 * pn.y + U.n2c_22 * pn.z + U.n2c_23
    );
    if (pc.z <= 0.20f || pc.z >= 5.0f)
        return;

    int32_t u = int32_t(round(pc.x * U.fx / pc.z + U.cx));
    int32_t v = int32_t(round(pc.y * U.fy / pc.z + U.cy));
    int32_t W = int32_t(U.width);
    int32_t H = int32_t(U.height);
    if (u < 0 || u >= W || v < 0 || v >= H)
        return;

    float d_meas = depth_img[v * W + u];
    if (!rg_valid_depth(d_meas))
        return;

    float4 ev0 = rayguard_ev[3u * tid + 0u];
    // 1. Center near-surface protection
    if (abs(d_meas - pc.z) <= U.surface_protect_m) {
        ev0.x = min(ev0.x + 1.0f, 30000.0f);
        rayguard_ev[3u * tid + 0u] = ev0;
        return;
    }

    // 2. Projected Gaussian footprint 5-point cross protection
    float max_s_w = max(abs(so.x), max(abs(so.y), abs(so.z))) * U.dp_inv_scale;
    float f_med = 0.5f * (U.fx + U.fy);
    int32_t r_px = clamp(int32_t(ceil(f_med * max_s_w / max(pc.z, 1e-3f))), 0, 4);
    if (r_px > 0) {
        float d1 = depth_img[v * W + clamp(u + r_px, 0, W - 1)];
        float d2 = depth_img[v * W + clamp(u - r_px, 0, W - 1)];
        float d3 = depth_img[clamp(v + r_px, 0, H - 1) * W + u];
        float d4 = depth_img[clamp(v - r_px, 0, H - 1) * W + u];
        bool fp_hit =
            (rg_valid_depth(d1) && abs(d1 - pc.z) <= U.surface_protect_m) ||
            (rg_valid_depth(d2) && abs(d2 - pc.z) <= U.surface_protect_m) ||
            (rg_valid_depth(d3) && abs(d3 - pc.z) <= U.surface_protect_m) ||
            (rg_valid_depth(d4) && abs(d4 - pc.z) <= U.surface_protect_m);
        if (fp_hit) {
            ev0.x = min(ev0.x + 1.0f, 30000.0f);
            rayguard_ev[3u * tid + 0u] = ev0;
            return;
        }
    }

    // 3. Free-space contradiction test with inline 3x3 depth-edge veto
    float sig = 0.005f + 0.0040f * d_meas;
    float margin = max(0.25f, 3.0f * sig);
    if (d_meas <= pc.z + margin)
        return;
    if (rg_is_depth_edge(u, v, W, H, depth_img))
        return;

    float3 cam_on = float3(U.cam_on_x, U.cam_on_y, U.cam_on_z);
    float3 ray = pn - cam_on;
    float rlen = max(length(ray), 1e-6f);
    float3 rdir = ray / rlen;

    float4 ev1 = rayguard_ev[3u * tid + 1u];
    float4 ev2 = rayguard_ev[3u * tid + 2u];
    if (ev0.y <= 0.5f) {
        ev1.xyz = cam_on;
        ev2.xyz = rdir;
    }
    ev0.y = min(ev0.y + 1.0f, 30000.0f);
    float b_dist = length(cam_on - ev1.xyz) * U.dp_inv_scale;
    ev0.z = max(ev0.z, b_dist);
    float one_minus_cos = 1.0f - clamp(dot(rdir, ev2.xyz), -1.0f, 1.0f);
    ev0.w = max(ev0.w, one_minus_cos);

    rayguard_ev[3u * tid + 0u] = ev0;
    rayguard_ev[3u * tid + 1u] = ev1;
    rayguard_ev[3u * tid + 2u] = ev2;
}

#endif  // MCMC_PHASE == MCMC_PHASE_RayGuardObserve


#if MCMC_PHASE == MCMC_PHASE_RayGuardApply

layout(binding=0) RWStructuredBuffer<float4> scales_opacs;   // (N, 4)
layout(binding=1) RWStructuredBuffer<float4> g_scales_opacs; // (N, 2, 4)
layout(binding=2) RWStructuredBuffer<float4> rayguard_ev;    // (3 * N, 4)
layout(binding=3) StructuredBuffer<int32_t> group_ids;       // (N,)
layout(binding=4) StructuredBuffer<uint32_t> group_state;    // (G,)
layout(binding=5) StructuredBuffer<uint32_t> splat_state;    // (N,)
layout(binding=6) RWByteAddressBuffer stats_out;             // 4 uint32s: [proposed, skipped_protected, applied, 0]

#define rayGuardApply main

[numthreads(GROUP_SIZE, 1, 1)]
void rayGuardApply(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform RayGuardApplyUniforms U
) {
    const uint32_t tid = thread_id.x;
    if (tid >= U.num_splats)
        return;

    float4 ev0 = rayguard_ev[3u * tid + 0u];
    if (U.mode == 1u) {
        // Pose-revision invalidation: clear contradiction evidence, preserve support_hits (ev0.x)
        ev0.y = 0.0f;
        ev0.z = 0.0f;
        ev0.w = 0.0f;
        rayguard_ev[3u * tid + 0u] = ev0;
        return;
    }

    float4 ev1 = rayguard_ev[3u * tid + 1u];
    float4 so = scales_opacs[tid];
    if (!isfinite(so.w) || so.w < 0.08f)
        return;
    if (ev0.y < U.min_rays || ev0.x > 0.5f || ev0.z < U.min_baseline_m || ev0.w < U.min_1_minus_cos)
        return;
    if (U.current_step < ev1.w)
        return;

    stats_out.InterlockedAdd(0u, 1u);  // proposed

    if (U.respect_protected_states != 0u) {
        const uint32_t st = splat_base_state(splat_state[tid]);
        bool prot = (st == 0u || st == 1u || st == 3u);
        if (!prot) {
            const int32_t gid = group_ids[tid];
            if (gid >= 0 && uint32_t(gid) < U.num_groups) {
                const uint32_t gst = group_state[gid];
                if (gst == 1u || gst == 2u)
                    prot = true;
            }
        }
        if (prot) {
            stats_out.InterlockedAdd(4u, 1u);  // skipped_protected
            return;
        }
    }

    const float p = clamp(so.w, 1e-7f, 1.0f - 1e-7f);
    const float logit = log(p / (1.0f - p)) - U.delta_logit;
    const float new_p = 1.0f / (1.0f + exp(-clamp(logit, -20.0f, 20.0f)));
    so.w = max(new_p, U.min_opacity);
    scales_opacs[tid] = so;
    if (U.reset_moments != 0u) {
        g_scales_opacs[2u * tid + 0u].w = 0.0f;
        g_scales_opacs[2u * tid + 1u].w = 0.0f;
    }
    ev1.w = U.current_step + U.cooldown_steps;
    rayguard_ev[3u * tid + 1u] = ev1;
    stats_out.InterlockedAdd(8u, 1u);  // applied
}

#endif  // MCMC_PHASE == MCMC_PHASE_RayGuardApply


#if MCMC_PHASE == MCMC_PHASE_RayGuardRemap

// Identity-safe GPU gather of 3*float4 RayGuard evidence across sort/prune/compact/relocate.
// Entries with remap_src[tid] == 0xFFFFFFFFu (brand-new or relocated splats) are zeroed.
layout(binding=0) StructuredBuffer<uint32_t> remap_src;  // (new_n,)
layout(binding=1) StructuredBuffer<float4> src_ev;       // (old_n * 3, 4)
layout(binding=2) RWStructuredBuffer<float4> dst_ev;     // (new_n * 3, 4)

#define rayGuardRemap main

[numthreads(GROUP_SIZE, 1, 1)]
void rayGuardRemap(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform RayGuardRemapUniforms U
) {
    const uint32_t tid = thread_id.x;
    if (tid >= U.new_n)
        return;
    const uint32_t src = remap_src[tid];
    if (src == 0xFFFFFFFFu) {
        dst_ev[3u * tid + 0u] = float4(0.0f);
        dst_ev[3u * tid + 1u] = float4(0.0f);
        dst_ev[3u * tid + 2u] = float4(0.0f);
    } else {
        dst_ev[3u * tid + 0u] = src_ev[3u * src + 0u];
        dst_ev[3u * tid + 1u] = src_ev[3u * src + 1u];
        dst_ev[3u * tid + 2u] = src_ev[3u * src + 2u];
    }
}

#endif  // MCMC_PHASE == MCMC_PHASE_RayGuardRemap
