// Sparse-tile selection for the backward pass (see docs in gs_trainer.cpp, executeSelectSparseTiles).
//
//   ENTRY 1 (sparse_tile_residual): one 16x16 workgroup per tile, writes the tile's mean |render - target|
//            (rgb, alpha-masked like the SSIM shader) as the bits of a non-negative float, so uint order == float order.
//   ENTRY 2 (sparse_tile_select):   one workgroup of 1024 threads. Picks the `keep` tiles with the largest
//            residual (exact k-th largest by a 4-pass 8-bit radix select, ties kept, zero residual never kept),
//            dilates by `halo` tiles, compacts the survivors in tile order into `active_tiles` and writes the
//            indirect dispatch arguments. `full != 0` keeps every tile (sentinel step). When the step was
//            skipped by the step gate (its raster-backward arguments are zero) the dispatch is empty.

#include "config.slang"

#define _ENTRY_residual 1
#define _ENTRY_select 2

#ifndef ENTRY
#define ENTRY 0
#endif

#if ENTRY == _ENTRY_residual

layout(binding=0) StructuredBuffer<float4> pixel_state;
layout(binding=1) StructuredBuffer<uint32_t> ref_image;
layout(binding=2) RWStructuredBuffer<uint32_t> tile_residual;

struct ResidualUniforms {
    uint image_width;
    uint image_height;
    uint grid_width;
    uint _pad;
}

groupshared float partial[TILE_SIZE / SUBGROUP_SIZE];

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform ResidualUniforms u
) {
    uint rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;
    float r = 0.0f;
    if (dispatchThreadID.x < u.image_width && dispatchThreadID.y < u.image_height) {
        uint pix = dispatchThreadID.y * u.image_width + dispatchThreadID.x;
        uint32_t ref = ref_image[pix];
        float alpha = float(ref >> 24) / 255.0f;
        float3 p2 = float3(ref & 255, (ref >> 8) & 255, (ref >> 16) & 255) / 255.0f;
        float3 p1 = max(pixel_state[pix].rgb, float3(0.0f));
        float3 d = abs(p1 - p2) * alpha;
        r = d.x + d.y + d.z;
    }
    float s = WaveActiveSum(r);
    if (WaveIsFirstLane())
        partial[rank / SUBGROUP_SIZE] = s;
    GroupMemoryBarrierWithGroupSync();
    if (rank == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TILE_SIZE / SUBGROUP_SIZE; i++)
            total += partial[i];
        uint w = min(uint(TILE_WIDTH), u.image_width - groupID.x * TILE_WIDTH);
        uint h = min(uint(TILE_HEIGHT), u.image_height - groupID.y * TILE_HEIGHT);
        tile_residual[groupID.y * u.grid_width + groupID.x] = asuint(total / (3.0f * float(w * h)));
    }
}

#elif ENTRY == _ENTRY_select

static const uint NT = 1024;

layout(binding=0) StructuredBuffer<uint32_t> tile_residual;
layout(binding=1) RWStructuredBuffer<uint32_t> tile_mask;
layout(binding=2) RWStructuredBuffer<uint32_t> active_tiles;
layout(binding=3) RWStructuredBuffer<uint32_t> out_args;         // x, y, z, pad
layout(binding=4) StructuredBuffer<uint32_t> gate_args;
layout(binding=5) RWStructuredBuffer<uint32_t> counters;         // [0] dispatched tiles (accumulated on the GPU)

struct SelectUniforms {
    uint num_tiles;
    uint grid_width;
    uint keep;           // number of tiles to keep before dilation
    uint full;           // 1: keep every tile
    uint halo;           // dilation radius in tiles
    uint args_offset;    // word offset of this step's arguments in out_args
    uint gate_ok_index;  // word index in gate_args of this step's raster-backward x count, or 0xffffffff
    uint _pad;
}

groupshared uint wave_sums[NT / SUBGROUP_SIZE];
groupshared uint digit_hist[256];
groupshared uint picked[2];

uint groupSum(uint v, uint tid) {
    uint s = WaveActiveSum(v);
    if (WaveIsFirstLane())
        wave_sums[tid / SUBGROUP_SIZE] = s;
    GroupMemoryBarrierWithGroupSync();
    uint total = 0;
    for (uint i = 0; i < NT / SUBGROUP_SIZE; i++)
        total += wave_sums[i];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(NT, 1, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uniform SelectUniforms u
) {
    const uint tid = groupThreadID.x;
    const uint n = u.num_tiles;
    const bool ok = u.gate_ok_index == 0xffffffffu || gate_args[u.gate_ok_index] != 0;

    // k-th largest residual (k = keep) by four 8-bit digit passes (MSD radix select)
    uint threshold = 0;
    if (u.full == 0) {
        uint mask = 0;
        uint k = u.keep;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (tid < 256)
                digit_hist[tid] = 0;
            GroupMemoryBarrierWithGroupSync();
            for (uint t = tid; t < n; t += NT) {
                uint r = tile_residual[t];
                if ((r & mask) == threshold) {
                    uint old;
                    InterlockedAdd(digit_hist[(r >> shift) & 255], 1u, old);
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (tid == 0) {
                uint acc = 0;
                uint digit = 0;
                for (int d = 255; d >= 0; d--) {
                    if (acc + digit_hist[d] >= k) {
                        digit = uint(d);
                        break;
                    }
                    acc += digit_hist[d];
                }
                picked[0] = digit;
                picked[1] = k - acc;
            }
            GroupMemoryBarrierWithGroupSync();
            threshold |= picked[0] << shift;
            mask |= 255u << shift;
            k = picked[1];
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint t = tid; t < n; t += NT) {
        uint r = tile_residual[t];
        tile_mask[t] = (u.full != 0 || (r >= threshold && r > 0)) ? 1u : 0u;
    }
    AllMemoryBarrierWithGroupSync();

    // contiguous chunk per thread so the compacted list stays in tile order
    const uint chunk = (n + NT - 1) / NT;
    const uint lo = min(tid * chunk, n);
    const uint hi = min(lo + chunk, n);
    const int gw = int(u.grid_width);
    const int gh = int((n + u.grid_width - 1) / u.grid_width);
    const int halo = int(u.halo);

    uint count = 0;
    for (uint pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            // exclusive prefix over the threads: wave prefix + preceding wave totals
            uint wave_total = WaveActiveSum(count);
            uint within = WavePrefixSum(count);
            if (WaveIsFirstLane())
                wave_sums[tid / SUBGROUP_SIZE] = wave_total;
            GroupMemoryBarrierWithGroupSync();
            uint base = within;
            for (uint i = 0; i < tid / SUBGROUP_SIZE; i++)
                base += wave_sums[i];
            GroupMemoryBarrierWithGroupSync();  // wave_sums is reused by the final groupSum
            count = base;
        }
        uint cursor = count;
        uint found = 0;
        for (uint t = lo; t < hi; t++) {
            int tx = int(t % u.grid_width), ty = int(t / u.grid_width);
            bool active = false;
            for (int dy = -halo; dy <= halo && !active; dy++)
                for (int dx = -halo; dx <= halo; dx++) {
                    int nx = tx + dx, ny = ty + dy;
                    if (nx >= 0 && nx < gw && ny >= 0 && ny < gh && tile_mask[uint(ny * gw + nx)] != 0) {
                        active = true;
                        break;
                    }
                }
            if (active) {
                if (pass == 1)
                    active_tiles[cursor + found] = t;
                found++;
            }
        }
        count = found;  // pass 0: chunk count to prefix-sum; pass 1: kept for the group total
    }
    uint total = groupSum(count, tid);
    if (tid == 0) {
        uint dispatched = ok ? total : 0;
        out_args[u.args_offset] = dispatched;
        out_args[u.args_offset + 1] = 1;
        out_args[u.args_offset + 2] = 1;
        out_args[u.args_offset + 3] = 0;
        counters[0] += dispatched;
    }
}

#endif
