// L1 + SSIM pixel-loss gradient in two dispatches.
//
//   STAGE 1 (ssim_stats): per colour channel, Gaussian statistics of (render x, ground truth y) -> the three
//       partial derivatives of the SSIM map, (dM/dmu1, dM/dsigma1^2, dM/dsigma12), written as planar fp32
//       maps for the interior [5, size - 5) of the SSIM grid (9 floats per grid pixel).
//   STAGE 2 (ssim_grad): Gaussian blur of the three maps (the transposed filter), the chain rule
//       dSSIM/dx = blur(dmu1) + 2 x blur(dsigma1^2) + y blur(dsigma12), the L1 gradient, the ReLU of the
//       render, and the write of dL/dpixel. With useSsim = 0 only the L1 gradient is written (stage 1 is
//       then not dispatched).
//
// Both stages use register-blocked separable FIRs on a 16x16 tile with LDS tiles of at most 18 KB; the map is
// the only global-memory intermediate (the former pair wrote 12 floats per pixel and read it with float4s).
// The SSIM value itself is not needed by training and is not computed.
//
// DOWNSAMPLE = 2 (the *_ds2 variants) evaluates SSIM on the 2x2 box average of the (alpha-premultiplied)
// images; every full-resolution pixel of a 2x2 block then receives a quarter of the gradient of its grid pixel
// (the weight is folded into gradWeightSSIM by the host).

#ifndef STAGE
#define STAGE 1
#endif
#ifndef DOWNSAMPLE
#define DOWNSAMPLE 1
#endif

static const float GAUSS_COEFFS[6] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f
};
static const float W11[11] = {
    GAUSS_COEFFS[0], GAUSS_COEFFS[1], GAUSS_COEFFS[2], GAUSS_COEFFS[3], GAUSS_COEFFS[4], GAUSS_COEFFS[5],
    GAUSS_COEFFS[4], GAUSS_COEFFS[3], GAUSS_COEFFS[2], GAUSS_COEFFS[1], GAUSS_COEFFS[0]
};

static const uint BLOCK = 16;
static const uint HALO = 5;
static const uint TILE = BLOCK + 2 * HALO;  // 26: input tile of stage 1, map tile of stage 2
static const uint NUM_THREADS = BLOCK * BLOCK;
static const uint ds = DOWNSAMPLE;  // SSIM grid = 1/ds of the image
// Odd row strides keep the strided LDS accesses of a wave conflict-free.
static const uint IMG_STRIDE = TILE + 1;
static const uint STAT_STRIDE = BLOCK + 1;

static const float C1 = 0.01 * 0.01;
static const float C2 = 0.03 * 0.03;

struct Uniforms {
    uint imageWidth;
    uint imageHeight;
    float gradWeightL1;
    float gradWeightSSIM;  // includes 1 / downsample^2
    uint useSsim;
    uint useTileList;  // stage 2, DOWNSAMPLE 1 only: one workgroup per listed 16x16 tile instead of per image tile
}

// Alpha-premultiplied (render, ground truth) of one full-resolution pixel; zero outside the image.
float2x3 fetchPixel(StructuredBuffer<float4> img1, StructuredBuffer<uint32_t> img2, int x, int y, uint w, uint h) {
    if (x < 0 || y < 0 || x >= int(w) || y >= int(h))
        return float2x3(0.0f);
    uint idx = uint(y) * w + uint(x);
    float3 v1 = max(img1[idx].rgb, float3(0.0f));
    uint32_t val = img2[idx];
    float alpha = float(val >> 24) / 255.0f;
    float3 v2 = float3(val & 255, (val >> 8) & 255, (val >> 16) & 255) / 255.0f;
    return float2x3(v1 * alpha, v2 * alpha);
}

#if STAGE == 1

layout(binding=0) StructuredBuffer<float4> img1Buffer;    // render (pixel_state), RGB + alpha
layout(binding=1) StructuredBuffer<uint32_t> img2Buffer;  // ground truth, RGBA8
layout(binding=2) RWStructuredBuffer<float> mapBuffer;    // [channel][grid pixel][3 maps]

groupshared float2 sImg[TILE * IMG_STRIDE + 2];         // (x, y) of the channel
groupshared float4 sStat[TILE * STAT_STRIDE];           // horizontally filtered (x, y, x^2 + y^2, x y), [TILE][BLOCK]

[numthreads(BLOCK, BLOCK, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint w = uniforms.imageWidth;
    const uint h = uniforms.imageHeight;
    const int gridW = int(w / ds);
    const int gridH = int(h / ds);
    const int originX = int(groupID.x * BLOCK);
    const int originY = int(groupID.y * BLOCK);
    const uint tid = groupThreadID.y * BLOCK + groupThreadID.x;

    // Input tile of the grid (alpha-premultiplied, ds x ds box average), held across the channel loop.
    float3 pre1[3];
    float3 pre2[3];
    [ForceUnroll]
    for (uint s = 0; s < 3; ++s) {
        uint i = s * NUM_THREADS + tid;
        float3 a1 = float3(0.0f), a2 = float3(0.0f);
        if (i < TILE * TILE) {
            int gx = originX + int(i % TILE) - int(HALO);
            int gy = originY + int(i / TILE) - int(HALO);
            if (gx >= 0 && gy >= 0 && gx < gridW && gy < gridH) {
                for (uint dy = 0; dy < ds; ++dy)
                    for (uint dx = 0; dx < ds; ++dx) {
                        float2x3 p = fetchPixel(img1Buffer, img2Buffer, gx * int(ds) + int(dx), gy * int(ds) + int(dy), w, h);
                        a1 += p[0];
                        a2 += p[1];
                    }
                a1 *= 1.0f / float(ds * ds);
                a2 *= 1.0f / float(ds * ds);
            }
        }
        pre1[s] = a1;
        pre2[s] = a2;
    }

    [ForceUnroll]
    for (uint ci = 0; ci < 3; ++ci) {
        [ForceUnroll]
        for (uint s = 0; s < 3; ++s) {
            uint i = s * NUM_THREADS + tid;
            if (i < TILE * TILE)
                sImg[(i / TILE) * IMG_STRIDE + i % TILE] = float2(pre1[s][ci], pre2[s][ci]);
        }
        GroupMemoryBarrierWithGroupSync();

        // Horizontal pass: 4 blocks of 4 columns x 26 rows, 14 input columns per block.
        if (tid < 4 * TILE) {
            uint row = tid % TILE;
            uint col0 = (tid / TILE) * 4;
            float4 q[14];
            [ForceUnroll]
            for (uint k = 0; k < 14; ++k) {
                float2 p = sImg[row * IMG_STRIDE + col0 + k];
                q[k] = float4(p.x, p.y, p.x * p.x + p.y * p.y, p.x * p.y);
            }
            [ForceUnroll]
            for (uint j = 0; j < 4; ++j) {
                float4 acc = float4(0.0f);
                [ForceUnroll]
                for (uint k = 0; k < 11; ++k)
                    acc += q[j + k] * W11[k];
                sStat[row * STAT_STRIDE + col0 + j] = acc;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Vertical pass and SSIM partial derivatives: 8 blocks of 2 rows x 16 columns.
        if (tid < 8 * BLOCK) {
            uint col = tid % BLOCK;
            uint row0 = (tid / BLOCK) * 2;
            float4 q[12];
            [ForceUnroll]
            for (uint k = 0; k < 12; ++k)
                q[k] = sStat[(row0 + k) * STAT_STRIDE + col];
            [ForceUnroll]
            for (uint j = 0; j < 2; ++j) {
                int gx = originX + int(col);
                int gy = originY + int(row0 + j);
                if (gx >= int(HALO) && gx < gridW - int(HALO) && gy >= int(HALO) && gy < gridH - int(HALO)) {
                    float4 acc = float4(0.0f);
                    [ForceUnroll]
                    for (uint k = 0; k < 11; ++k)
                        acc += q[j + k] * W11[k];
                    float mu1 = acc.x;
                    float mu2 = acc.y;
                    float mu1Sq = mu1 * mu1;
                    float mu2Sq = mu2 * mu2;
                    float A = mu1Sq + mu2Sq + C1;
                    float B = acc.z - mu1Sq - mu2Sq + C2;  // sigma1^2 + sigma2^2 + C2
                    float C_ = 2.0f * mu1 * mu2 + C1;
                    float D_ = 2.0f * (acc.w - mu1 * mu2) + C2;
                    float invA = 1.0f / A;
                    float invB = 1.0f / B;
                    float invAB = invA * invB;
                    uint base = 3u * (ci * uint(gridW * gridH) + uint(gy * gridW + gx));
                    mapBuffer[base] = 2.0f * invAB * (mu2 * (D_ - C_) + C_ * D_ * mu1 * (invB - invA));
                    mapBuffer[base + 1u] = -C_ * D_ * invAB * invB;
                    mapBuffer[base + 2u] = 2.0f * C_ * invAB;
                }
            }
        }
    }
}

#else  // STAGE == 2

layout(binding=0) StructuredBuffer<float4> img1Buffer;
layout(binding=1) StructuredBuffer<uint32_t> img2Buffer;
layout(binding=2) StructuredBuffer<float> mapBuffer;
layout(binding=3) RWStructuredBuffer<float4> dLdimg1Buffer;
layout(binding=4) StructuredBuffer<uint32_t> activeTiles;  // compacted tile list when useTileList != 0

groupshared float4 sMap[TILE * IMG_STRIDE];        // (dmu1, dsigma1^2, dsigma12), [TILE][TILE]
groupshared float4 sRow[TILE * STAT_STRIDE];       // horizontally filtered maps, [TILE][BLOCK]

[numthreads(BLOCK, BLOCK, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint w = uniforms.imageWidth;
    const uint h = uniforms.imageHeight;
    const int gridW = int(w / ds);
    const int gridH = int(h / ds);
    const uint gridPixels = uint(gridW * gridH);
    uint2 tile = groupID.xy;
    if (uniforms.useTileList != 0) {
        uint tileIdx = activeTiles[groupID.x];
        uint tilesPerRow = (w + BLOCK - 1) / BLOCK;
        tile = uint2(tileIdx % tilesPerRow, tileIdx / tilesPerRow);
    }
    const int originX = int(tile.x * BLOCK);
    const int originY = int(tile.y * BLOCK);
    const uint tx = groupThreadID.x;
    const uint ty = groupThreadID.y;
    const uint tid = ty * BLOCK + tx;
    const bool useSsim = uniforms.useSsim != 0;

    // The full-resolution pixels of this thread's grid pixel (ds x ds of them) and their box average.
    float3 own1[ds * ds];
    float3 own2[ds * ds];
    float3 avg1 = float3(0.0f);
    float3 avg2 = float3(0.0f);
    [ForceUnroll]
    for (uint s = 0; s < ds * ds; ++s) {
        float2x3 p = fetchPixel(img1Buffer, img2Buffer, (originX + int(tx)) * int(ds) + int(s % ds),
                                (originY + int(ty)) * int(ds) + int(s / ds), w, h);
        own1[s] = p[0];
        own2[s] = p[1];
        avg1 += p[0];
        avg2 += p[1];
    }
    avg1 *= 1.0f / float(ds * ds);
    avg2 *= 1.0f / float(ds * ds);

    float3 gradSsim = float3(0.0f);

    if (useSsim) {
        [ForceUnroll]
        for (uint ci = 0; ci < 3; ++ci) {
            // Map tile with a 5-pixel halo; zero outside the interior [5, size - 5).
            [ForceUnroll]
            for (uint s = 0; s < 3; ++s) {
                uint i = s * NUM_THREADS + tid;
                if (i < TILE * TILE) {
                    int gx = originX + int(i % TILE) - int(HALO);
                    int gy = originY + int(i / TILE) - int(HALO);
                    float4 d = float4(0.0f);
                    if (gx >= int(HALO) && gx < gridW - int(HALO) && gy >= int(HALO) && gy < gridH - int(HALO)) {
                        uint base = 3u * (ci * gridPixels + uint(gy * gridW + gx));
                        d = float4(mapBuffer[base], mapBuffer[base + 1u], mapBuffer[base + 2u], 0.0f);
                    }
                    sMap[(i / TILE) * IMG_STRIDE + i % TILE] = d;
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // Horizontal pass: 8 blocks of 2 columns x 26 rows.
            if (tid < (BLOCK / 2) * TILE) {
                uint row = tid % TILE;
                uint col0 = (tid / TILE) * 2;
                float3 q[12];
                [ForceUnroll]
                for (uint k = 0; k < 12; ++k)
                    q[k] = sMap[row * IMG_STRIDE + col0 + k].xyz;
                [ForceUnroll]
                for (uint j = 0; j < 2; ++j) {
                    float3 acc = float3(0.0f);
                    [ForceUnroll]
                    for (uint k = 0; k < 11; ++k)
                        acc += q[j + k] * W11[k];
                    sRow[row * STAT_STRIDE + col0 + j] = float4(acc, 0.0f);
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // Vertical pass: dSSIM / d(grid pixel), chain rule through mu, sigma1^2 and sigma12.
            float3 acc = float3(0.0f);
            [ForceUnroll]
            for (uint k = 0; k < 11; ++k)
                acc += sRow[(ty + k) * STAT_STRIDE + tx].xyz * W11[k];
            gradSsim[ci] = acc.x + 2.0f * avg1[ci] * acc.y + avg2[ci] * acc.z;
        }
    }

    // L1 + SSIM gradient of the full-resolution pixels of this grid pixel.
    const bool inGrid = int(tx) + originX < gridW && int(ty) + originY < gridH;
    [ForceUnroll]
    for (uint s = 0; s < ds * ds; ++s) {
        int x = (originX + int(tx)) * int(ds) + int(s % ds);
        int y = (originY + int(ty)) * int(ds) + int(s / ds);
        if (x < int(w) && y < int(h)) {
            float3 g = sign(own1[s] - own2[s]) * uniforms.gradWeightL1;
            if (useSsim && inGrid)
                g += gradSsim * uniforms.gradWeightSSIM;
            g *= float3(own1[s] > 0.0f);
            dLdimg1Buffer[y * int(w) + x] = float4(g, 0.0f);
        }
    }
}

#endif
