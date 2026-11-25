#pragma once

#include "gs_pipeline.h"

#include "perf_timer.h"
#include "scheduler.h"


PACK_STRUCT(struct VulkanGSRendererUniforms {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t num_splats;
    uint32_t active_sh;
    uint32_t step;
    uint32_t camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
    uint32_t use_tile_list;  // 1: backward kernels run over the compacted active-tile list
    uint32_t freeze_flags;  // bit 0: IMAGINED groups present, bit 1: protected LOCKED groups present (backward skips their atomics)
    uint32_t num_groups;
    uint32_t ago_telemetry_enabled;
    float depth_loss_scale;    // lambda_depth / (H * W); 0.0 disables depth loss
    float depth_huber_delta;   // Huber delta in meters
    float depth_inv_scale;     // 1.0 / world_to_normalized_scale (meters per normalized unit)
    uint32_t render_exclude_flags;  // S104: bit 0: GROUP_STATE_EXCLUDED_RENDER, bit 1: splat FROZEN_INVISIBLE, bit 2: GPU workload counters enabled
});


class VulkanGSRenderer : public VulkanGSPipeline {
public:
    VulkanGSRenderer();
    ~VulkanGSRenderer();

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id);
    void cleanup();

    void executeProjectionForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, size_t alloc_reserve=0);
    void executeGenerateKeys(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeComputeTileRanges(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeForwardDepth(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeBackward(
        const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers,
        const _VulkanBuffer* train_depth_weight_buf = nullptr
    );
    // true when the backward pass of the scheduler pick `impl` is dispatched per pixel (2D) rather than per tile
    static bool backwardImplIsPerPixel(int impl) { return impl == 0; }

    // read_count: legacy path, blocks on a fence to learn the intersection count on the host
    void executeCalculateIndexBufferOffset(VulkanGSPipelineBuffers& buffers, bool read_count=true);
    void executeSort(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int num_bits);

    // ---- batched training steps without host round trips (see step_gate.slang) ----
    // Intersection-sized buffers keep a capacity; the gate checks the GPU-side count against it.
    size_t isectCapacity(const VulkanGSPipelineBuffers& buffers) const;
    // test hook: reallocate the intersection buffers to exactly this capacity (also shrinks)
    void setIsectCapacityExact(VulkanGSPipelineBuffers& buffers, size_t capacity);
    void ensureIsectCapacity(VulkanGSPipelineBuffers& buffers, size_t want);
    // Call inside one DeviceGuard around the whole batch: begin, then per step beginStep + tiles, then end.
    void gateBeginBatch();
    void gateBeginStep(uint32_t step_in_batch);
    // cumsum + gate + generate keys + sort + tile ranges, all driven by the GPU-side count
    void executeTilesDeferred(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers,
                              const GatePlan& plan);
    void writeIsectCount(uint32_t n);  // legacy path: host-known count for the sort / tile-range kernels
    void gateEndBatch();  // copies the status to the host mirror; the batch fence makes it readable
    struct GateResult {
        uint32_t steps_done = 0;
        bool overflow = false;
        uint32_t needed = 0;  // intersection count of the first overflowing step
        uint32_t isects[kGateMaxSteps] = {};
    };
    GateResult gateResult() const;
    int sampleBackwardImpl();  // scheduler pick (RasterBackwardImpl as int) for the next backward pass
    void setBackwardPinAfter(size_t updates);
    // Force one backward variant (RasterBackwardImpl value) instead of the scheduler; -1 restores scheduling.
    void forceBackwardImpl(int impl);
    std::vector<int> backwardImplCandidates();

    // ---- sparse tile backward ----
    // Per-tile residual -> quantile threshold -> halo dilation -> compacted tile list + indirect arguments,
    // all on the GPU (sparse_tiles.slang). The raster-backward and SSIM-backward kernels then run over the
    // listed tiles only. Must be called after the forward render and before both backward kernels.
    void executeSelectSparseTiles(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers,
                                  float quantile, int halo);
    // Grow the tile-sized scratch buffers; call outside a command batch.
    void ensureSparseTileBuffers(size_t num_tiles);
    // Tiles dispatched by sparse backward passes so far (device counter; reads back, never call mid-step).
    uint64_t sparseTilesDispatched();

    // S104 Phase 12: Device-side GPU workload counters (zeroed & incremented only when stage profiling is enabled).
    // Slots: [0]=gpu_projected_gaussians, [1]=gpu_tile_intersections,
    //        [2]=gpu_backward_tiles_executed, [3]=gpu_backward_tiles_skipped.
    static constexpr size_t kAgoGpuCounterCount = 8u;
    void resetAgoGpuCounters();
    bool readAgoGpuCounters(uint32_t out_counters[4]);

private:
    enum class RasterBackwardImpl {
        PerPixel = 0,
        PerSplat = 1,
        Tensor_0_8_0 = 2,
        Tensor_0_8_8 = 3,
        Tensor_1_16_0 = 4,
        size = 5,
        Default = 0x3fffffff,
    };
    std::vector<RasterBackwardImpl> rasterizeBackwardAlternatives;
    ThompsonSamplingScheduler rasterizeBackwardScheduler;
    size_t backward_pin_after = 0;
    int forced_backward_impl = -1;
    RasterBackwardImpl sampleBackwardAlternative();
    int gate_backward_impl = 0;  // pick made at plan time for the step being recorded
    void initRasterizationBackwardScheduler();

protected:
    virtual DeviceRequirement getDeviceRequirement();

    void executeCumsum(
        VulkanGSPipelineBuffers &buffers,
        Buffer<int32_t> &input_buffer,
        Buffer<int32_t> &output_buffer
    );
    int32_t executeSum(
        VulkanGSPipelineBuffers &buffers,
        Buffer<int32_t> &input_buffer
    );
    void executeWhere(
        VulkanGSPipelineBuffers& buffers,
        Buffer<int32_t> &input_buffer,
        Buffer<int32_t> &output_buffer
    );

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(15);
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(7);
    _ComputePipeline pipeline_step_gate = _ComputePipeline(4);
    _ComputePipeline pipeline_sparse_residual = _ComputePipeline(3);
    _ComputePipeline pipeline_sparse_select = _ComputePipeline(6);
    _VulkanBuffer sparse_residual, sparse_mask, sparse_active_tiles;  // (tiles,) uint32
    _VulkanBuffer sparse_args;      // one VkDispatchIndirectCommand
    _VulkanBuffer sparse_counters;  // [0]: tiles dispatched by sparse backward passes (wraps at 2^32)
    _VulkanBuffer ago_gpu_counters_buf_;  // S104: 8-uint device atomic workload counters
    size_t sparse_capacity = 0;
    uint32_t sparse_counter_last = 0;
    uint64_t sparse_counter_total = 0;
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)
    };
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(7);
    _ComputePipelinePair pipeline_rasterize_forward_depth = _ComputePipelinePair(9);
    _ComputePipelinePair pipeline_rasterize_backward[(size_t)RasterBackwardImpl::size] = {
        _ComputePipelinePair(22),  // per pixel
        _ComputePipelinePair(22),  // per splat
        _ComputePipelinePair(22),  // tensor
        _ComputePipelinePair(22),  // tensor
        _ComputePipelinePair(22),  // tensor
    };
    // Per-step AGO rasterize-backward scratch buffers (zeroed & accumulated only when ago_telemetry_enabled != 0)
    Buffer<float> ago_contrib_step_buf_;
    Buffer<float> ago_abs_grad_step_buf_;
    struct _CumsumComputePipeline {
        _ComputePipeline single_pass = _ComputePipeline(2);
        _ComputePipeline block_scan = _ComputePipeline(3);
        _ComputePipeline scan_block_sums = _ComputePipeline(3);
        _ComputePipeline add_block_offsets = _ComputePipeline(3);
    } pipeline_cumsum;
    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(4);
        _ComputePipeline spine = _ComputePipeline(3);
        _ComputePipeline downsweep = _ComputePipeline(7);
    } pipeline_sorting_1, pipeline_sorting_2;
    _ComputePipeline pipeline_sum = _ComputePipeline(2);
    _ComputePipeline pipeline_where = _ComputePipeline(3);
    _ComputePipeline pipeline_null = _ComputePipeline(0);

};
