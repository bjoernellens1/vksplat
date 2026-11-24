#include "gs_renderer.h"
#include <atomic>

#include <fstream>
#include <memory>
#include <csignal>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif


VulkanGSRenderer::VulkanGSRenderer()
    : VulkanGSPipeline() {
}

VulkanGSRenderer::~VulkanGSRenderer() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}

void VulkanGSRenderer::cleanup() {
    for (auto* b : {&sparse_residual, &sparse_mask, &sparse_active_tiles, &sparse_args, &sparse_counters})
        destroyBuffer(*b);
    sparse_capacity = 0;
    destroyBuffer(ago_contrib_step_buf_.deviceBuffer);
    ago_contrib_step_buf_.clear();
    ago_contrib_step_buf_.shrink_to_fit();
    destroyBuffer(ago_abs_grad_step_buf_.deviceBuffer);
    ago_abs_grad_step_buf_.clear();
    ago_abs_grad_step_buf_.shrink_to_fit();
    VulkanGSPipeline::cleanup();
}


VulkanGSPipeline::DeviceRequirement VulkanGSRenderer::getDeviceRequirement() {
    const uint32_t minSharedMemory = ([=]() -> uint32_t {
        static constexpr uint32_t TILE_SIZE = TILE_HEIGHT*TILE_WIDTH;
        uint32_t raster_backward_0 = TILE_SIZE*(9+1+1)*4;  // per pixel
        uint32_t raster_backward_1 = TILE_SIZE*(4+4+1)*4;  // per splat
        // uint32_t raster_backward_2 = TILE_SIZE*(1+2+4+4+6+2*8+4)*4;  // tensor 0 8 0
        // uint32_t raster_backward_3 = TILE_SIZE*(1+2+4+4+6+2*8+4+6)*4;  // tensor 0 8 8
        // uint32_t raster_backward_4 = TILE_SIZE*(1+2+4+4+2*16+4)*4;  // tensor 1 16 0
        uint32_t ssim = (26*27*4 + 26*17*4)*4;  // tiles of the SSIM stages (stats: 26*27*2 + 2 + 26*17*4 floats)
        uint32_t radix_sort_rts = 4096*sizeof(sortingKey_t)+256*4;
        return std::max({
            raster_backward_0, raster_backward_1,
            // raster_backward_2, raster_backward_3, raster_backward_4,
            ssim, radix_sort_rts
        });
    })();
    return DeviceRequirement{
        { 12*16777216/256, 4096/std::min(TILE_HEIGHT,TILE_WIDTH), 1 },
        { 1024, 1*std::max(TILE_HEIGHT,TILE_WIDTH), 1 },
        minSharedMemory
    };
}


void VulkanGSRenderer::initialize(const std::map<std::string, std::string> &spirv_paths, int device_id) {

    VulkanGSPipeline::initialize(device_id);
    
    createComputePipeline(pipeline_projection_forward, spirv_paths.at("projection_forward"));
    createComputePipeline(pipeline_generate_keys, spirv_paths.at("generate_keys"));
    createComputePipeline(pipeline_step_gate, spirv_paths.at("step_gate"));
    createGateBuffers();
    createComputePipeline(pipeline_sparse_residual, spirv_paths.at("sparse_tile_residual"));
    createComputePipeline(pipeline_sparse_select, spirv_paths.at("sparse_tile_select"));
    createBuffer(4 * sizeof(uint32_t), sparse_args);
    createBuffer(sizeof(uint32_t), sparse_counters);
    {
        DEVICE_GUARD;
        vkCmdFillBuffer(command_buffer, sparse_counters.buffer, 0, sizeof(uint32_t), 0);
    }
    ensureSparseTileBuffers(1);
    for (int i = 0; i < 2; ++i) {
        createComputePipeline(pipeline_compute_tile_ranges[i], spirv_paths.at("compute_tile_ranges"));
        createComputePipeline(pipeline_rasterize_forward[i], spirv_paths.at("rasterize_forward"));
        createComputePipeline(pipeline_rasterize_forward_depth[i], spirv_paths.at("rasterize_forward_depth"));
        createComputePipeline(pipeline_rasterize_backward[0][i], spirv_paths.at("rasterize_backward_0"));
        createComputePipeline(pipeline_rasterize_backward[1][i], spirv_paths.at("rasterize_backward_1"));
        createComputePipeline(pipeline_rasterize_backward[2][i], spirv_paths.at("rasterize_backward_2"), 35840);
        createComputePipeline(pipeline_rasterize_backward[3][i], spirv_paths.at("rasterize_backward_3"), 41984);
        createComputePipeline(pipeline_rasterize_backward[4][i], spirv_paths.at("rasterize_backward_4"), 35840);
    }
    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
    createComputePipeline(pipeline_sorting_1.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_1.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_1.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_2.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_2.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_2.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sum, spirv_paths.at("sum"));
    createComputePipeline(pipeline_where, spirv_paths.at("where"));

}


void VulkanGSRenderer::executeProjectionForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t alloc_reserve
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::ProjectionForward> timer(this);

    size_t num_splats = buffers.num_splats;

    bufferMemoryBarrier({
        { buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.sh_coeffs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.scales_opacs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    size_t alloc_size = std::max(num_splats, alloc_reserve);
    executeCompute(
        {{num_splats, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_projection_forward,
        {
            // inputs
            buffers.xyz_ws.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.scales_opacs.deviceBuffer,
            // outputs
            reserveDeviceBuffer(buffers.tiles_touched, alloc_size),
            reserveDeviceBuffer(buffers.rect_tile_space, alloc_size),
            reserveDeviceBuffer(buffers.radii, alloc_size),
            reserveDeviceBuffer(buffers.xy_vs, 2*alloc_size),
            reserveDeviceBuffer(buffers.depths, alloc_size),
            reserveDeviceBuffer(buffers.inv_cov_vs_opacity, 4*alloc_size),
            reserveDeviceBuffer(buffers.rgb, 3*alloc_size),
        }
    );

}

void VulkanGSRenderer::executeGenerateKeys(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::GenerateKeys> timer(this);

    size_t num_elements = buffers.num_splats;
    size_t num_indices = buffers.num_indices;

    // Reset ping-pong parity and pre-reserve sorting buffers with headroom to avoid per-step descriptor updates
    // (deferred steps: the capacity is ensured by the caller and the count is unknown on the host)
    const bool deferred = gate_active;
    buffers.is_unsorted_1 = true;
    if (!deferred && buffers.sorting_keys_1.deviceBuffer.allocSize < num_indices * sizeof(sortingKey_t)) {
        size_t reserve = std::max(num_indices + (num_indices >> 1), num_elements * 16);
        resizeDeviceBuffer(buffers.sorting_keys_1, reserve);
        resizeDeviceBuffer(buffers.sorting_keys_2, reserve);
        resizeDeviceBuffer(buffers.sorting_gauss_idx_1, reserve);
        resizeDeviceBuffer(buffers.sorting_gauss_idx_2, reserve);
        resizeDeviceBuffer(buffers._sorting_histogram_cumsum, _CEIL_DIV(reserve, 4096) * 256);
    }
    if (!deferred)
        writeIsectCount((uint32_t)num_indices);  // sort and tile-range kernels read the count from a buffer

    // barrier shouldn't be needed as this is after cumsum and read element
    #if 0
    bufferMemoryBarrier({
        { buffers.xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rect_tile_space.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);
    #endif

    executeCompute(
        {{num_elements, 64}},
        &uniforms, sizeof(uniforms),
        pipeline_generate_keys,
        {
            // inputs
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.depths.deviceBuffer,
            buffers.rect_tile_space.deviceBuffer,
            buffers.index_buffer_offset.deviceBuffer,
            // outputs
            deferred ? buffers.unsorted_keys().deviceBuffer : resizeDeviceBuffer(buffers.unsorted_keys(), num_indices),
            deferred ? buffers.unsorted_gauss_idx().deviceBuffer : resizeDeviceBuffer(buffers.unsorted_gauss_idx(), num_indices),
        },
        GATE_GENERATE_KEYS
    );

}

void VulkanGSRenderer::executeComputeTileRanges(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::ComputeTileRanges> timer(this);

    size_t num_indices = buffers.num_indices;
    size_t num_tiles = (size_t)(uniforms.grid_height*uniforms.grid_width);

    bufferMemoryBarrier({
        { buffers.sorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    executeCompute(
        {{num_indices+1, 256}},
        &uniforms, sizeof(uniforms),
        pipeline_compute_tile_ranges[buffers.is_unsorted_1],
        {
            // inputs
            buffers.sorted_keys().deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.tile_ranges, num_tiles+1),
            // intersection count (written by writeIsectCount or the step gate)
            isect_count,
        },
        GATE_TILE_RANGES
    );
}

void VulkanGSRenderer::executeRasterizeForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);

    size_t num_pixels = uniforms.image_height * uniforms.image_width;

    if (!gate_active && buffers.num_indices == 0) {
        clearDeviceBuffer(buffers.pixel_state, 4 * num_pixels);
        clearDeviceBuffer(buffers.n_contributors, num_pixels);
        return;
    }

    bufferMemoryBarrier({
        { buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.tile_ranges.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    executeCompute(
        {{uniforms.image_width, TILE_WIDTH}, {uniforms.image_height, TILE_HEIGHT}},
        &uniforms, sizeof(uniforms),
        pipeline_rasterize_forward[buffers.is_unsorted_1],
        std::vector<_VulkanBuffer>({
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.pixel_state, 4*num_pixels),
            resizeDeviceBuffer(buffers.n_contributors, num_pixels),
        }),
        GATE_RASTER_FWD
    );

    // _displayImage(buffers.pixel_state, uniforms.image_width, true);
    // exit(0);
}

void VulkanGSRenderer::executeRasterizeForwardDepth(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);

    size_t num_pixels = uniforms.image_height * uniforms.image_width;

    if (!gate_active && buffers.num_indices == 0) {
        clearDeviceBuffer(buffers.pixel_state, 4 * num_pixels);
        clearDeviceBuffer(buffers.n_contributors, num_pixels);
        clearDeviceBuffer(buffers.pixel_depth, 4 * num_pixels);
        return;
    }

    bufferMemoryBarrier({
        { buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.tile_ranges.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    executeCompute(
        {{uniforms.image_width, TILE_WIDTH}, {uniforms.image_height, TILE_HEIGHT}},
        &uniforms, sizeof(uniforms),
        pipeline_rasterize_forward_depth[buffers.is_unsorted_1],
        std::vector<_VulkanBuffer>({
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.pixel_state, 4 * num_pixels),
            resizeDeviceBuffer(buffers.n_contributors, num_pixels),
            // depth input & output
            buffers.depths.deviceBuffer,
            resizeDeviceBuffer(buffers.pixel_depth, 4 * num_pixels),
        }),
        GATE_RASTER_FWD
    );
}


void VulkanGSRenderer::initRasterizationBackwardScheduler() {
    if (!rasterizeBackwardScheduler.empty())
        return;

    // empirically tuned per vendor to filter out slow ones
    if (deviceInfo.vendor == DeviceVendor::Intel_R_)
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerPixel);
    else if (deviceInfo.vendor == DeviceVendor::NVIDIA ||
             deviceInfo.vendor == DeviceVendor::AMD) {
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerSplat);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_8);
    }
    else {
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerPixel);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerSplat);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_0);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_8);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_1_16_0);
    }

    for (size_t i = 0; i < rasterizeBackwardAlternatives.size(); ++i) {
        int scheduled_impl = (int)rasterizeBackwardAlternatives[i];
        for (int j = 0; j < 2; ++j)
            if (pipeline_rasterize_backward[scheduled_impl][j].shader == VK_NULL_HANDLE) {
                printf("WARNING: a backward implementation is disabled due to hardware limitation.\n");
                rasterizeBackwardAlternatives.erase(rasterizeBackwardAlternatives.begin()+i);
                --i;
                break;
            }
    }
    if (rasterizeBackwardAlternatives.empty())
        throw std::runtime_error("No rasterization backward implementation is available");

    rasterizeBackwardScheduler = ThompsonSamplingScheduler(
        rasterizeBackwardAlternatives.size(),
        0.0002,  // initial_noise_std, in seconds (0.2 ms, matched to per-step backward latency)
        200,  // max_no_update
        1000,  // adapt_tau
        10  // warmup_tau
    );
    rasterizeBackwardScheduler.set_pin_after(backward_pin_after);

    auto timerCallback = [&](const std::vector<std::pair<size_t, double>>& time_updates) {
        for (size_t impl = 0; impl < (size_t)RasterBackwardImpl::size; impl++) {
            auto [count, dt] = time_updates[PerfTimer::_RasterizeBackwardScheduling_PerPixel + impl];
            if (count == 0)
                continue;
            auto alternative = std::find(
                rasterizeBackwardAlternatives.begin(), rasterizeBackwardAlternatives.end(),
                (RasterBackwardImpl)impl);
            if (alternative != rasterizeBackwardAlternatives.end())
                rasterizeBackwardScheduler.update(
                    alternative - rasterizeBackwardAlternatives.begin(), dt / double(count));
        }
    };
    timerCallbacks.push_back(timerCallback);
}

void VulkanGSRenderer::executeRasterizeBackward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer* train_depth_weight_buf
) {
    DEVICE_GUARD;
    std::shared_ptr<void> timer = nullptr;
    RasterBackwardImpl scheduled_impl = RasterBackwardImpl::Default;
    if (RASTERIZE_BACKWARD_USE_SCHEDULING) {
        using namespace PerfTimer;
        initRasterizationBackwardScheduler();
        scheduled_impl = gate_active ? (RasterBackwardImpl)gate_backward_impl : sampleBackwardAlternative();
        // scheduled_impl = RasterBackwardImpl::PerSplat;
        if (scheduled_impl == RasterBackwardImpl::PerPixel)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_PerPixel>>(this);
        else if (scheduled_impl == RasterBackwardImpl::PerSplat)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_PerSplat>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_0_8_0)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_0_8_0>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_0_8_8)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_0_8_8>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_1_16_0)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_1_16_0>>(this);
    }
    else {
        timer = std::make_shared<PerfTimer::Timer<PerfTimer::RasterizeBackward>>(this);
    }

    size_t num_elements = buffers.num_splats;
    const bool ago_enabled = (uniforms.ago_telemetry_enabled != 0u);
    const bool depth_enabled = (uniforms.depth_loss_scale > 0.0f && train_depth_weight_buf != nullptr &&
                                train_depth_weight_buf->buffer != VK_NULL_HANDLE);

    clearDeviceBuffer(buffers.v_xy_vs, 2*num_elements);
    clearDeviceBuffer(buffers.v_inv_cov_vs_opacity, 4*num_elements);
    clearDeviceBuffer(buffers.v_rgb, 3*num_elements);
    if (ago_enabled) {
        clearDeviceBuffer(ago_contrib_step_buf_, std::max<size_t>(1, num_elements));
        clearDeviceBuffer(ago_abs_grad_step_buf_, std::max<size_t>(1, num_elements));
    }
    if (depth_enabled) {
        clearDeviceBuffer(buffers.v_depths, std::max<size_t>(1, num_elements));
    }
    if (!gate_active && buffers.num_indices == 0)
        return;

    std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers = {
        { buffers.n_contributors.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_xy_vs.deviceBuffer, TRANSFER_WRITE },
        { buffers.v_inv_cov_vs_opacity.deviceBuffer, TRANSFER_WRITE },
        { buffers.v_rgb.deviceBuffer, TRANSFER_WRITE },
    };
    if (ago_enabled) {
        barriers.push_back({ ago_contrib_step_buf_.deviceBuffer, TRANSFER_WRITE });
        barriers.push_back({ ago_abs_grad_step_buf_.deviceBuffer, TRANSFER_WRITE });
    }
    if (depth_enabled) {
        barriers.push_back({ buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE });
        barriers.push_back({ buffers.pixel_depth.deviceBuffer, COMPUTE_SHADER_WRITE });
        barriers.push_back({ buffers.v_depths.deviceBuffer, TRANSFER_WRITE });
    }
    bufferMemoryBarrier(barriers, COMPUTE_SHADER_READ_WRITE);

    std::vector<std::pair<size_t, size_t>> dims;
    if (scheduled_impl == RasterBackwardImpl::PerPixel)
        dims = {{(int)uniforms.image_width, TILE_WIDTH}, {(int)uniforms.image_height, TILE_HEIGHT}};
    else
        dims = {{uniforms.grid_width*uniforms.grid_height, 1}};
    const bool sparse = uniforms.use_tile_list != 0;
    {
        auto& w = PerfTimer::work_counters;
        const uint64_t grid_tiles = (uint64_t)uniforms.grid_width * uniforms.grid_height;
        w.backward_dispatches++;
        w.backward_tiles_total += grid_tiles;
        if (sparse)
            w.backward_sparse_passes++;
        else
            w.backward_tiles_dispatched += grid_tiles;  // sparse passes are counted on the GPU
    }

    VulkanGSRendererUniforms effective_uniforms = uniforms;
    if (!depth_enabled) {
        effective_uniforms.depth_loss_scale = 0.0f;
    }

    executeCompute(
        dims,
        &effective_uniforms, sizeof(effective_uniforms),
        RASTERIZE_BACKWARD_USE_SCHEDULING ? pipeline_rasterize_backward[(size_t)scheduled_impl][buffers.is_unsorted_1] :
            pipeline_rasterize_backward[1][buffers.is_unsorted_1],  // per-splat backward
        {
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            buffers.pixel_state.deviceBuffer,
            buffers.n_contributors.deviceBuffer,
            buffers.v_pixel_state.deviceBuffer,
            // outputs
            buffers.v_xy_vs.deviceBuffer,
            buffers.v_inv_cov_vs_opacity.deviceBuffer,
            buffers.v_rgb.deviceBuffer,
            sparse_active_tiles,
            // group & per-splat tables (read only when uniforms.freeze_flags != 0; any valid buffer is bound otherwise)
            uniforms.freeze_flags ? buffers.group_ids.deviceBuffer : sparse_active_tiles,
            uniforms.freeze_flags ? buffers.group_state.deviceBuffer : sparse_active_tiles,
            uniforms.freeze_flags ? buffers.splat_state.deviceBuffer : sparse_active_tiles,
            // AGO-0 per-step visibility & AbsGS buffers (written only when ago_telemetry_enabled != 0)
            ago_enabled ? ago_contrib_step_buf_.deviceBuffer : sparse_active_tiles,
            ago_enabled ? ago_abs_grad_step_buf_.deviceBuffer : sparse_active_tiles,
            // D1 physical expected-depth loss buffers (read/written only when depth_loss_scale > 0)
            depth_enabled ? buffers.depths.deviceBuffer : sparse_active_tiles,
            depth_enabled ? buffers.pixel_depth.deviceBuffer : sparse_active_tiles,
            depth_enabled ? *train_depth_weight_buf : sparse_active_tiles,
            depth_enabled ? buffers.v_depths.deviceBuffer : sparse_active_tiles,
        },
        GATE_RASTER_BWD,
        sparse ? &sparse_args : nullptr
    );

}


void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers &buffers,
    Buffer<int32_t> &input_buffer,
    Buffer<int32_t> &output_buffer
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::_Cumsum> timer(this);

    size_t num_elements = input_buffer.deviceSize();
    const size_t block_0 = 1024;
    const size_t block_limit = deviceInfo.subgroupSize*deviceInfo.subgroupSize*deviceInfo.subgroupSize;
    const size_t block = std::min(block_0, block_limit);

    uint32_t uniforms[2] = {
        (uint32_t)num_elements, 1
    };
    // int uniform_size = 2*sizeof(uint32_t);
    int uniform_size = 1*sizeof(uint32_t);

    bufferMemoryBarrier({
        { input_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    reserveDeviceBuffer(output_buffer, num_elements);

    if (num_elements <= block_0) {
        executeCompute(
            {{num_elements, block_0}},
            uniforms, uniform_size,
            pipeline_cumsum.single_pass,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
            }
        );
    }

    else if (num_elements <= block*block) {
        resizeDeviceBuffer(buffers._cumsum_blockSums, _CEIL_DIV(num_elements, block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
    }

    else if (num_elements <= block*block*block) {
        size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, _CEIL_DIV(num_elements_1,block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements_1/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
    }

    // can't reasonably expect more than 1G splats
    // although there may be more than 1G sorting indices
    else {
        _THROW_ERROR("Too many numbers for cumsum");
    }

}

void VulkanGSRenderer::executeCalculateIndexBufferOffset(
    VulkanGSPipelineBuffers& buffers,
    bool read_count
) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);

    size_t num_elements = buffers.num_splats;

    executeCumsum(
        buffers,
        buffers.tiles_touched,
        buffers.index_buffer_offset
    );

    if (!read_count)
        return;
    if (commandBatchInProgress) bufferMemoryBarrier({
        { buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);
    int num_indices = readElement<int32_t>(buffers.index_buffer_offset.deviceBuffer, num_elements-1);
    buffers.num_indices = (size_t)num_indices;
    {
        auto& w = PerfTimer::work_counters;
        w.forward_passes++;
        w.tile_intersections_total += (uint64_t)num_indices;
        w.tile_intersections_last = (uint64_t)num_indices;
        w.tile_intersections_max = std::max(w.tile_intersections_max, (uint64_t)num_indices);
    }
    // printf("num_splats=%d num_indices=%d\n", (int)num_elements, (int)num_indices);
}

void VulkanGSRenderer::executeSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::SortRTS> timer(this);

    size_t num_elements = buffers.unsorted_keys().deviceSize();
    if (num_elements != buffers.unsorted_gauss_idx().deviceSize())
        _THROW_ERROR("number of elements don't match in executeSort");

    const int RADIX = 256;
    const int WORKGROUP_SIZE = 512;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

    auto& globalHistogram = buffers._sorting_histogram;
    auto& partitionHistogram = buffers._sorting_histogram_cumsum;

    // deferred: num_elements is the buffer capacity (the dispatch size comes from the GPU-side count)
    const size_t num_parts = _CEIL_DIV(num_elements, PARTITION_SIZE);

    int max_nonzero_bit = 8*sizeof(sortingKey_t);
    if (num_bits == -1 && sizeof(sortingKey_t) == 8) {
        int32_t num_tiles = (int32_t)(uniforms.grid_height*uniforms.grid_width);
        max_nonzero_bit = 23;  // float fraction bits
        int32_t temp = num_tiles;
        while (temp)
            temp >>= 1, max_nonzero_bit++;
    }
    else if (num_bits >= 0)
        max_nonzero_bit = num_bits;
    int num_passes = _CEIL_DIV(max_nonzero_bit, 8);

    resizeDeviceBuffer(partitionHistogram, num_parts*RADIX);
    resizeDeviceBuffer(buffers.sorted_keys(), num_elements);
    resizeDeviceBuffer(buffers.sorted_gauss_idx(), num_elements);

    clearDeviceBuffer(globalHistogram, num_passes * sizeof(sortingKey_t)*RADIX);
    bufferMemoryBarrier({
        { globalHistogram.deviceBuffer, TRANSFER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    for (int pass = 0; 8*pass < max_nonzero_bit; pass++) {

        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_1 : pipeline_sorting_2;

        uint32_t uniforms[2];
        uniforms[0] = (uint32_t)pass;
        uniforms[1] = (uint32_t)num_elements;

        if (pass)
        bufferMemoryBarrier({
            { buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.upsweep,
            {
                buffers.unsorted_keys().deviceBuffer,
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
                isect_count,
            },
            GATE_SORT
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{RADIX, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
                isect_count,
            }
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.downsweep,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
                buffers.unsorted_keys().deviceBuffer,
                buffers.unsorted_gauss_idx().deviceBuffer,
                buffers.sorted_keys().deviceBuffer,
                buffers.sorted_gauss_idx().deviceBuffer,
                isect_count,
            },
            GATE_SORT
        );

        buffers.is_unsorted_1 = !buffers.is_unsorted_1;
    }
    buffers.is_unsorted_1 = !buffers.is_unsorted_1;

}


// ---------------------------------------------------------------------------------------------
// Batched training steps: GPU-driven tile pipeline (no host readback of the intersection count)

void VulkanGSRenderer::writeIsectCount(uint32_t n) {
    DEVICE_GUARD;
    vkCmdUpdateBuffer(command_buffer, isect_count.buffer, 0, sizeof(uint32_t), &n);
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

size_t VulkanGSRenderer::isectCapacity(const VulkanGSPipelineBuffers& buffers) const {
    // all four ping-pong buffers are allocated together; keep one spare key for the tile-range kernel
    size_t alloc = std::min({
        buffers.sorting_keys_1.deviceBuffer.allocSize, buffers.sorting_keys_2.deviceBuffer.allocSize,
        buffers.sorting_gauss_idx_1.deviceBuffer.allocSize, buffers.sorting_gauss_idx_2.deviceBuffer.allocSize});
    size_t elements = alloc / sizeof(sortingKey_t);
    return elements > 0 ? elements - 1 : 0;
}

void VulkanGSRenderer::ensureIsectCapacity(VulkanGSPipelineBuffers& buffers, size_t want) {
    size_t capacity = isectCapacity(buffers);
    if (capacity < want) {
        // contents are scratch; geometric growth like the append buffers
        capacity = std::max(want, capacity + (capacity >> 1));
        const size_t elements = capacity + 1;
        resizeDeviceBuffer(buffers.sorting_keys_1, elements);
        resizeDeviceBuffer(buffers.sorting_keys_2, elements);
        resizeDeviceBuffer(buffers.sorting_gauss_idx_1, elements);
        resizeDeviceBuffer(buffers.sorting_gauss_idx_2, elements);
        resizeDeviceBuffer(buffers._sorting_histogram_cumsum, _CEIL_DIV(elements, 4096) * 256);
        PerfTimer::work_counters.step_buffer_reallocs++;
    }
    // logical size = the whole allocation (a legacy step may have set it to its exact count)
    for (auto* b : {&buffers.sorting_keys_1, &buffers.sorting_keys_2})
        b->deviceBuffer.size = b->deviceBuffer.allocSize;
    for (auto* b : {&buffers.sorting_gauss_idx_1, &buffers.sorting_gauss_idx_2})
        b->deviceBuffer.size = b->deviceBuffer.allocSize;
}

void VulkanGSRenderer::setIsectCapacityExact(VulkanGSPipelineBuffers& buffers, size_t capacity) {
    const size_t elements = capacity + 1;
    for (auto* b : {&buffers.sorting_keys_1, &buffers.sorting_keys_2})
        resizeDeviceBuffer(*b, elements, false);
    for (auto* b : {&buffers.sorting_gauss_idx_1, &buffers.sorting_gauss_idx_2})
        resizeDeviceBuffer(*b, elements, false);
    resizeDeviceBuffer(buffers._sorting_histogram_cumsum, _CEIL_DIV(elements, 4096) * 256, false);
}

void VulkanGSRenderer::gateBeginBatch() {
    // status = 0: no step done, no overflow
    vkCmdFillBuffer(command_buffer, gate_status.buffer, 0, gate_status.size, 0);
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    gate_active = true;
    gate_step = 0;
}

void VulkanGSRenderer::gateBeginStep(uint32_t step_in_batch) {
    if (step_in_batch >= kGateMaxSteps)
        throw std::runtime_error("too many steps in one gated batch");
    gate_step = step_in_batch;
    timestamp_scheduling_only = step_in_batch > 0;
    // WAR/RAW hazards between the scratch buffers of consecutive steps (the fence used to provide this)
    if (step_in_batch > 0)
        fullBarrier();
}

void VulkanGSRenderer::executeTilesDeferred(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const GatePlan& plan
) {
    DEVICE_GUARD;
    gate_plan = plan;

    executeCalculateIndexBufferOffset(buffers, false);
    fullBarrier();  // projection outputs and the prefix sum are read by the gate and by key generation

    struct { uint32_t num_splats, capacity, step_idx, pad; uint32_t base[kGateSlots][4]; } gate_uniforms = {};
    gate_uniforms.num_splats = (uint32_t)buffers.num_splats;
    gate_uniforms.capacity = (uint32_t)isectCapacity(buffers);
    gate_uniforms.step_idx = gate_step;
    memcpy(gate_uniforms.base, plan.base, sizeof(plan.base));
    executeCompute(
        {{1, 1}},
        &gate_uniforms, sizeof(gate_uniforms),
        pipeline_step_gate,
        {
            buffers.index_buffer_offset.deviceBuffer,
            gate_status,
            isect_count,
            gate_args,
        }
    );
    {  // gate outputs: indirect arguments, count, status
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    executeGenerateKeys(uniforms, buffers);
    executeSort(uniforms, buffers, -1);
    executeComputeTileRanges(uniforms, buffers);
}

void VulkanGSRenderer::gateEndBatch() {
    gate_active = false;
    timestamp_scheduling_only = false;
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    VkBufferCopy region = {};
    region.size = kGateStatusWords * sizeof(uint32_t);
    vkCmdCopyBuffer(command_buffer, gate_status.buffer, gate_host_buffer, 1, &region);
    VkMemoryBarrier host = {};
    host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &host, 0, nullptr, 0, nullptr);
}

VulkanGSRenderer::GateResult VulkanGSRenderer::gateResult() const {
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = gate_host_memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(device, 1, &range);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    GateResult r;
    r.steps_done = gate_host_status[0];
    r.overflow = gate_host_status[1] != 0;
    r.needed = gate_host_status[2];
    for (uint32_t i = 0; i < kGateMaxSteps; i++)
        r.isects[i] = gate_host_status[4 + i];
    return r;
}

void VulkanGSRenderer::ensureSparseTileBuffers(size_t num_tiles) {
    if (num_tiles <= sparse_capacity)
        return;
    if (commandBatchInProgress)
        _THROW_ERROR("sparse tile buffers must be grown outside a command batch");
    for (auto* b : {&sparse_residual, &sparse_mask, &sparse_active_tiles}) {
        destroyBuffer(*b);
        createBuffer(num_tiles * sizeof(uint32_t), *b);
    }
    sparse_capacity = num_tiles;
}

void VulkanGSRenderer::executeSelectSparseTiles(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    float quantile,
    int halo
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::SelectSparseTiles> timer(this);
    const uint32_t num_tiles = uniforms.grid_width * uniforms.grid_height;
    if (num_tiles > sparse_capacity)
        _THROW_ERROR("sparse tile buffers too small for this image");

    const uint32_t residual_uniforms[4] = {uniforms.image_width, uniforms.image_height, uniforms.grid_width, 0};
    executeCompute(
        {{uniforms.grid_width, 1}, {uniforms.grid_height, 1}},
        residual_uniforms, sizeof(residual_uniforms),
        pipeline_sparse_residual,
        { buffers.pixel_state.deviceBuffer, buffers.ref_image.deviceBuffer, sparse_residual }
    );
    fullBarrier();

    struct { uint32_t num_tiles, grid_width, keep, full, halo, args_offset, gate_ok_index, pad; } select_uniforms = {};
    select_uniforms.num_tiles = num_tiles;
    select_uniforms.grid_width = uniforms.grid_width;
    select_uniforms.keep = std::max<uint32_t>(1, (uint32_t)std::ceil((1.0 - (double)quantile) * num_tiles));
    select_uniforms.full = 0;
    select_uniforms.halo = (uint32_t)std::max(halo, 0);
    select_uniforms.args_offset = 0;
    select_uniforms.gate_ok_index = gate_active ? (gate_step * kGateSlots + GATE_RASTER_BWD) * 4 : 0xffffffffu;
    executeCompute(
        {{1, 1}},
        &select_uniforms, sizeof(select_uniforms),
        pipeline_sparse_select,
        { sparse_residual, sparse_mask, sparse_active_tiles, sparse_args, gate_args, sparse_counters }
    );
    {  // list, arguments and counter: read by the SSIM and raster backward kernels (indirect arguments included)
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
}

uint64_t VulkanGSRenderer::sparseTilesDispatched() {
    if (commandBatchInProgress)
        _THROW_ERROR("sparseTilesDispatched called inside a command batch");
    // a diagnostic readback, not part of any step: keep fence_waits_total a measure of the step path
    auto& fences = PerfTimer::work_counters.fence_waits_total;
    const uint64_t fences_before = fences;
    const uint32_t now = (uint32_t)readElement<int32_t>(sparse_counters, 0);
    fences = fences_before;
    sparse_counter_total += (uint32_t)(now - sparse_counter_last);
    sparse_counter_last = now;
    return sparse_counter_total;
}

int VulkanGSRenderer::sampleBackwardImpl() {
    initRasterizationBackwardScheduler();
    gate_backward_impl = (int)sampleBackwardAlternative();
    return gate_backward_impl;
}

VulkanGSRenderer::RasterBackwardImpl VulkanGSRenderer::sampleBackwardAlternative() {
    if (forced_backward_impl >= 0)
        return (RasterBackwardImpl)forced_backward_impl;
    return rasterizeBackwardAlternatives[rasterizeBackwardScheduler.sample()];
}

std::vector<int> VulkanGSRenderer::backwardImplCandidates() {
    initRasterizationBackwardScheduler();
    std::vector<int> out;
    for (auto impl : rasterizeBackwardAlternatives)
        out.push_back((int)impl);
    return out;
}

void VulkanGSRenderer::forceBackwardImpl(int impl) {
    if (impl >= 0) {
        auto c = backwardImplCandidates();
        if (std::find(c.begin(), c.end(), impl) == c.end())
            throw std::invalid_argument("backward variant not available on this device");
    }
    forced_backward_impl = impl;
}

void VulkanGSRenderer::setBackwardPinAfter(size_t updates) {
    backward_pin_after = updates;
    if (!rasterizeBackwardScheduler.empty())
        rasterizeBackwardScheduler.set_pin_after(updates);
}


int32_t VulkanGSRenderer::executeSum(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t> &input_buffer
) {
    PerfTimer::Timer<PerfTimer::_Sum> timer(this);
    // DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();

    clearDeviceBuffer(buffers._temp_sum, 1);
    bufferMemoryBarrier({
        { buffers._temp_sum.deviceBuffer, TRANSFER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{num_elements, 1024}},
        &num_elements, sizeof(uint32_t),
        pipeline_sum,
        {
            input_buffer.deviceBuffer,
            buffers._temp_sum.deviceBuffer,
        }
    );

    bufferMemoryBarrier({
        { buffers._temp_sum.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);

    copyFromDevice(buffers._temp_sum);
    return buffers._temp_sum[0];
}

void VulkanGSRenderer::executeWhere(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t> &input_buffer,
    Buffer<int32_t> &output_buffer
) {
    DEVICE_GUARD;
    PerfTimer::Timer<PerfTimer::_Where> timer(this);

    size_t num_elements = input_buffer.deviceSize();

    executeCumsum(
        buffers,
        input_buffer,
        buffers._temp_cumsum
    );

    bufferMemoryBarrier({
        { buffers._temp_cumsum.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);
    int32_t num_nonzero = readElement<int32_t>(buffers._temp_cumsum.deviceBuffer, num_elements-1);

    executeCompute(
        {{num_elements, 256}},
        (uint32_t*)&num_elements, sizeof(uint32_t),
        pipeline_where,
        {
            input_buffer.deviceBuffer,
            buffers._temp_cumsum.deviceBuffer,
            resizeDeviceBuffer(output_buffer, num_nonzero),
        }
    );
}
