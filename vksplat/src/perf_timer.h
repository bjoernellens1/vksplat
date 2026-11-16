#pragma once

#include "gs_pipeline.h"

#include <chrono>
#include <mutex>
#include <map>


namespace PerfTimer {


#define PERF_TIMER_TRAIN_STAGES \
    _(ProjectionForward) \
    _(GenerateKeys) \
    _(ComputeTileRanges) \
    _(RasterizeForward) \
    _(RasterizeBackward) \
    _(_Cumsum) \
    _(CalculateIndexBufferOffset) \
    _(SortRTS) \
    _(_Sum) \
    _(_Where) \
    _(CopyTrainImageToDevice) \
    _(ComputeSSIMGradient) \
    _(FusedProjectionBackwardOptimizerStep) \
    _(DefaultPostBackward) \
    _(MCMCPostBackward) \
    _(MortonSorting) \
    _(_RasterizeBackwardScheduling_PerPixel) \
    _(_RasterizeBackwardScheduling_PerSplat) \
    _(_RasterizeBackwardScheduling_Tensor_0_8_0) \
    _(_RasterizeBackwardScheduling_Tensor_0_8_8) \
    _(_RasterizeBackwardScheduling_Tensor_1_16_0) \
    _(_RasterizeBackwardScheduling_N) \

#define _(name) name ,
enum TrainStage {
    PERF_TIMER_TRAIN_STAGES
    END
};
#undef _


void reset();

void hostTic();
void hostToc();

template<TrainStage stage>
struct Timer {
    VulkanGSPipeline *module;
    std::chrono::time_point<std::chrono::high_resolution_clock> then;
    // static std::mutex mutex;

    Timer(VulkanGSPipeline *module);
    ~Timer();
};

void pushMarker(VulkanGSPipeline *module);
void popMarkers(VulkanGSPipeline *module);

std::vector<std::pair<size_t, double>> update(std::vector<double> times);

std::map<std::string, std::tuple<size_t, double>> get_summary();

// Work counters accumulated from values the host already has (no extra GPU readback).
// Backward tiles skipped by the sparse_tile_grad_thresh early-exit are decided on the GPU and are
// not counted: that would need a device atomic plus a new readback.
struct WorkCounters {
    uint64_t forward_passes = 0;
    uint64_t tile_intersections_total = 0;  // sum of num_indices over forward passes
    uint64_t tile_intersections_last = 0;
    uint64_t tile_intersections_max = 0;
    uint64_t backward_dispatches = 0;
    uint64_t backward_tiles_dispatched = 0;  // grid_width * grid_height per backward dispatch
    // Reserved-capacity append: growth events of the per-splat buffer set, blocking fence waits inside
    // append calls, and reallocations of per-step (projection / gradient) buffers.
    uint64_t append_reallocs = 0;
    uint64_t append_fence_waits = 0;
    uint64_t step_buffer_reallocs = 0;
    uint64_t fence_waits_total = 0;  // every blocking fence wait (any submit path)
};
extern WorkCounters work_counters;

std::map<std::string, uint64_t> get_work_counters();


}  // namespace PerfTimer
