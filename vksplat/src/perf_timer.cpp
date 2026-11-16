#include "perf_timer.h"

#include <stack>

namespace PerfTimer {

// names of train stages as strings
#define _(name) #name ,
static std::string _TrainStageNames[TrainStage::END] = {
    PERF_TIMER_TRAIN_STAGES
};
#undef _


static struct _TimerObject {
    size_t count = 0;
    double total_time = 0.0;
} stages[TrainStage::END];

std::vector<std::pair<TrainStage, int>> marks;
std::vector<TrainStage> pushedMarks;

bool hostHold = false;
std::chrono::time_point<std::chrono::high_resolution_clock> hostStartTime;
double hostTimeDelta = -1.0;

WorkCounters work_counters;

// reset timing and counts
void reset() {
    for (int i = 0; i < int(TrainStage::END); i++) {
        stages[i] = {0, 0.0};
    }
    marks.clear();
    work_counters = {};
}

void hostTic() {
    if (hostHold)
        _THROW_ERROR("hostTic");
    hostHold = true;
    hostStartTime = std::chrono::high_resolution_clock::now();
}

void hostToc() {
    if (hostTimeDelta < 0.0) {
        hostTimeDelta = 0.0;
        return;
    }
    if (!hostHold)
        _THROW_ERROR("hostToc");
    hostHold = false;
    auto hostEndTime = std::chrono::high_resolution_clock::now();
    hostTimeDelta += std::chrono::duration<double>(hostEndTime-hostStartTime).count();
}

template<TrainStage stage>
Timer<stage>::Timer(VulkanGSPipeline *module) : module(module) {
    then = std::chrono::high_resolution_clock::now();
    constexpr bool kScheduling = stage >= _RasterizeBackwardScheduling_PerPixel && stage < _RasterizeBackwardScheduling_N;
    if ((kScheduling || !module->timestamp_scheduling_only) && module->writeTimestamp(1)) {
        marks.emplace_back(stage, 1);
        stamped = true;
    }
}

template<TrainStage stage>
Timer<stage>::~Timer() {
    // untimed stages of batched steps are not counted either, so total_time / count stays a per-step mean
    if (stamped || !module->timestamp_scheduling_only) {
        PerfTimer::stages[int(stage)].count += 1;
        if (stage >= _RasterizeBackwardScheduling_PerPixel && stage < _RasterizeBackwardScheduling_N)
            PerfTimer::stages[RasterizeBackward].count += 1;
    }

    if (stamped && module->writeTimestampNoExcept(-1))
        marks.emplace_back(stage, -1);
}

void pushMarker(VulkanGSPipeline *module) {

    if (!module->writeTimestamp(-1))
        _THROW_ERROR("Failed to write exit timestamp in pushMarker");
    
    int depth = 1;
    for (int i = (int)marks.size()-1; i >= 0; --i) {
        auto [stage, delta] = marks[i];
        depth -= delta;
        if (depth == 0) {
            pushedMarks.push_back(stage);
            marks.emplace_back(stage, -1);
            return;
        }
    }
    _THROW_ERROR("Empty stack in pushMarker");
}

void popMarkers(VulkanGSPipeline *module) {
    while (!pushedMarks.empty()) {
        auto stage = pushedMarks.back();
        pushedMarks.pop_back();
        PerfTimer::stages[int(stage)].total_time += hostTimeDelta;
        if (!module->writeTimestamp(1))
            _THROW_ERROR("Failed to write enter timestamp in popMarkers");
        marks.emplace_back(stage, 1);
    }
    hostTimeDelta = 0.0;
}

std::vector<std::pair<size_t, double>> update(std::vector<double> times) {
    if (times.size() != marks.size())
        _THROW_ERROR(
            "Number of timestamps (" + std::to_string(times.size()) +
            ") and number of marks (" + std::to_string(marks.size()) +
            ") mismatch in batch time update"
        );
    std::vector<std::pair<size_t, double>> results(TrainStage::END, {0, 0.0});
    std::vector<std::pair<TrainStage, double>> stack;
    for (size_t i = 0; i < times.size(); i++) {
        auto [stage, delta] = marks[i];
        if (delta == 1) {
            stack.emplace_back(stage, times[i]);
        }
        else {
            double dt = times[i] - stack.back().second;
            PerfTimer::stages[int(stage)].total_time += dt;
            if (stage >= _RasterizeBackwardScheduling_PerPixel && stage < _RasterizeBackwardScheduling_N)
                PerfTimer::stages[RasterizeBackward].total_time += dt;
            stack.pop_back();
            results[stage].first += 1;
            results[stage].second += dt;
        }
    }
    marks.clear();
    return results;
}

// return a summary of timing
std::map<std::string, std::tuple<size_t, double>> get_summary() {
    std::map<std::string, std::tuple<size_t, double>> summary;
    for (int i = 0; i < int(TrainStage::END); i++) {
        std::string name = _TrainStageNames[i];
        summary[name] = { stages[i].count, stages[i].total_time };
    }
    return summary;
}

std::map<std::string, uint64_t> get_work_counters() {
    const WorkCounters& w = work_counters;
    return {
        {"forward_passes", w.forward_passes},
        {"tile_intersections_total", w.tile_intersections_total},
        {"tile_intersections_last", w.tile_intersections_last},
        {"tile_intersections_max", w.tile_intersections_max},
        {"backward_dispatches", w.backward_dispatches},
        {"backward_tiles_dispatched", w.backward_tiles_dispatched},
        {"backward_tiles_total", w.backward_tiles_total},
        {"backward_sparse_passes", w.backward_sparse_passes},
        {"append_reallocs", w.append_reallocs},
        {"append_fence_waits", w.append_fence_waits},
        {"step_buffer_reallocs", w.step_buffer_reallocs},
        {"fence_waits_total", w.fence_waits_total},
        {"gate_overflows", w.gate_overflows},
        {"gate_steps_redone", w.gate_steps_redone},
    };
}

// template instantiation of timers
#define _(name) template struct Timer<name>;
PERF_TIMER_TRAIN_STAGES
#undef _
// template <TrainStage name> Timer<name>::Timer();
// template <TrainStage name> Timer<name>::~Timer();

};  // namespace PerfTimer
