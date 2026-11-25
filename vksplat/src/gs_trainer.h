#pragma once

#include "gs_renderer.h"

#include "colmap_reader.h"
#include "ago/core/action_protocol.h"
#include "ago/runtime/gpu_executor.h"

#include <random>
#include <atomic>
#include <limits>
#include <unordered_map>


// see simple_trainer.py for details
struct TrainerConfig {

    std::string output_dir;
    std::string output_ply;

    // dataset
    std::string dataset_dir;
    std::string image_dir;
    std::string mask_dir;
    std::string sparse_dir;
    int eval_interval;

    enum class CacheImage {
        CPU, GPU
    } image_cache_device;

    float global_scale;
    float init_scale;
    float init_opacity;

    // Optional Sceneweave controlled-init ablation. Zero preserves legacy
    // normalization and 4-NN scales over the complete COLMAP cloud.
    // Nonzero: use the first N seed points as a common normalization/4-NN
    // reference and initialize appended points with a fixed world-metric scale.
    size_t init_reference_count = 0;
    float init_added_scale_m = 0.0f;
    // 0 legacy random/isotropic, 1 local surface normal + anisotropy,
    // 2 random orientation + identical anisotropic scales (capacity control).
    int init_surface_mode = 0;
    float init_normal_scale_ratio = 0.25f;
    // >0 freezes 3D mean updates and post-backward densification
    // once step >= freeze_geometry_after_iter.
    int freeze_geometry_after_iter = 0;
    // Orthogonal mechanism-isolating flags (decoupled from freeze_geometry_after_iter)
    bool enable_lr_boost = false;
    int max_sh_degree_override = -1;
    bool enable_pre_freeze_prune = false;
    bool selective_plane_freeze = false;
    float unfreeze_loss_spike_ratio = 1.35f;

    enum class Strategy {
        Default, MCMC,
    } strategy;

    // optimizer
    int max_steps;
    float ssim_lambda;
    int ssim_every = 1;       // SSIM term only on steps with step % ssim_every == 0 (L1 only otherwise)
    int ssim_downsample = 1;  // 2: SSIM on the 2x2 box-averaged images
    float means_lr;
    float means_lr_final;
    float features_dc_lr;
    float features_rest_lr;
    float opacities_lr;
    float scales_lr;
    float quats_lr;
    float scale_reg;
    float opacity_reg;
    float feature_lr_final_scale = 1.0f;
    float opacity_lr_final_scale = 1.0f;
    float scale_lr_final_scale = 1.0f;
    float structural_reset_opa = 0.25f;

    inline float get_means_lr(int step, float scene_scale) const {
        float progress = fmin(float(step)/float(std::max(1, this->max_steps)), 1.0f);
        return this->means_lr * pow(this->means_lr_final/this->means_lr, progress) * scene_scale;
    }

    inline float get_lr_decay(int step, float final_scale) const {
        if (final_scale <= 0.0f || final_scale == 1.0f || this->max_steps <= 0)
            return (final_scale <= 0.0f) ? 0.0f : 1.0f;
        float progress = std::clamp(float(step) / float(this->max_steps), 0.0f, 1.0f);
        return std::pow(final_scale, progress);
    }

    // strategy
    int refine_start_iter;
    int refine_stop_iter;
    int refine_every;

    // default strategy
    float prune_opa;
    float grow_grad2d;
    float grow_scale3d;
    float grow_scale2d;
    float prune_scale3d;
    float prune_scale2d;
    int refine_scale2d_stop_iter;
    int reset_every;
    int stop_reset_at;
    int pause_refine_after_reset;

    // MCMC & capacity strategy
    float noise_lr;
    float min_opacity;
    float grow_factor;
    int cap_max;
    int cap_alloc_max = 0;  // monotonic device allocation reserve for MCMC when cap_max ramps dynamically
    int cap_speculative_max = 0;  // >0 caps structural (gid >= 0) splats during ADC

    inline size_t mcmc_alloc_cap() const {
        return static_cast<size_t>(std::max(cap_alloc_max, cap_max));
    }
    float surfel_normal_cap_m = 0.03f;  // world-metric normal thickness cap for L_normal
    bool mcmc_protect_structural_groups = false;
    // raster backward drops the geometry/opacity (IMAGINED) or all (protected LOCKED) gradient atomics of frozen splats
    bool freeze_skip_backward = true;
    // Sparse backward: on non-sentinel steps only the tiles whose residual is at least the
    // `sparse_tile_quantile`-quantile of all tile residuals (plus `sparse_tile_halo` tiles of dilation) get
    // a raster/SSIM backward pass. 0 = dense. Every `sparse_full_every`-th step (step % k == 0) is dense.
    float sparse_tile_quantile = 0.0f;
    int sparse_tile_halo = 1;
    int sparse_full_every = 4;
    // >0: stop exploring the rasterize-backward variants after this many timing updates and keep the
    // fastest measured one (0 = keep sampling, the default).
    int rasterize_backward_pin_after = 20;
    // AGO-0 External Strategy Ownership: when true, disables internal ADC densification,
    // MCMC growth/relocation, and global opacity resets so AGO owns splat lifecycle.
    bool external_strategy_mode = false;
    // AGO-L0 Parameter-specific refinement bitmask:
    //   PARAM_POS=1, PARAM_ROT=2, PARAM_SCALE=4, PARAM_OPAC=8, PARAM_SH=16 (default 31 = all)
    static constexpr uint32_t AGO_PARAM_POS   = 1u;
    static constexpr uint32_t AGO_PARAM_ROT   = 2u;
    static constexpr uint32_t AGO_PARAM_SCALE = 4u;
    static constexpr uint32_t AGO_PARAM_OPAC  = 8u;
    static constexpr uint32_t AGO_PARAM_SH    = 16u;
    static constexpr uint32_t AGO_PARAM_ALL   = 31u;
    uint32_t ago_param_mask = AGO_PARAM_ALL;
};



class VulkanGSTrainer : public VulkanGSRenderer {
public:
    VulkanGSTrainer();
    ~VulkanGSTrainer();

    struct Camera {
        // https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
        enum Model {
            SIMPLE_PINHOLE = 0,  // f cx cy
            PINHOLE = 1,  // fx fy cx cy
            SIMPLE_RADIAL = 2,  // f cx cy k
            RADIAL = 3,  // f cx cy k1 k2
            OPENCV = 4,  // fx fy cx cy k1 k2 p1 p2
            OPENCV_FISHEYE = 5,  // fx fy cx cy k1 k2 k3 k4
            FULL_OPENCV = 6,  // fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
            FOV = 7,  // fx fy cx cy omega
            SIMPLE_RADIAL_FISHEYE = 8,  // f cx cy k
            RADIAL_FISHEYE = 9,  // f cx cy k1 k2
            THIN_PRISM_FISHEYE = 10,  // fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
            RAD_TAN_THIN_PRISM_FISHEYE = 11,  // fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3
        } model;
        int h, w;
        float fx, fy, cx, cy;
        float k1, k2, p1, p2, k3, k4, k5, k6, sx1, sy1;
        glm::mat4 world_view_transform;
    };

    struct DatasetImage {
        std::string image_path;
        std::string mask_path;
        Camera camera;
        Buffer<uint8_t> buffer;  // (h, w, 4)
        Buffer<float> depth_weight;  // (h * w * 2) interleaved [depth_m, weight], empty if no depth attached
        bool has_depth = false;
    };

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id);
    void cleanup();

    void load_colmap_dataset(const TrainerConfig &config, VulkanGSPipelineBuffers& buffers);

    size_t num_train() const { return dataset_train.size(); }
    size_t num_val() const { return dataset_val.size(); }

    DatasetImage& get_train_image(size_t idx) { return dataset_train[idx]; }
    DatasetImage& get_val_image(size_t idx) { return dataset_val[idx]; }
    glm::mat4 get_dataparser_transform() const { return dataparser_transform; }

    void camera_to_uniforms(const Camera &cam, VulkanGSRendererUniforms &uniforms) const;
    void pose_to_uniforms(
        int h, int w, float fx, float fy, float cx, float cy,
        const float* c2w_world_row_major,
        VulkanGSRendererUniforms &uniforms
    ) const;
    float get_world_to_normalized_scale() const;
    void get_train_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;
    size_t train_tile_count(size_t idx) const;  // raster tiles of training view idx
    void get_val_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;

    // true if the step has no densification/prune/reset/Morton work and can run in a gated batch
    bool stepIsDeferrable(const TrainerConfig &config, int step) const;
    GatePlan buildGatePlan(const VulkanGSRendererUniforms& uniforms, const VulkanGSPipelineBuffers& buffers, int backward_impl) const;
    void executeComputeSSIMGradient(const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, size_t train_idx, int step);
    void executeFusedProjectionBackwardOptimizerStep(const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int step);

    void executeDefaultPostBackward(const TrainerConfig &config, const VulkanGSRendererUniforms& renderer_uniforms, VulkanGSPipelineBuffers& buffers, int step);
    void executeMCMCPostBackward(const TrainerConfig &config, const VulkanGSRendererUniforms& renderer_uniforms, VulkanGSPipelineBuffers& buffers, int step);
    void executeMortonSorting(const VulkanGSRendererUniforms &uniforms, VulkanGSPipelineBuffers& buffers);

    void writePLY(std::string filename, VulkanGSPipelineBuffers& buffers);

    size_t get_num_anchored_splats() const { return num_anchored_splats; }
    bool is_selective_freeze_unfrozen() const { return selective_freeze_unfrozen; }
    const std::vector<int32_t>& get_splat_group_ids() const { return splat_group_ids; }
    const std::vector<int32_t>& get_splat_entity_ids() const { return splat_entity_ids; }
    const std::vector<uint32_t>& get_splat_states() const { return splat_states; }
    size_t get_mcmc_relocated_count() const { return mcmc_relocated_count; }
    size_t get_mcmc_added_count() const { return mcmc_added_count; }
    void set_growth_cap(size_t ceiling) { runtime_growth_cap.store(ceiling, std::memory_order_relaxed); }
    size_t get_growth_cap() const { return runtime_growth_cap.load(std::memory_order_relaxed); }
    void set_selective_freeze_unfrozen(bool unfrozen) {
        selective_freeze_unfrozen = unfrozen;
        if (!unfrozen) {
            anchored_xyz_snapshot_valid = false;
        }
    }
    void reset_anchored_snapshot(size_t new_num_anchored) {
        num_anchored_splats = new_num_anchored;
        anchored_xyz_snapshot_valid = false;
        if (num_anchored_splats > 0) {
            resizeDeviceBuffer(anchored_xyz_ws, 3 * num_anchored_splats);
        }
    }
    void set_splat_group_ids(const std::vector<int32_t>& gids) {
        splat_group_ids = gids;
        bumpMapGeneration();
    }
    void set_splat_entity_ids(const std::vector<int32_t>& eids) {
        splat_entity_ids = eids;
    }
    void set_splat_states_vec(const std::vector<uint32_t>& states) {
        splat_states = states;
        bumpMapGeneration();
    }
    void set_active_indices_gpu(const std::vector<uint32_t>& active_indices, size_t total_splats);
    void invalidateActiveIndexDispatch() {
        use_active_index_dispatch_ = false;
        use_ago_indirect_dispatch_ = false;
        active_splat_count_ = 0u;
    }
    void bumpMapGeneration() {
        ++map_generation_;
        invalidateActiveIndexDispatch();
    }
    uint64_t get_map_generation() const { return map_generation_; }
    uint64_t get_active_set_generation() const { return active_set_generation_; }
    uint32_t get_active_splat_count() const { return active_splat_count_; }
    bool get_use_active_index_dispatch() const { return use_active_index_dispatch_; }
    std::vector<uint32_t> get_active_indices_host();

    // AGO-0 opt-in zero-perturbation per-Gaussian telemetry (8 floats/Gaussian)
    void set_ago_telemetry_enabled(bool enabled, size_t num_splats = 0);
    bool get_ago_telemetry_enabled() const { return ago_telemetry_enabled_; }
    void reset_ago_telemetry(size_t num_splats);
    std::vector<float> get_ago_telemetry_host(size_t num_splats);
    void set_ago_telemetry_host(const float* data, size_t n_floats);
    int get_ago_last_step() const { return ago_last_step_; }
    void set_ago_last_step(int step) { ago_last_step_ = step; }

    // AGO-2 GPU-resident zero-readback active-set selection & 16-byte step-gain summary
    struct AgoActiveSetSummary {
        uint32_t active_count = 0u;
        uint32_t num_gaussians = 0u;
        uint32_t eligible_count = 0u;
        float active_fraction = 1.0f;
        float total_predicted_gain = 0.0f;
        float mean_predicted_gain = 0.0f;
        float max_predicted_gain = 0.0f;
        float gain_threshold = 0.0f;
        bool fell_back_to_full = false;
    };
    AgoActiveSetSummary select_ago_active_set_gpu(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        float gain_threshold = 0.0f,
        float newborn_warmup_steps = 2.0f,
        bool exclude_frozen = true
    );
    AgoActiveSetSummary select_ago_adaptive_active_set_gpu(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        float target_gain_ratio = 0.5f,
        float newborn_warmup_steps = 2.0f,
        bool exclude_frozen = true
    );
    AgoActiveSetSummary select_ago_exact_k_gpu(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        float target_fraction = 0.5f,
        float newborn_warmup_steps = 10.0f,
        bool exclude_frozen = true
    );

    // P2: Fully GPU-native histogram -> cutoff -> compaction in a single command batch
    // with ZERO synchronous host readbacks. Writes VkDispatchIndirectCommand into
    // `ago_cutoff_indirect_buf_` with ActiveSetState::EMPTY -> groupCountX = 0.
    struct AgoCutoffIndirectState {
        uint32_t group_count_x = 0u;
        uint32_t group_count_y = 1u;
        uint32_t group_count_z = 1u;
        uint32_t cutoff_bin = 256u;
        uint32_t boundary_quota = 0u;
        uint32_t selected_count = 0u;
        uint32_t active_set_state = static_cast<uint32_t>(vksplat::ago::ActiveSetState::EMPTY);
        uint32_t selection_mode = 0u;
        uint32_t eligible_count = 0u;
        uint32_t newborn_count = 0u;
        float total_predicted_gain = 0.0f;
        float max_predicted_gain = 0.0f;
    };
    void ensureAgoSelectionBuffers(VulkanGSPipelineBuffers& buffers);
    void select_ago_active_set_gpu_async(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        float target_fraction = 0.5f,
        float newborn_warmup_steps = 10.0f,
        bool exclude_frozen = true,
        uint32_t max_budget_splats = 0u,
        uint32_t selection_mode = 0u,
        float budget_cost_units = 0.0f
    );
    AgoCutoffIndirectState get_ago_cutoff_indirect_state();
    bool get_use_ago_indirect_dispatch() const { return use_ago_indirect_dispatch_; }

    // P11-B: GPU-resident 6-component RegionPriorityVector & bounded structural candidates
    struct AgoRegionPriorityVector {
        uint32_t region_id = 0u;
        uint32_t splat_count = 0u;
        uint32_t eligible_count = 0u;
        float u_geom = 0.0f;
        float u_app = 0.0f;
        float u_split = 0.0f;
        float u_birth = 0.0f;
        float u_merge = 0.0f;
        float u_corr = 0.0f;
        uint32_t dominant_op = 0u;  // 0=REFINE_GEOM, 1=REFINE_APP, 2=SPLIT, 3=BIRTH, 4=MERGE_PRUNE, 5=CORRECTION
        uint32_t eligibility_flags = 0u;
    };
    struct AgoStructuralCandidates {
        std::vector<uint32_t> split_candidate_indices;
        std::vector<uint32_t> prune_candidate_indices;
        uint32_t split_candidates_total = 0u;
        uint32_t prune_candidates_total = 0u;
    };
    std::vector<AgoRegionPriorityVector> compute_ago_region_priority_vectors(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        float split_scale_thresh = 0.02f,
        float split_grad_thresh = 1e-4f,
        float prune_opacity_thresh = 0.05f,
        float merge_scale_thresh = 0.002f,
        float newborn_warmup_steps = 10.0f
    );
    AgoStructuralCandidates compute_ago_structural_candidates(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        uint32_t max_split = 64u,
        uint32_t max_prune = 64u,
        float split_scale_thresh = 0.02f,
        float split_grad_thresh = 1e-4f,
        float prune_opacity_thresh = 0.05f,
        float merge_scale_thresh = 0.002f,
        float newborn_warmup_steps = 10.0f
    );
    std::pair<size_t, size_t> dispatchAgoRegionPriorityPass(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        uint32_t max_split,
        uint32_t max_prune,
        float split_scale_thresh,
        float split_grad_thresh,
        float prune_opacity_thresh,
        float merge_scale_thresh,
        float newborn_warmup_steps
    );

    // AGO-0 / AGO-L0: External strategy operation dispatch (explicitly returns REJECTED_UNSUPPORTED
    // for unsupported ops rather than silently falling back).
    vksplat::ago::CompletionStatus ago_execute_external_op(
        const TrainerConfig& config,
        VulkanGSPipelineBuffers& buffers,
        vksplat::ago::OperationType op,
        int32_t group_id = -1
    );

    // AGO-L0 / S93: Bit-exact RAM eviction & restoration of Gaussian params, Adam moments, and group states.
    uint64_t ago_offload_snapshot_to_ram(VulkanGSPipelineBuffers& buffers, bool evict_gpu = false);
    bool ago_restore_snapshot_from_ram(VulkanGSPipelineBuffers& buffers, uint64_t snapshot_id);
    bool ago_drop_ram_snapshot(uint64_t snapshot_id);
    size_t ago_ram_snapshot_bytes(uint64_t snapshot_id) const;
    size_t ago_total_ram_snapshot_bytes() const;
    size_t ago_gpu_device_allocated_bytes(const VulkanGSPipelineBuffers& buffers) const;
    bool is_ago_gpu_evicted() const { return ago_gpu_evicted_; }
    size_t ago_evicted_splats() const { return ago_evicted_splats_; }

    // S100: Per-stage Vulkan timestamp profiling & workload funnel telemetry (opt-in; default disabled).
    static constexpr uint32_t kStageQ_ProjStart   = 16u;
    static constexpr uint32_t kStageQ_ProjEnd     = 17u;
    static constexpr uint32_t kStageQ_TileStart   = 18u;
    static constexpr uint32_t kStageQ_TileEnd     = 19u;
    static constexpr uint32_t kStageQ_FwdStart    = 20u;
    static constexpr uint32_t kStageQ_FwdEnd      = 21u;
    static constexpr uint32_t kStageQ_BwdStart    = 22u;
    static constexpr uint32_t kStageQ_BwdEnd      = 23u;
    static constexpr uint32_t kStageQ_OptStart    = 24u;
    static constexpr uint32_t kStageQ_OptEnd      = 25u;
    static constexpr uint32_t kStageQ_AgoSelStart = 26u;
    static constexpr uint32_t kStageQ_AgoSelEnd   = 27u;

    struct AgoStageTimings {
        std::optional<double> projection_ms;
        std::optional<double> tile_intersect_sort_ms;
        std::optional<double> forward_raster_ms;
        std::optional<double> backward_raster_ms;
        std::optional<double> optimizer_ms;
        std::optional<double> ago_selection_ms;
    };
    struct AgoWorkloadFunnel {
        size_t total_stored_gaussians = 0u;
        size_t gpu_resident_gaussians = 0u;
        size_t projection_candidates = 0u;
        std::optional<size_t> visible_gaussians;
        size_t backward_active_gaussians = 0u;
        size_t geometry_updated_gaussians = 0u;
        size_t frozen_gaussians = 0u;
    };
    void set_ago_stage_profiling(bool enabled);
    bool get_ago_stage_profiling() const { return ago_stage_profiling_enabled_; }
    void resetAgoStepStageTimestamps();
    void collectAgoStageTimings();
    void collectAgoSelectionStageTiming();
    const AgoStageTimings& get_ago_stage_timings() const { return last_ago_stage_timings_; }
    AgoWorkloadFunnel compute_ago_workload_funnel(
        const TrainerConfig& config,
        const VulkanGSPipelineBuffers& buffers
    ) const;

    // AGO-L0: Non-blocking Vulkan GPU executor wired to trainer device/queue with clean teardown.
    vksplat::ago::NonBlockingGpuExecutor& ago_gpu_executor() { return ago_gpu_executor_; }
    vksplat::ago::SubmitResult ago_gpu_submit_action(const vksplat::ago::AGOAction& action);
    std::vector<vksplat::ago::CompletionRecord> ago_gpu_poll_completions(size_t max_count = 16u);
    void ago_gpu_shutdown() { ago_gpu_executor_.shutdown(); }

    AgoActiveSetSummary get_ago_step_gain_summary(size_t current_num_gaussians) const;

    // GPU Temporal Depth Reprojection (atomic uint32 z-buffer + view-angle cosine weight + support count)
    struct TemporalDepthReprojectResult {
        std::vector<float> depth;      // (dst_h * dst_w)
        std::vector<float> weight;     // (dst_h * dst_w)
        std::vector<uint32_t> support; // (dst_h * dst_w)
        double kernel_ms = 0.0;
        double total_ms = 0.0;
    };
    TemporalDepthReprojectResult reproject_temporal_depth_gpu(
        const float* src_depths,       // (num_views * src_h * src_w)
        size_t num_views,
        int src_h,
        int src_w,
        const float* view_params_16,   // (num_views * 16): T_s2d rows 0..2 + (inv_fx_s, inv_fy_s, cx_s, cy_s)
        int dst_h,
        int dst_w,
        float dst_fx,
        float dst_fy,
        float dst_cx,
        float dst_cy,
        float max_depth_m = 10.0f,
        float min_depth_m = 0.05f,
        float min_cos_angle = 0.0f
    );

    // D1 opt-in native differentiable physical expected-depth supervision
    void set_train_depth(size_t frame_idx, const float* depth_m, const float* weight_map, int h, int w);
    void clear_train_depths();
    void set_depth_loss_weight(float lambda_depth, float huber_delta = 0.15f);
    float get_depth_loss_weight() const { return depth_loss_weight_; }
    float get_depth_huber_delta() const { return depth_huber_delta_; }
    bool frame_has_depth_loss(size_t train_idx) const {
        return depth_loss_weight_ > 0.0f && train_idx < dataset_train.size() && dataset_train[train_idx].has_depth;
    }
    const _VulkanBuffer* get_train_depth_buffer(size_t train_idx) const {
        return frame_has_depth_loss(train_idx) ? &dataset_train[train_idx].depth_weight.deviceBuffer : nullptr;
    }

    // CSE-0: Streaming frame and optimizer-preserving splat append APIs
    void uploadFrameImage(Buffer<uint8_t>& image, const uint8_t* rgba_pixels, size_t n_bytes);
    void clear_train_images();
    void clear_val_images();
    size_t append_train_image_from_memory(
        const std::string& image_name,
        const uint8_t* rgba_pixels,
        int h,
        int w,
        float fx,
        float fy,
        float cx,
        float cy,
        const float* c2w_world_row_major
    );
    size_t append_val_image_from_memory(
        const std::string& image_name,
        const uint8_t* rgba_pixels,
        int h,
        int w,
        float fx,
        float fy,
        float cx,
        float cy,
        const float* c2w_world_row_major
    );
    void replace_train_image_from_memory(
        size_t train_idx,
        const std::string& image_name,
        const uint8_t* rgba_pixels,
        int h,
        int w,
        float fx,
        float fy,
        float cx,
        float cy,
        const float* c2w_world_row_major
    );
    size_t append_splats_preserving_optimizer(
        VulkanGSPipelineBuffers& buffers,
        const float* new_xyz_norm,
        const float* new_sh_coeffs,
        const float* new_rotations,
        const float* new_scales,
        const float* new_opacities,
        const int32_t* new_group_ids,
        size_t n_add,
        const uint32_t* new_splat_states = nullptr
    );

    void ensureGroupBuffersOnDevice(VulkanGSPipelineBuffers& buffers, size_t num_splats);
    void ensureCellCreditBuffersOnDevice(VulkanGSPipelineBuffers& buffers, size_t num_splats);
    void syncGroupStateSize(VulkanGSPipelineBuffers& buffers, const int32_t* gids, size_t count);
    void set_local_cell_credits(VulkanGSPipelineBuffers& buffers, const int32_t* cell_ids, size_t n_splats,
                                const int32_t* credits, size_t n_cells, const int32_t* eligible_override = nullptr);
    void clear_local_cell_credits();
    std::vector<int32_t> get_local_cell_credits(VulkanGSPipelineBuffers& buffers);
    bool has_local_cell_credits() const { return use_local_cell_credits_; }
    const std::vector<int32_t>& get_splat_cell_ids() const { return splat_cell_ids; }
    const std::vector<int32_t>& get_cell_eligible() const { return cell_eligible_host; }
    size_t get_adc_host_readbacks() const { return adc_host_mask_readbacks_; }
    void reset_adc_host_readbacks() { adc_host_mask_readbacks_ = 0; }
    void set_group_state(VulkanGSPipelineBuffers& buffers, int32_t group_id, uint32_t state);
    void set_group_states(VulkanGSPipelineBuffers& buffers, const int32_t* gids, const uint32_t* states, size_t count);
    void set_splat_states_sparse(VulkanGSPipelineBuffers& buffers, const uint32_t* rows, const uint32_t* states, size_t count);
    void set_splat_anchors(VulkanGSPipelineBuffers& buffers, const float* anchors_8, size_t count);
    void set_splat_anchors_sparse(VulkanGSPipelineBuffers& buffers, const uint32_t* rows, const float* anchors_8, size_t count);
    void reanchor_splats_sparse(VulkanGSPipelineBuffers& buffers, const uint32_t* rows,
                                const float* xyz, const float* scales, const float* quats,
                                const float* sh_dc, const uint32_t* states, size_t count);
    struct DecayOpacityStats {
        size_t proposed = 0;
        size_t deduped = 0;
        size_t skipped_out_of_range = 0;
        size_t skipped_protected = 0;
        size_t applied = 0;
    };

    // Sparse GPU-resident logit-space opacity decay.
    // Deduplicates `rows` on host before dispatch, filters protected states/groups when
    // `respect_protected_states=true`, validates `delta_logit >= 0` and `0 < min_opacity < 1`,
    // and returns exact proposed vs. applied counts.
    // Index-lifetime contract: `rows` are valid only within the current compaction/Morton epoch and must NOT
    // be deferred across ADC prune/split or MCMC relocation/compaction.
    DecayOpacityStats decay_opacities_sparse(VulkanGSPipelineBuffers& buffers, const uint32_t* rows, size_t count,
                                             float delta_logit, float min_opacity = 1e-4f, bool reset_moments = true,
                                             bool respect_protected_states = true);
    // GPU-resident incremental RayGuard: syncs per-splat evidence by persistent `splat_uids`,
    // uploads only the HxW float32 depth map, projects all N splats on GPU (with inline 3x3
    // depth-edge veto, 5-point footprint protection, baseline + angular parallax), applies
    // logit-space decay, and reads back only 16 bytes of exact atomic stats.
    DecayOpacityStats observe_and_apply_rayguard_gpu(
        VulkanGSPipelineBuffers& buffers,
        const float* depth_m,
        int height,
        int width,
        float fx,
        float fy,
        float cx,
        float cy,
        const float* n2c_3x4_row_major,
        const float* cam_o_norm_3,
        float dp_inv_scale,
        int current_step,
        float delta_logit = 3.5f,
        float min_opacity = 0.002f,
        bool reset_moments = true,
        bool respect_protected_states = true,
        int min_rays = 2,
        float min_baseline_m = 0.20f,
        float min_parallax_deg = 8.0f,
        float surface_protect_m = 0.12f,
        int cooldown_steps = 500
    );
    void invalidate_rayguard_contradictions_gpu(VulkanGSPipelineBuffers& buffers);
    void reset_rayguard_evidence_gpu();
    const std::vector<uint64_t>& get_splat_uids() const { return splat_uids; }
    uint64_t get_next_splat_uid() const { return next_splat_uid_; }
    void set_splat_uids_vec(std::vector<uint64_t> uids, uint64_t next_uid) {
        splat_uids = std::move(uids);
        next_splat_uid_ = std::max<uint64_t>(next_uid, 1u);
    }
    void retract_group_sparse(VulkanGSPipelineBuffers& buffers, int32_t group_id);
    void clamp_scales(VulkanGSPipelineBuffers& buffers, float max_scale, float tangent_cap);
    // Rewrites geometry (xyz, scales, quats; rows in ascending splat index of the group) and resets the geometry
    // Adam moments of one group's splats on the GPU. Throws when `count` differs from the group's splat count.
    void update_group_sparse(VulkanGSPipelineBuffers& buffers, int32_t group_id,
                             const float* xyz, const float* scales, const float* quats, size_t count);
    // Removes every splat of a RETRACTED (3) group from all per-splat buffers (one GPU gather per buffer, no
    // readback; the keep list comes from the host group mirror). Returns old index -> new index (-1 = removed).
    std::vector<int32_t> compact_retracted(VulkanGSPipelineBuffers& buffers);

    struct AppearanceInitResult {
        std::vector<float> sh_coeffs;   // (N * 16 * 3)
        std::vector<float> rgb_init;    // (N * 3)
        std::vector<int32_t> obs_count; // (N)
        std::vector<float> weight_sum;  // (N)
        double wls_sh_fit_rmse = 0.0;
        double kernel_wall_ms = 0.0;
        double total_wall_ms = 0.0;
    };

    // Native Slang/Vulkan multi-view RGB-D projection + regularized WLS degree-0/1 SH solver.
    AppearanceInitResult project_and_fit_sh_vulkan(
        const float* means_world,      // (N, 3)
        const float* normals_world,    // (N, 3) or nullptr
        size_t n_points,
        const uint8_t* rgb_views_u8,   // (V, H_rgb, W_rgb, rgb_channels)
        size_t n_views,
        int h_rgb,
        int w_rgb,
        int rgb_channels,
        const float* depth_views_f32,  // (V, H_d, W_d)
        int h_d,
        int w_d,
        const float* view_params_24,   // (V, 24): w2c_r0, w2c_r1, w2c_r2, cam_o, k_rgb, k_depth (6 float4s/view)
        const float* fallback_rgb_f32, // (N, 3)
        int sh_degree = 1,
        float reg_lambda = 0.25f,
        float depth_tol_m = 0.12f,
        float min_cos_angle = 0.15f
    );

    struct AppendAndInitSurfelsResult {
        size_t n_appended = 0;
        double kernel_ms = 0.0;
        double total_ms = 0.0;
        double mean_wls_rmse = 0.0;
        size_t n_valid_sh = 0;
    };

    // Fused GPU-resident appearance init + optimizer-preserving surfel append (no SH host readback).
    AppendAndInitSurfelsResult append_and_init_surfels_vulkan(
        VulkanGSPipelineBuffers& buffers,
        const float* means_world,      // (M, 3)
        const float* normals_world,    // (M, 3) or nullptr
        const float* xyz_norm,         // (M, 3)
        const float* scales_norm,      // (M, 3)
        const float* quats_norm,       // (M, 4)
        const float* opacities,        // (M,)
        const int32_t* group_ids,      // (M,) or nullptr
        const uint32_t* new_splat_states, // (M,) or nullptr
        const float* fallback_rgb_f32, // (M, 3)
        size_t n_add,
        const uint8_t* rgb_views_u8,   // (V, H_rgb, W_rgb, rgb_channels) or nullptr when V==0
        size_t n_views,
        int h_rgb,
        int w_rgb,
        int rgb_channels,
        const float* depth_views_f32,  // (V, H_d, W_d) or nullptr when V==0
        int h_d,
        int w_d,
        const float* view_params_24,   // (V, 24) or nullptr when V==0
        int sh_degree = 1,
        float reg_lambda = 1e-2f,
        float depth_tol_m = 0.12f,
        float min_cos_angle = 0.15f
    );

private:
    std::default_random_engine rng;

    std::vector<DatasetImage> dataset_train;
    std::vector<DatasetImage> dataset_val;
    glm::mat4 dataparser_transform = glm::mat4(1.0f);
    float scene_scale = 1.0f;

    size_t num_anchored_splats = 0;
    bool anchored_xyz_snapshot_valid = false;
    bool selective_freeze_unfrozen = false;
    float pre_freeze_grad_ema = 0.0f;
    std::vector<int32_t> splat_group_ids;
    std::vector<int32_t> splat_entity_ids;
    std::vector<uint64_t> splat_uids;       // (N,) persistent monotonic per-splat identity across sort/prune/relocate
    uint64_t next_splat_uid_ = 1u;
    std::vector<uint32_t> splat_states;     // (N,) per-splat evidence state mirror
    std::vector<float> splat_anchors_host;  // (8*N,) host mirror of per-splat geometric anchors
    bool use_local_cell_credits_ = false;
    bool splat_cell_ids_dirty_ = false;
    bool cell_credits_gpu_dirty_ = false;
    size_t adc_host_mask_readbacks_ = 0;
    std::vector<int32_t> splat_cell_ids;
    std::vector<int32_t> cell_credits_host;
    std::vector<int32_t> cell_eligible_host;
    Buffer<uint32_t> group_op_rows;  // splat indices of update_group_sparse / keep list of compact_retracted
    Buffer<float> group_op_vals;
    Buffer<uint32_t> compact_scratch;  // copy of the buffer being gathered
    Buffer<float> rayguard_ev_;          // (12 * N) GPU-resident per-splat RayGuard evidence (3 float4/splat)
    Buffer<float> rayguard_ev_scratch_;  // scratch for UID-keyed GPU gather remap
    Buffer<float> rayguard_depth_;       // (H * W) current keyframe depth map
    Buffer<uint32_t> rayguard_stats_;    // (4) GPU atomic stats [proposed, skipped_protected, applied, 0]
    std::vector<uint64_t> rayguard_uids_;
    void syncRayGuardEvidenceOnDevice(VulkanGSPipelineBuffers& buffers);

    // Reusable device buffers for appearance_init.slang
    Buffer<float> appear_means_;
    Buffer<float> appear_normals_;
    Buffer<float> appear_fallback_;
    Buffer<float> appear_view_params_;
    Buffer<uint8_t> appear_rgb_u8_;
    Buffer<float> appear_depth_f32_;
    Buffer<float> appear_out_sh_;
    Buffer<float> appear_out_rgb_;
    Buffer<int32_t> appear_out_obs_;
    Buffer<float> appear_out_wsum_;
    Buffer<float> appear_out_se_;
    _ComputePipeline pipeline_appearance_init = _ComputePipeline(11);
    double runAppearanceInitKernel(
        const float* means_world,
        const float* normals_world,
        size_t n_points,
        const uint8_t* rgb_views_u8,
        size_t n_views,
        int h_rgb,
        int w_rgb,
        int rgb_channels,
        const float* depth_views_f32,
        int h_d,
        int w_d,
        const float* view_params_24,
        const float* fallback_rgb_f32,
        int sh_degree,
        float reg_lambda,
        float depth_tol_m,
        float min_cos_angle,
        const _VulkanBuffer& target_sh_buf,
        bool write_reordered_sh,
        uint32_t sh_dst_offset
    );

    // GPU active-set compaction for Adam optimizer dispatch
    Buffer<uint32_t> active_indices_buf_;
    uint32_t active_splat_count_ = 0u;
    bool use_active_index_dispatch_ = false;
    uint64_t map_generation_ = 1u;
    uint64_t active_set_generation_ = 0u;

    // AGO-0 opt-in per-Gaussian telemetry (8 floats/Gaussian, zero overhead when disabled)
    bool ago_telemetry_enabled_ = false;
    Buffer<float> ago_telemetry_buf_;
    int ago_last_step_ = 0;

    // AGO-2 GPU-resident active-set compaction, 256-bin log-gain histogram, & summary buffers
    Buffer<uint32_t> ago_select_summary_buf_;
    Buffer<uint32_t> ago_hist_buf_;
    Buffer<uint32_t> ago_cutoff_indirect_buf_;
    Buffer<uint32_t> ago_region_accum_buf_;
    Buffer<uint32_t> ago_structural_cand_buf_;
    bool use_ago_indirect_dispatch_ = false;
    _ComputePipeline pipeline_ago_select_active_set = _ComputePipeline(6);
    _ComputePipeline pipeline_ago_gain_histogram = _ComputePipeline(5);
    _ComputePipeline pipeline_ago_histogram_cutoff = _ComputePipeline(2);
    _ComputePipeline pipeline_ago_select_active_set_indirect = _ComputePipeline(7);
    _ComputePipeline pipeline_ago_region_vector_priority = _ComputePipeline(7);
    AgoActiveSetSummary last_ago_summary_{};
    float ago_prev_mean_gain_ = 0.0f;

    // AGO-L0: Non-blocking GPU executor + device marker buffer + host RAM snapshots
    vksplat::ago::NonBlockingGpuExecutor ago_gpu_executor_;
    Buffer<uint32_t> ago_exec_marker_buf_;
    struct AgoRamSnapshot {
        size_t num_splats = 0;
        uint64_t map_generation = 1u;
        int ago_last_step = 0;
        uint64_t next_splat_uid = 1u;
        std::vector<uint8_t> xyz_ws_bytes;
        std::vector<uint8_t> sh_coeffs_bytes;
        std::vector<uint8_t> rotations_bytes;
        std::vector<uint8_t> scales_opacs_bytes;
        std::vector<uint8_t> g_xyz_ws_bytes;
        std::vector<uint8_t> g_sh_coeffs_1_bytes;
        std::vector<uint8_t> g_sh_coeffs_2_bytes;
        std::vector<uint8_t> g_rotations_bytes;
        std::vector<uint8_t> g_scales_opacs_bytes;
        std::vector<int32_t> group_ids;
        std::vector<uint32_t> group_state;
        std::vector<uint32_t> splat_state;
        std::vector<int32_t> splat_entity_ids;
        std::vector<uint64_t> splat_uids;
        std::vector<float> splat_anchors_host;
        size_t total_bytes() const noexcept {
            return xyz_ws_bytes.size() + sh_coeffs_bytes.size() + rotations_bytes.size() +
                   scales_opacs_bytes.size() + g_xyz_ws_bytes.size() + g_sh_coeffs_1_bytes.size() +
                   g_sh_coeffs_2_bytes.size() + g_rotations_bytes.size() + g_scales_opacs_bytes.size() +
                   group_ids.size() * sizeof(int32_t) + group_state.size() * sizeof(uint32_t) +
                   splat_state.size() * sizeof(uint32_t) + splat_entity_ids.size() * sizeof(int32_t) +
                   splat_uids.size() * sizeof(uint64_t) + splat_anchors_host.size() * sizeof(float);
        }
    };
    std::unordered_map<uint64_t, AgoRamSnapshot> ago_ram_snapshots_;
    uint64_t next_ago_snapshot_id_ = 1u;
    bool ago_gpu_evicted_ = false;
    size_t ago_evicted_splats_ = 0u;

    // S100: Opt-in per-stage Vulkan timestamp profiling state
    bool ago_stage_profiling_enabled_ = false;
    AgoStageTimings last_ago_stage_timings_{};

    // Reusable device buffers & pipelines for temporal_depth_reproject.slang
    Buffer<float> treproj_src_depths_;
    Buffer<float> treproj_view_params_;
    Buffer<uint32_t> treproj_dst_zbuf_;
    Buffer<float> treproj_dst_weight_;
    Buffer<uint32_t> treproj_dst_support_;
    Buffer<float> treproj_dst_depth_;
    _ComputePipeline pipeline_temporal_depth_scatter = _ComputePipeline(5);
    _ComputePipeline pipeline_temporal_depth_finalize = _ComputePipeline(3);

    // D1 opt-in differentiable physical expected-depth loss (strictly opt-in; 0.0 = disabled)
    float depth_loss_weight_ = 0.0f;
    float depth_huber_delta_ = 0.15f;

    size_t mcmc_relocated_count = 0;
    size_t mcmc_added_count = 0;
    std::atomic<size_t> runtime_growth_cap{std::numeric_limits<size_t>::max()};
    Buffer<float> anchored_xyz_ws;

    _ComputePipeline pipeline_ssim_stats = _ComputePipeline(3);
    _ComputePipeline pipeline_ssim_grad = _ComputePipeline(5);
    _ComputePipeline pipeline_ssim_stats_ds2 = _ComputePipeline(3);  // SSIM on the 2x2 box-averaged images
    _ComputePipeline pipeline_ssim_grad_ds2 = _ComputePipeline(5);
    _ComputePipeline pipeline_fused_projection_backward_optimizer = _ComputePipeline(22);
    struct _DefaultComputePipeline {
        _ComputePipeline update_state = _ComputePipeline(4);
        _ComputePipeline compute_grow_mask = _ComputePipeline(10);
        _ComputePipeline duplicate = _ComputePipeline(12);
        _ComputePipeline split = _ComputePipeline(12);
        _ComputePipeline compute_prune_mask = _ComputePipeline(6);
        _ComputePipeline prune = _ComputePipeline(4);
        _ComputePipeline prune_mean = _ComputePipeline(4);
        _ComputePipeline prune_sh = _ComputePipeline(4);
        _ComputePipeline reset_opa = _ComputePipeline(5);
    } pipeline_default;
    struct _MCMCComputePipeline {
        _ComputePipeline inject_noise = _ComputePipeline(7);
        _ComputePipeline compute_probs = _ComputePipeline(7);
        _ComputePipeline compute_relocation_index_map = _ComputePipeline(8);
        _ComputePipeline compute_relocation = _ComputePipeline(7);
        _ComputePipeline update_relocation = _ComputePipeline(10);
        _ComputePipeline compute_add_index_map = _ComputePipeline(3);
        _ComputePipeline compute_add = _ComputePipeline(2);
        _ComputePipeline update_add = _ComputePipeline(10);
        _ComputePipeline mask_retracted_group = _ComputePipeline(4);
        _ComputePipeline clamp_scales = _ComputePipeline(2);
        _ComputePipeline update_group_rows = _ComputePipeline(8);
        _ComputePipeline compact_gather = _ComputePipeline(3);
        _ComputePipeline compact_gather_sh = _ComputePipeline(3);
        _ComputePipeline decay_opacity_rows = _ComputePipeline(6);
        _ComputePipeline rayguard_observe = _ComputePipeline(7);
        _ComputePipeline rayguard_apply = _ComputePipeline(7);
        _ComputePipeline rayguard_remap = _ComputePipeline(3);
    } pipeline_mcmc;
    struct _MortonSortComputePipeline {
        _ComputePipeline compute_stats = _ComputePipeline(2);
        _ComputePipeline generate_keys = _ComputePipeline(4);
        _ComputePipeline apply_indices = _ComputePipeline(3);
        _ComputePipeline apply_indices_sh = _ComputePipeline(3);
        _ComputePipeline update_buffer = _ComputePipeline(2);
        _ComputePipeline update_buffer_sh = _ComputePipeline(2);
    } pipeline_morton_sort;

    void barrierAllGaussParams(VulkanGSPipelineBuffers& buffers);

};

