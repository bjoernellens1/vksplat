#pragma once

#include "gs_renderer.h"

#include "colmap_reader.h"

#include <random>
#include <atomic>
#include <limits>


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
    }
    void set_splat_entity_ids(const std::vector<int32_t>& eids) {
        splat_entity_ids = eids;
    }
    void set_splat_states_vec(const std::vector<uint32_t>& states) {
        splat_states = states;
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
    void retract_group_sparse(VulkanGSPipelineBuffers& buffers, int32_t group_id);
    void clamp_scales(VulkanGSPipelineBuffers& buffers, float max_scale, float tangent_cap);
    // Rewrites geometry (xyz, scales, quats; rows in ascending splat index of the group) and resets the geometry
    // Adam moments of one group's splats on the GPU. Throws when `count` differs from the group's splat count.
    void update_group_sparse(VulkanGSPipelineBuffers& buffers, int32_t group_id,
                             const float* xyz, const float* scales, const float* quats, size_t count);
    // Removes every splat of a RETRACTED (3) group from all per-splat buffers (one GPU gather per buffer, no
    // readback; the keep list comes from the host group mirror). Returns old index -> new index (-1 = removed).
    std::vector<int32_t> compact_retracted(VulkanGSPipelineBuffers& buffers);

private:
    std::default_random_engine rng;

    std::vector<DatasetImage> dataset_train;
    std::vector<DatasetImage> dataset_val;
    glm::mat4 dataparser_transform;
    float scene_scale;

    size_t num_anchored_splats = 0;
    bool anchored_xyz_snapshot_valid = false;
    bool selective_freeze_unfrozen = false;
    float pre_freeze_grad_ema = 0.0f;
    std::vector<int32_t> splat_group_ids;
    std::vector<int32_t> splat_entity_ids;
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
    size_t mcmc_relocated_count = 0;
    size_t mcmc_added_count = 0;
    std::atomic<size_t> runtime_growth_cap{std::numeric_limits<size_t>::max()};
    Buffer<float> anchored_xyz_ws;

    _ComputePipeline pipeline_ssim_stats = _ComputePipeline(3);
    _ComputePipeline pipeline_ssim_grad = _ComputePipeline(5);
    _ComputePipeline pipeline_ssim_stats_ds2 = _ComputePipeline(3);  // SSIM on the 2x2 box-averaged images
    _ComputePipeline pipeline_ssim_grad_ds2 = _ComputePipeline(5);
    _ComputePipeline pipeline_fused_projection_backward_optimizer = _ComputePipeline(17);
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

