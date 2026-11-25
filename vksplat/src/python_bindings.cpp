#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gs_renderer.h"
#include "buffer.h"
#include "gs_trainer.h"
#include "ago/core/scheduler.h"
#include "ago/runtime/gpu_executor.h"

#include <chrono>
#include <memory>

namespace py = pybind11;

class VkSplat {
public:
    // VulkanGSRenderer trainer;
    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config;
    std::unique_ptr<vksplat::ago::AgoScheduler> ago_scheduler_;
    bool isect_capacity_forced = false;  // test hook: skip the proactive capacity growth once
    size_t last_isects = 0;  // intersection count of the last gated step (sizes the next batch)
    const _VulkanBuffer* current_train_depth_buf = nullptr;  // active frame's depth+weight GPU buffer when D1 depth loss is enabled

    template<typename Ti, typename To, bool on_device=true>
    py::array_t<To> buffer_to_array(
        Buffer<Ti> &buffer,
        size_t n1=(size_t)(-1),
        size_t n2=(size_t)(-1),
        std::string reorder = "",
        size_t n1_actual=(size_t)-1,
        size_t n1_offset=(size_t)0
    ) {
        if (on_device)
            trainer.copyFromDevice(buffer);
        if (reorder == "sh")
            buffers.undoReorderSH(buffer, buffers.num_splats);

        size_t n0 = buffer.size();
        if (n1 != (size_t)(-1)) n0 /= n1;
        if (n2 != (size_t)(-1)) n0 /= n2;

        std::vector<To> buffer_float;
        auto toFloat = [&](Ti* buffer_half) -> To* {
            size_t n = buffer.size();
            buffer_float.resize(n);
            for (size_t i = 0; i < n; i++)
                buffer_float[i] = (To)halfToFloat((uint16_t)buffer_half[i]);
            return buffer_float.data();
        };

        if (n1 == (size_t)(-1) && n2 == (size_t)(-1)) {
            typedef std::array<py::ssize_t, 1> dim;
            return py::array_t<To>(
                dim({ (py::ssize_t)n0 }),
                dim({ sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? (To*)buffer.data() : toFloat(buffer.data())
            );
        }
        else if (n2 == (size_t)(-1)) {
            typedef std::array<py::ssize_t, 2> dim2;
            if (reorder == "mean" || n1 == 3 || n1 == 4) {
                n0 = std::min(n0, buffers.num_splats);
            }
            if (n1_actual == (size_t)-1)
                n1_actual = n1;
            return py::array_t<To>(
                dim2({ (py::ssize_t)n0, (py::ssize_t)n1_actual }),
                dim2({ (py::ssize_t)(n1*sizeof(To)), sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? n1_offset+(To*)buffer.data() : toFloat(n1_offset+buffer.data())
            );
        }
        else {
            typedef std::array<py::ssize_t, 3> dim3;
            if (reorder == "sh") {
                n0 = std::min(n0, buffers.num_splats);
            }
            py::array_t<To> res(
                dim3({ (py::ssize_t)n0, (py::ssize_t)n1, (py::ssize_t)n2 }),
                dim3({ (py::ssize_t)(n1*n2*sizeof(To)), (py::ssize_t)(n2*sizeof(To)), sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? (To*)buffer.data() : toFloat(buffer.data())
            );
            if (reorder == "sh") {
                buffers.reorderSH(buffer);
            }
            return res;
        }
    }

    template<typename T>
    void array_to_buffer(
        py::array_t<T> array,
        Buffer<T> &buffer
    ) {
        auto array_buf = array.request();
        buffer.assign((T*)(array_buf.ptr), (T*)(array_buf.ptr) + array_buf.size);
        trainer.copyToDevice(buffer);
    }

public:

    // Refuse to run SPIR-V older than its sources (shader_freshness.py sits next to this extension).
    static void check_shader_freshness(const std::string &spirv_dir) {
        py::module_ os_path = py::module_::import("os.path");
        std::string pkg_dir = os_path.attr("dirname")(py::module_::import("vksplat").attr("__file__")).cast<std::string>();
        std::string script = pkg_dir + "/shader_freshness.py";
        if (!os_path.attr("isfile")(script).cast<bool>()) {
            PyErr_WarnEx(PyExc_RuntimeWarning, ("shader freshness check skipped: missing " + script).c_str(), 1);
            return;
        }
        py::module_ util = py::module_::import("importlib.util");
        py::object spec = util.attr("spec_from_file_location")("vksplat_shader_freshness", script);
        py::object mod = util.attr("module_from_spec")(spec);
        spec.attr("loader").attr("exec_module")(mod);
        mod.attr("check")(spirv_dir);
    }

    void initialize(std::string spirv_dir, int device_id) {
        check_shader_freshness(spirv_dir);
        buffers.num_splats = 0;

        std::vector<std::string> spirv_paths = {
            "projection_forward",
            "generate_keys",
            "step_gate",
            "sparse_tile_residual",
            "sparse_tile_select",
            "compute_tile_ranges",
            "rasterize_forward",
            "rasterize_forward_depth",
            "rasterize_backward_0",
            "rasterize_backward_1",
            "rasterize_backward_2",
            "rasterize_backward_3",
            "rasterize_backward_4",
            "cumsum_single_pass",
            "cumsum_block_scan",
            "cumsum_scan_block_sums",
            "cumsum_add_block_offsets",
            "radix_sort/upsweep",
            "radix_sort/spine",
            "radix_sort/downsweep",
            "ssim_stats",
            "ssim_grad",
            "ssim_stats_ds2",
            "ssim_grad_ds2",
            "fused_projection_backward_optimizer",
            "sum",
            "where",
            "default_update_state",
            "default_compute_grow_mask",
            "default_duplicate",
            "default_split",
            "default_compute_prune_mask",
            "default_prune",
            "default_prune_mean",
            "default_prune_sh",
            "default_reset_opa",
            "mcmc_inject_noise",
            "mcmc_compute_probs",
            "mcmc_compute_relocation_index_map",
            "mcmc_compute_relocation",
            "mcmc_update_relocation",
            "mcmc_compute_add_index_map",
            "mcmc_compute_add",
            "mcmc_update_add",
            "mcmc_mask_retracted_group",
            "mcmc_clamp_scales",
            "mcmc_update_group_rows",
            "mcmc_compact_gather",
            "mcmc_compact_gather_sh",
            "mcmc_decay_opacity_rows",
            "mcmc_rayguard_observe",
            "mcmc_rayguard_apply",
            "mcmc_rayguard_remap",
            "morton_sort_compute_stats",
            "morton_sort_generate_keys",
            "morton_sort_apply_indices",
            "morton_sort_apply_indices_sh",
            "morton_sort_update_buffer",
            "morton_sort_update_buffer_sh",
            "appearance_init",
            "ago_select_active_set",
            "ago_gain_histogram",
            "ago_histogram_cutoff",
            "ago_select_active_set_indirect",
            "temporal_depth_reproject_scatter",
            "temporal_depth_reproject_finalize",
        };
        std::map<std::string, std::string> spirv_paths_dict;
        for (std::string name : spirv_paths) {
            spirv_paths_dict[name] = 
                (name.find('/') == std::string::npos) ?
                spirv_dir + "generated/" + name + ".spv" :
                spirv_dir + name + ".spv";
        }

        trainer.initialize(spirv_paths_dict, device_id);

        PerfTimer::reset();
    }

    py::dict set_train_config(py::dict train_config) {

        config.output_dir = train_config["output_dir"].cast<std::string>();
        config.output_ply = train_config["output_ply"].cast<std::string>();

        // dataset
        config.dataset_dir = train_config["dataset_dir"].cast<std::string>();
        config.image_dir = train_config["image_dir"].cast<std::string>();
        config.mask_dir = train_config["mask_dir"].cast<std::string>();
        config.sparse_dir = train_config["sparse_dir"].cast<std::string>();
        config.eval_interval = train_config["eval_interval"].cast<int>();

        std::string image_cache_device = train_config["image_cache_device"].cast<std::string>();
        if (image_cache_device == "cpu")
            config.image_cache_device = TrainerConfig::CacheImage::CPU;
        else if (image_cache_device == "gpu")
            config.image_cache_device = TrainerConfig::CacheImage::GPU;
        else throw std::runtime_error("Unknow image_cache_device: " + image_cache_device);

        config.global_scale = train_config["global_scale"].cast<float>();
        config.init_scale = train_config["init_scale"].cast<float>();
        config.init_opacity = train_config["init_opacity"].cast<float>();
        // Optional keys: preserve existing callers that pass pre-Sceneweave dicts.
        config.init_reference_count = train_config.contains(py::str("init_reference_count"))
            ? train_config["init_reference_count"].cast<size_t>() : size_t(0);
        config.init_added_scale_m = train_config.contains(py::str("init_added_scale_m"))
            ? train_config["init_added_scale_m"].cast<float>() : 0.0f;
        config.init_surface_mode = train_config.contains(py::str("init_surface_mode"))
            ? train_config["init_surface_mode"].cast<int>() : 0;
        config.init_normal_scale_ratio = train_config.contains(py::str("init_normal_scale_ratio"))
            ? train_config["init_normal_scale_ratio"].cast<float>() : 0.25f;
        config.freeze_geometry_after_iter = train_config.contains(py::str("freeze_geometry_after_iter"))
            ? train_config["freeze_geometry_after_iter"].cast<int>() : 0;
        config.enable_lr_boost = train_config.contains(py::str("enable_lr_boost"))
            ? train_config["enable_lr_boost"].cast<bool>() : false;
        config.max_sh_degree_override = train_config.contains(py::str("max_sh_degree_override"))
            ? train_config["max_sh_degree_override"].cast<int>() : -1;
        config.enable_pre_freeze_prune = train_config.contains(py::str("enable_pre_freeze_prune"))
            ? train_config["enable_pre_freeze_prune"].cast<bool>() : false;
        config.selective_plane_freeze = train_config.contains(py::str("selective_plane_freeze"))
            ? train_config["selective_plane_freeze"].cast<bool>() : false;
        config.unfreeze_loss_spike_ratio = train_config.contains(py::str("unfreeze_loss_spike_ratio"))
            ? train_config["unfreeze_loss_spike_ratio"].cast<float>() : 1.35f;

        std::string strategy = train_config["strategy"].cast<std::string>();
        if (strategy == "default")
            config.strategy = TrainerConfig::Strategy::Default;
        else if (strategy == "mcmc")
            config.strategy = TrainerConfig::Strategy::MCMC;
        else throw std::runtime_error("Unknow strategy: " + strategy);

        // optimizer
        config.max_steps = train_config["max_steps"].cast<int>();
        config.ssim_lambda = train_config["ssim_lambda"].cast<float>();
        config.ssim_every = train_config.contains(py::str("ssim_every")) ? train_config["ssim_every"].cast<int>() : 1;
        config.ssim_downsample = train_config.contains(py::str("ssim_downsample")) ? train_config["ssim_downsample"].cast<int>() : 1;
        if (config.ssim_every < 1)
            throw std::runtime_error("ssim_every must be >= 1");
        if (config.ssim_downsample != 1 && config.ssim_downsample != 2)
            throw std::runtime_error("ssim_downsample must be 1 or 2");
        config.means_lr = train_config["means_lr"].cast<float>();
        config.means_lr_final = train_config["means_lr_final"].cast<float>();
        config.features_dc_lr = train_config["features_dc_lr"].cast<float>();
        config.features_rest_lr = train_config["features_rest_lr"].cast<float>();
        config.opacities_lr = train_config["opacities_lr"].cast<float>();
        config.scales_lr = train_config["scales_lr"].cast<float>();
        config.quats_lr = train_config["quats_lr"].cast<float>();
        config.scale_reg = train_config["scale_reg"].cast<float>();
        config.opacity_reg = train_config["opacity_reg"].cast<float>();
        config.feature_lr_final_scale = train_config.contains(py::str("feature_lr_final_scale"))
            ? train_config["feature_lr_final_scale"].cast<float>() : 1.0f;
        config.opacity_lr_final_scale = train_config.contains(py::str("opacity_lr_final_scale"))
            ? train_config["opacity_lr_final_scale"].cast<float>() : 1.0f;
        config.scale_lr_final_scale = train_config.contains(py::str("scale_lr_final_scale"))
            ? train_config["scale_lr_final_scale"].cast<float>() : 1.0f;
        config.structural_reset_opa = train_config.contains(py::str("structural_reset_opa"))
            ? train_config["structural_reset_opa"].cast<float>() : 0.25f;

        // strategy
        config.refine_start_iter = train_config["refine_start_iter"].cast<int>();
        config.refine_stop_iter = train_config["refine_stop_iter"].cast<int>();
        config.refine_every = train_config["refine_every"].cast<int>();

        // default strategy
        config.prune_opa = train_config["prune_opa"].cast<float>();
        config.grow_grad2d = train_config["grow_grad2d"].cast<float>();
        config.grow_scale3d = train_config["grow_scale3d"].cast<float>();
        config.grow_scale2d = train_config["grow_scale2d"].cast<float>();
        config.prune_scale3d = train_config["prune_scale3d"].cast<float>();
        config.prune_scale2d = train_config["prune_scale2d"].cast<float>();
        config.refine_scale2d_stop_iter = train_config["refine_scale2d_stop_iter"].cast<int>();
        config.reset_every = train_config["reset_every"].cast<int>();
        config.stop_reset_at = train_config["stop_reset_at"].cast<int>();
        config.pause_refine_after_reset = train_config["pause_refine_after_reset"].cast<int>();

        // MCMC & capacity strategy
        config.noise_lr = train_config["noise_lr"].cast<float>();
        config.min_opacity = train_config["min_opacity"].cast<float>();
        config.grow_factor = train_config["grow_factor"].cast<float>();
        config.cap_max = train_config["cap_max"].cast<int>();
        config.cap_alloc_max = std::max(config.cap_alloc_max, config.cap_max);
        config.cap_speculative_max = train_config.contains(py::str("cap_speculative_max"))
            ? train_config["cap_speculative_max"].cast<int>() : 0;
        config.surfel_normal_cap_m = train_config.contains(py::str("surfel_normal_cap_m"))
            ? train_config["surfel_normal_cap_m"].cast<float>() : 0.03f;
        config.mcmc_protect_structural_groups = train_config.contains(py::str("mcmc_protect_structural_groups"))
            ? train_config["mcmc_protect_structural_groups"].cast<bool>() : false;
        config.freeze_skip_backward = train_config.contains(py::str("freeze_skip_backward"))
            ? train_config["freeze_skip_backward"].cast<bool>() : true;
        set_sparse_backward(
            train_config.contains(py::str("sparse_tile_quantile")) ? train_config["sparse_tile_quantile"].cast<float>() : 0.0f,
            train_config.contains(py::str("sparse_tile_halo")) ? train_config["sparse_tile_halo"].cast<int>() : 1,
            train_config.contains(py::str("sparse_full_every")) ? train_config["sparse_full_every"].cast<int>() : 4);
        config.rasterize_backward_pin_after = train_config.contains(py::str("rasterize_backward_pin_after"))
            ? train_config["rasterize_backward_pin_after"].cast<int>() : 20;
        trainer.setBackwardPinAfter((size_t)std::max(config.rasterize_backward_pin_after, 0));
        config.external_strategy_mode = train_config.contains(py::str("external_strategy_mode"))
            ? train_config["external_strategy_mode"].cast<bool>() : false;
        config.ago_param_mask = train_config.contains(py::str("ago_param_mask"))
            ? train_config["ago_param_mask"].cast<uint32_t>() : TrainerConfig::AGO_PARAM_ALL;

        // load dataset
        trainer.load_colmap_dataset(config, buffers);
        trainer.get_train_camera(0, uniforms);
        uniforms.active_sh = 3;

        // return metadata
        py::dict train_meta;
        glm::mat4 dataparser_transform = trainer.get_dataparser_transform();
        train_meta["dataparser_transform"] = py::array_t<float>(
            {4, 4},
            {sizeof(float), 4 * sizeof(float)},   // column-major
            reinterpret_cast<float*>(&dataparser_transform)
        );
        train_meta["device"] = trainer.get_device_info();
        train_meta["num_anchored_splats"] = trainer.get_num_anchored_splats();
        const auto& gids = trainer.get_splat_group_ids();
        train_meta["splat_group_ids"] = py::array_t<int32_t>(
            {gids.size()},
            {sizeof(int32_t)},
            gids.data()
        );
        return train_meta;
    }

    py::array_t<int32_t> get_splat_group_ids() {
        const auto& gids = trainer.get_splat_group_ids();
        return py::array_t<int32_t>({gids.size()}, {sizeof(int32_t)}, gids.data());
    }
    // Device-resident structural group state (test/diagnostic readback, one-off copy)
    py::array_t<int32_t> get_device_group_ids() {
        return buffer_to_array<int32_t, int32_t>(buffers.group_ids);
    }
    py::array_t<uint32_t> get_device_group_state() {
        return buffer_to_array<uint32_t, uint32_t>(buffers.group_state);
    }
    // Allocated capacity in elements of the per-splat parameter buffers (logical count is num_splats)
    std::map<std::string, size_t> get_splat_capacity() {
        return {
            {"xyz_ws", buffers.xyz_ws.deviceBuffer.allocSize / (3 * sizeof(float))},
            {"rotations", buffers.rotations.deviceBuffer.allocSize / (4 * sizeof(float))},
            {"group_ids", buffers.group_ids.deviceBuffer.allocSize / sizeof(int32_t)},
            {"g_rotations", buffers.g_rotations.deviceBuffer.allocSize / (8 * sizeof(float))},
            {"v_xy_vs", buffers.v_xy_vs.deviceBuffer.allocSize / (2 * sizeof(float))},
            {"tiles_touched", buffers.tiles_touched.deviceBuffer.allocSize / sizeof(int32_t)},
        };
    }
    py::array_t<int32_t> get_splat_entity_ids() {
        const auto& eids = trainer.get_splat_entity_ids();
        return py::array_t<int32_t>({eids.size()}, {sizeof(int32_t)}, eids.data());
    }
    py::dict get_mcmc_stats() {
        py::dict st;
        st["relocated_count"] = trainer.get_mcmc_relocated_count();
        st["added_count"] = trainer.get_mcmc_added_count();
        st["num_splats"] = buffers.num_splats;
        st["cap_max"] = config.cap_max;
        st["cap_alloc_max"] = config.cap_alloc_max;
        st["growth_cap"] = trainer.get_growth_cap();
        st["grow_factor"] = config.grow_factor;
        st["refine_every"] = config.refine_every;
        st["mcmc_protect_structural_groups"] = config.mcmc_protect_structural_groups;
        st["imagined_protection"] = true;
        return st;
    }
    void set_growth_cap(size_t cap) {
        trainer.set_growth_cap(cap);
    }
    void set_densification_budget(
        int cap_max,
        int cap_speculative_max = -1,
        float grow_factor = -1.0f,
        int refine_every = -1
    ) {
        if (cap_max >= 0) {
            config.cap_max = cap_max;
            config.cap_alloc_max = std::max(config.cap_alloc_max, cap_max);
        }
        if (cap_speculative_max >= 0)
            config.cap_speculative_max = cap_speculative_max;
        if (grow_factor > 0.0f)
            config.grow_factor = grow_factor;
        if (refine_every >= 1)
            config.refine_every = refine_every;
    }
    py::dict get_densification_budget() const {
        py::dict d;
        d["cap_max"] = config.cap_max;
        d["cap_alloc_max"] = config.cap_alloc_max;
        d["cap_speculative_max"] = config.cap_speculative_max;
        d["grow_factor"] = config.grow_factor;
        d["refine_every"] = config.refine_every;
        return d;
    }
    void set_stabilization_schedule(
        float structural_reset_opa = 0.25f,
        float lr_final_scale = 1.0f,
        int max_steps = -1
    ) {
        if (structural_reset_opa >= 0.0f)
            config.structural_reset_opa = std::clamp(structural_reset_opa, 0.0f, 1.0f);
        if (lr_final_scale > 0.0f) {
            config.feature_lr_final_scale = lr_final_scale;
            config.opacity_lr_final_scale = lr_final_scale;
            config.scale_lr_final_scale = lr_final_scale;
        }
        if (max_steps > 0)
            config.max_steps = max_steps;
    }
    py::dict get_stabilization_schedule() const {
        py::dict d;
        d["structural_reset_opa"] = config.structural_reset_opa;
        d["feature_lr_final_scale"] = config.feature_lr_final_scale;
        d["opacity_lr_final_scale"] = config.opacity_lr_final_scale;
        d["scale_lr_final_scale"] = config.scale_lr_final_scale;
        d["max_steps"] = config.max_steps;
        return d;
    }
    void set_selective_freeze_unfrozen(bool unfrozen) {
        trainer.set_selective_freeze_unfrozen(unfrozen);
    }
    void set_sparse_backward(float quantile, int halo, int full_every) {
        if (quantile < 0.0f || quantile >= 1.0f)
            throw std::runtime_error("sparse_tile_quantile must be in [0, 1)");
        if (halo < 0 || full_every < 1)
            throw std::runtime_error("sparse_tile_halo must be >= 0 and sparse_full_every >= 1");
        config.sparse_tile_quantile = quantile;
        config.sparse_tile_halo = halo;
        config.sparse_full_every = full_every;
    }

    void reset_anchored_snapshot(size_t new_num_anchored) {
        trainer.reset_anchored_snapshot(new_num_anchored);
    }
    void set_splat_group_ids(py::array_t<int32_t> gids_arr) {
        auto buf = gids_arr.request();
        std::vector<int32_t> vec((int32_t*)buf.ptr, (int32_t*)buf.ptr + buf.size);
        trainer.set_splat_group_ids(vec);
        buffers.group_ids.assign(vec.begin(), vec.end());
        if (!buffers.group_ids.empty()) {
            trainer.copyToDevice(buffers.group_ids);
        }
        trainer.syncGroupStateSize(buffers, vec.data(), vec.size());
    }
    void set_group_state(int32_t group_id, uint32_t state) {
        trainer.set_group_state(buffers, group_id, state);
    }
    void set_group_states(py::array_t<int32_t, py::array::c_style | py::array::forcecast> gids,
                          py::array_t<uint32_t, py::array::c_style | py::array::forcecast> states) {
        if (gids.size() != states.size())
            throw std::invalid_argument("group_ids and states must have the same length");
        trainer.set_group_states(buffers, gids.data(), states.data(), static_cast<size_t>(gids.size()));
    }
    py::array_t<uint32_t> get_splat_states() {
        const auto& sts = trainer.get_splat_states();
        if (sts.empty()) {
            std::vector<uint32_t> def_sts(buffers.num_splats, 0xFFFFFFFFu);
            return py::array_t<uint32_t>({def_sts.size()}, {sizeof(uint32_t)}, def_sts.data());
        }
        size_t n = std::min(buffers.num_splats, sts.size());
        std::vector<uint32_t> out(buffers.num_splats, 0xFFFFFFFFu);
        std::memcpy(out.data(), sts.data(), n * sizeof(uint32_t));
        return py::array_t<uint32_t>({out.size()}, {sizeof(uint32_t)}, out.data());
    }
    void set_splat_states(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> states_arr) {
        auto buf = states_arr.request();
        std::vector<uint32_t> vec((uint32_t*)buf.ptr, (uint32_t*)buf.ptr + buf.size);
        trainer.set_splat_states_vec(vec);
        buffers.splat_state.assign(vec.begin(), vec.end());
        if (!buffers.splat_state.empty()) {
            trainer.copyToDevice(buffers.splat_state);
        }
    }
    void set_splat_states_sparse(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> rows,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> states
    ) {
        if (rows.size() != states.size())
            throw std::invalid_argument("rows and states must have the same length");
        trainer.set_splat_states_sparse(buffers, rows.data(), states.data(), static_cast<size_t>(rows.size()));
    }
    void set_splat_surfel_2d_mask(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_arr) {
        constexpr uint32_t kSurfel2DFlag = 0x00010000u;
        constexpr uint32_t kLegacyState = 0xFFFFFFFFu;
        auto mbuf = mask_arr.request();
        size_t n = buffers.num_splats;
        if (static_cast<size_t>(mbuf.size) != n)
            throw std::invalid_argument("mask length must equal num_splats");
        const uint8_t* mptr = static_cast<const uint8_t*>(mbuf.ptr);
        if (buffers.splat_state.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.splat_state.size() >= n) {
            trainer.copyFromDevice(buffers.splat_state);
        }
        std::vector<uint32_t> sts(n, 5u);
        for (size_t i = 0; i < n; ++i) {
            uint32_t cur = (i < buffers.splat_state.size()) ? buffers.splat_state[i] : 5u;
            if (cur == kLegacyState) cur = 5u;
            uint32_t base = cur & 0xFFFFu;
            sts[i] = (mptr[i] != 0u) ? (base | kSurfel2DFlag) : base;
        }
        trainer.set_splat_states_vec(sts);
        buffers.splat_state.assign(sts.begin(), sts.end());
        if (!buffers.splat_state.empty()) {
            trainer.copyToDevice(buffers.splat_state);
        }
        trainer.copyFromDevice(buffers.scales_opacs);
        for (size_t i = 0; i < n; ++i) {
            if (mptr[i] != 0u && (4 * i + 2) < buffers.scales_opacs.size()) {
                buffers.scales_opacs[4 * i + 2] = 0.0f;
            }
        }
        trainer.copyToDevice(buffers.scales_opacs);
    }
    void promote_surfels_to_3d(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> rows,
        py::array_t<float, py::array::c_style | py::array::forcecast> sigma_n
    ) {
        constexpr uint32_t kSurfel2DFlag = 0x00010000u;
        constexpr uint32_t kPromoted3DFlag = 0x00020000u;
        constexpr uint32_t kLegacyState = 0xFFFFFFFFu;
        if (rows.size() != sigma_n.size())
            throw std::invalid_argument("rows and sigma_n must have the same length");
        size_t n = buffers.num_splats;
        size_t m = static_cast<size_t>(rows.size());
        if (m == 0) return;
        const uint32_t* rptr = rows.data();
        const float* sptr = sigma_n.data();
        if (buffers.splat_state.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.splat_state.size() >= n) {
            trainer.copyFromDevice(buffers.splat_state);
        }
        trainer.copyFromDevice(buffers.scales_opacs);
        std::vector<uint32_t> sts(buffers.splat_state.begin(), buffers.splat_state.end());
        if (sts.size() < n) sts.resize(n, 5u);
        for (size_t k = 0; k < m; ++k) {
            uint32_t idx = rptr[k];
            if (idx >= n) continue;
            if (sts[idx] != kLegacyState) {
                sts[idx] = (sts[idx] & (~kSurfel2DFlag)) | kPromoted3DFlag;
            }
            if ((4 * idx + 2) < buffers.scales_opacs.size()) {
                buffers.scales_opacs[4 * idx + 2] = std::max(sptr[k], 1e-4f);
            }
        }
        trainer.set_splat_states_vec(sts);
        buffers.splat_state.assign(sts.begin(), sts.end());
        trainer.copyToDevice(buffers.splat_state);
        trainer.copyToDevice(buffers.scales_opacs);
    }
    void set_splat_anchors(py::array_t<float, py::array::c_style | py::array::forcecast> anchors) {
        auto buf = anchors.request();
        if (buf.size == 0) {
            trainer.set_splat_anchors(buffers, nullptr, 0);
            return;
        }
        if ((buf.ndim == 2 && buf.shape[1] != 8) || (buf.size % 8 != 0))
            throw std::invalid_argument("anchors must have shape (N, 8)");
        trainer.set_splat_anchors(buffers, static_cast<const float*>(buf.ptr), static_cast<size_t>(buf.size / 8));
    }
    void set_splat_anchors_sparse(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> rows,
        py::array_t<float, py::array::c_style | py::array::forcecast> anchors
    ) {
        auto abuf = anchors.request();
        size_t n = static_cast<size_t>(rows.size());
        if ((size_t)abuf.size != 8 * n)
            throw std::invalid_argument("anchors must have shape (len(rows), 8)");
        trainer.set_splat_anchors_sparse(buffers, rows.data(), static_cast<const float*>(abuf.ptr), n);
    }
    py::array_t<float> get_splat_anchors() {
        if (buffers.splat_anchor.deviceBuffer.buffer == VK_NULL_HANDLE || buffers.splat_anchor.deviceSize() == 0)
            return py::array_t<float>(std::array<py::ssize_t, 2>{0, 8});
        return buffer_to_array<float, float>(buffers.splat_anchor, 8, (size_t)-1, "mean");
    }
    void reanchor_splats_sparse(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> rows,
        py::array_t<float, py::array::c_style | py::array::forcecast> xyz,
        py::array_t<float, py::array::c_style | py::array::forcecast> scales,
        py::array_t<float, py::array::c_style | py::array::forcecast> quats,
        py::array_t<float, py::array::c_style | py::array::forcecast> sh_dc,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> states
    ) {
        size_t n = static_cast<size_t>(rows.size());
        if (n == 0)
            return;
        if ((size_t)xyz.size() != 3 * n || (size_t)scales.size() != 3 * n || (size_t)quats.size() != 4 * n)
            throw std::invalid_argument("reanchor_splats_sparse expects xyz(R,3), scales(R,3), quats(R,4)");
        const float* dc_ptr = (sh_dc.size() == (py::ssize_t)(3 * n)) ? sh_dc.data() : nullptr;
        const uint32_t* st_ptr = (states.size() == (py::ssize_t)n) ? states.data() : nullptr;
        trainer.reanchor_splats_sparse(buffers, rows.data(), xyz.data(), scales.data(), quats.data(), dc_ptr, st_ptr, n);
    }
    py::dict decay_opacities_sparse(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> rows,
        float delta_logit,
        float min_opacity = 1e-4f,
        bool reset_moments = true,
        bool respect_protected_states = true
    ) {
        size_t n = static_cast<size_t>(rows.size());
        auto st = trainer.decay_opacities_sparse(buffers, rows.data(), n, delta_logit, min_opacity, reset_moments, respect_protected_states);
        py::dict d;
        d["proposed"] = st.proposed;
        d["deduped"] = st.deduped;
        d["skipped_out_of_range"] = st.skipped_out_of_range;
        d["skipped_protected"] = st.skipped_protected;
        d["applied"] = st.applied;
        return d;
    }
    py::array_t<uint64_t> get_splat_uids() {
        const auto& uids = trainer.get_splat_uids();
        return py::array_t<uint64_t>({uids.size()}, {sizeof(uint64_t)}, uids.data());
    }
    py::dict observe_and_apply_rayguard_gpu(
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_m,
        float fx,
        float fy,
        float cx,
        float cy,
        py::array_t<float, py::array::c_style | py::array::forcecast> n2c_3x4,
        py::array_t<float, py::array::c_style | py::array::forcecast> cam_o_norm,
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
    ) {
        auto dbuf = depth_m.request();
        if (dbuf.ndim != 2)
            throw std::invalid_argument("observe_and_apply_rayguard_gpu: depth_m must be 2D (H, W)");
        if (n2c_3x4.size() != 12)
            throw std::invalid_argument("observe_and_apply_rayguard_gpu: n2c_3x4 must have 12 floats");
        if (cam_o_norm.size() != 3)
            throw std::invalid_argument("observe_and_apply_rayguard_gpu: cam_o_norm must have 3 floats");
        auto st = trainer.observe_and_apply_rayguard_gpu(
            buffers,
            static_cast<const float*>(dbuf.ptr),
            static_cast<int>(dbuf.shape[0]),
            static_cast<int>(dbuf.shape[1]),
            fx, fy, cx, cy,
            n2c_3x4.data(),
            cam_o_norm.data(),
            dp_inv_scale,
            current_step,
            delta_logit,
            min_opacity,
            reset_moments,
            respect_protected_states,
            min_rays,
            min_baseline_m,
            min_parallax_deg,
            surface_protect_m,
            cooldown_steps
        );
        py::dict d;
        d["proposed"] = st.proposed;
        d["deduped"] = st.deduped;
        d["skipped_out_of_range"] = st.skipped_out_of_range;
        d["skipped_protected"] = st.skipped_protected;
        d["applied"] = st.applied;
        return d;
    }
    void invalidate_rayguard_contradictions_gpu() {
        trainer.invalidate_rayguard_contradictions_gpu(buffers);
    }
    void reset_rayguard_evidence_gpu() {
        trainer.reset_rayguard_evidence_gpu();
    }
    void set_freeze_skip_backward(bool enabled) { config.freeze_skip_backward = enabled; }
    void set_async_frame_upload(bool enabled) { trainer.waitUploads(); trainer.async_frame_upload = enabled; }
    void force_backward_variant(int impl) { trainer.forceBackwardImpl(impl); }
    std::vector<int> backward_variants() { return trainer.backwardImplCandidates(); }
    void clamp_scales(float max_scale, float tangent_cap) {
        trainer.clamp_scales(buffers, max_scale, tangent_cap);
    }
    void update_group_sparse(int32_t group_id,
                             py::array_t<float, py::array::c_style | py::array::forcecast> xyz,
                             py::array_t<float, py::array::c_style | py::array::forcecast> scales,
                             py::array_t<float, py::array::c_style | py::array::forcecast> quats) {
        const size_t n = static_cast<size_t>(xyz.size()) / 3;
        if (xyz.size() != (py::ssize_t)(3 * n) || scales.size() != (py::ssize_t)(3 * n) || quats.size() != (py::ssize_t)(4 * n))
            throw std::invalid_argument("update_group_sparse expects xyz (n,3), scales (n,3), quats (n,4)");
        trainer.update_group_sparse(buffers, group_id, xyz.data(), scales.data(), quats.data(), n);
    }
    py::array_t<int32_t> compact_retracted() {
        std::vector<int32_t> index_map = trainer.compact_retracted(buffers);
        return py::array_t<int32_t>({index_map.size()}, {sizeof(int32_t)}, index_map.data());
    }
    void retract_group_sparse(int32_t group_id) {
        trainer.retract_group_sparse(buffers, group_id);
    }

    void set_sh_coeffs_reordered(
        py::array_t<float, py::array::c_style | py::array::forcecast> sh_logical
    ) {
        auto sh_buf = sh_logical.request();
        if (sh_buf.ndim != 3 || (size_t)sh_buf.shape[0] != buffers.num_splats ||
            sh_buf.shape[1] != 16 || sh_buf.shape[2] != 3) {
            throw std::runtime_error("sh_coeffs must be (N, 16, 3) matching num_splats");
        }
        if (buffers.num_splats == 0)
            return;
        const float* ptr = static_cast<const float*>(sh_buf.ptr);
        buffers.sh_coeffs.assign(ptr, ptr + sh_buf.size);
        buffers.reorderSH(buffers.sh_coeffs);
        trainer.copyToDevice(buffers.sh_coeffs);
    }

    py::array_t<float> get_scales_opacities_packed() {
        return buffer_to_array<float, float>(buffers.scales_opacs, 4);
    }

    py::array_t<float> get_sh_dc() {
        size_t n = buffers.num_splats;
        py::array_t<float> out({(py::ssize_t)n, (py::ssize_t)3});
        if (n == 0)
            return out;
        float* dst = static_cast<float*>(out.request().ptr);
        static constexpr size_t kShDim = 12;
        if (SH_REORDER_SIZE <= 1) {
            for (size_t i = 0; i < n; ++i) {
                float rgb[3];
                trainer.downloadDeviceSlice(
                    buffers.sh_coeffs.deviceBuffer,
                    i * kShDim * 4 * sizeof(float),
                    rgb,
                    3 * sizeof(float)
                );
                dst[3 * i + 0] = rgb[0];
                dst[3 * i + 1] = rgb[1];
                dst[3 * i + 2] = rgb[2];
            }
            return out;
        }
        size_t num_blocks = _CEIL_DIV(n, SH_REORDER_SIZE);
        std::vector<float> block_dc(SH_REORDER_SIZE * 4);
        for (size_t g = 0; g < num_blocks; ++g) {
            size_t base = g * SH_REORDER_SIZE;
            size_t count = std::min<size_t>(SH_REORDER_SIZE, n - base);
            trainer.downloadDeviceSlice(
                buffers.sh_coeffs.deviceBuffer,
                g * kShDim * SH_REORDER_SIZE * 4 * sizeof(float),
                block_dc.data(),
                SH_REORDER_SIZE * 4 * sizeof(float)
            );
            for (size_t k = 0; k < count; ++k) {
                dst[3 * (base + k) + 0] = block_dc[4 * k + 0];
                dst[3 * (base + k) + 1] = block_dc[4 * k + 1];
                dst[3 * (base + k) + 2] = block_dc[4 * k + 2];
            }
        }
        return out;
    }
    void set_splat_entity_ids(py::array_t<int32_t> eids_arr) {
        auto buf = eids_arr.request();
        std::vector<int32_t> vec((int32_t*)buf.ptr, (int32_t*)buf.ptr + buf.size);
        trainer.set_splat_entity_ids(vec);
    }

    void clear_train_frames() {
        trainer.clear_train_images();
    }

    void clear_val_frames() {
        trainer.clear_val_images();
    }

    size_t append_train_frame(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_rgba,
        py::array_t<float, py::array::c_style | py::array::forcecast> c2w_world,
        float fx,
        float fy,
        float cx,
        float cy,
        std::string image_name = "stream_frame.png"
    ) {
        auto img_buf = image_rgba.request();
        if (img_buf.ndim != 3 || (img_buf.shape[2] != 4 && img_buf.shape[2] != 3))
            throw std::runtime_error("image_rgba must be (H, W, 4) or (H, W, 3) uint8 array");
        int h = (int)img_buf.shape[0];
        int w = (int)img_buf.shape[1];
        std::vector<uint8_t> rgba_vec;
        const uint8_t* rgba_ptr = static_cast<const uint8_t*>(img_buf.ptr);
        if (img_buf.shape[2] == 3) {
            rgba_vec.resize((size_t)h * (size_t)w * 4, 255u);
            for (size_t i = 0; i < (size_t)h * (size_t)w; ++i) {
                rgba_vec[4 * i + 0] = rgba_ptr[3 * i + 0];
                rgba_vec[4 * i + 1] = rgba_ptr[3 * i + 1];
                rgba_vec[4 * i + 2] = rgba_ptr[3 * i + 2];
            }
            rgba_ptr = rgba_vec.data();
        }
        auto pose_buf = c2w_world.request();
        if (pose_buf.ndim != 2 || pose_buf.shape[0] != 4 || pose_buf.shape[1] != 4)
            throw std::runtime_error("c2w_world must be (4, 4) float32 array");
        py::gil_scoped_release release;  // the staging memcpy runs without the GIL
        return trainer.append_train_image_from_memory(
            image_name, rgba_ptr, h, w, fx, fy, cx, cy, static_cast<const float*>(pose_buf.ptr)
        );
    }

    size_t append_val_frame(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_rgba,
        py::array_t<float, py::array::c_style | py::array::forcecast> c2w_world,
        float fx,
        float fy,
        float cx,
        float cy,
        std::string image_name = "val_frame.png"
    ) {
        auto img_buf = image_rgba.request();
        if (img_buf.ndim != 3 || (img_buf.shape[2] != 4 && img_buf.shape[2] != 3))
            throw std::runtime_error("image_rgba must be (H, W, 4) or (H, W, 3) uint8 array");
        int h = (int)img_buf.shape[0];
        int w = (int)img_buf.shape[1];
        std::vector<uint8_t> rgba_vec;
        const uint8_t* rgba_ptr = static_cast<const uint8_t*>(img_buf.ptr);
        if (img_buf.shape[2] == 3) {
            rgba_vec.resize((size_t)h * (size_t)w * 4, 255u);
            for (size_t i = 0; i < (size_t)h * (size_t)w; ++i) {
                rgba_vec[4 * i + 0] = rgba_ptr[3 * i + 0];
                rgba_vec[4 * i + 1] = rgba_ptr[3 * i + 1];
                rgba_vec[4 * i + 2] = rgba_ptr[3 * i + 2];
            }
            rgba_ptr = rgba_vec.data();
        }
        auto pose_buf = c2w_world.request();
        if (pose_buf.ndim != 2 || pose_buf.shape[0] != 4 || pose_buf.shape[1] != 4)
            throw std::runtime_error("c2w_world must be (4, 4) float32 array");
        return trainer.append_val_image_from_memory(
            image_name, rgba_ptr, h, w, fx, fy, cx, cy, static_cast<const float*>(pose_buf.ptr)
        );
    }

    void replace_train_frame(
        size_t train_idx,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_rgba,
        py::array_t<float, py::array::c_style | py::array::forcecast> c2w_world,
        float fx,
        float fy,
        float cx,
        float cy,
        std::string image_name = "stream_frame.png"
    ) {
        auto img_buf = image_rgba.request();
        if (img_buf.ndim != 3 || (img_buf.shape[2] != 4 && img_buf.shape[2] != 3))
            throw std::runtime_error("image_rgba must be (H, W, 4) or (H, W, 3) uint8 array");
        int h = (int)img_buf.shape[0];
        int w = (int)img_buf.shape[1];
        std::vector<uint8_t> rgba_vec;
        const uint8_t* rgba_ptr = static_cast<const uint8_t*>(img_buf.ptr);
        if (img_buf.shape[2] == 3) {
            rgba_vec.resize((size_t)h * (size_t)w * 4, 255u);
            for (size_t i = 0; i < (size_t)h * (size_t)w; ++i) {
                rgba_vec[4 * i + 0] = rgba_ptr[3 * i + 0];
                rgba_vec[4 * i + 1] = rgba_ptr[3 * i + 1];
                rgba_vec[4 * i + 2] = rgba_ptr[3 * i + 2];
            }
            rgba_ptr = rgba_vec.data();
        }
        auto pose_buf = c2w_world.request();
        if (pose_buf.ndim != 2 || pose_buf.shape[0] != 4 || pose_buf.shape[1] != 4)
            throw std::runtime_error("c2w_world must be (4, 4) float32 array");
        py::gil_scoped_release release;  // the staging memcpy runs without the GIL
        trainer.replace_train_image_from_memory(
            train_idx, image_name, rgba_ptr, h, w, fx, fy, cx, cy, static_cast<const float*>(pose_buf.ptr)
        );
    }

    size_t append_splats_preserving_optimizer(
        py::array_t<float, py::array::c_style | py::array::forcecast> xyz_ws,
        py::array_t<float, py::array::c_style | py::array::forcecast> sh_coeffs,
        py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
        py::array_t<float, py::array::c_style | py::array::forcecast> scales,
        py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> group_ids,
        py::object splat_states_obj = py::none()
    ) {
        auto xyz_buf = xyz_ws.request();
        auto sh_buf = sh_coeffs.request();
        auto rot_buf = rotations.request();
        auto scale_buf = scales.request();
        auto opac_buf = opacities.request();
        auto gid_buf = group_ids.request();
        size_t n_add = (size_t)xyz_buf.shape[0];
        if (n_add == 0)
            return buffers.num_splats;
        if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3)
            throw std::runtime_error("xyz_ws must be (M, 3)");
        if (sh_buf.ndim != 3 || (size_t)sh_buf.shape[0] != n_add || sh_buf.shape[1] != 16 || sh_buf.shape[2] != 3)
            throw std::runtime_error("sh_coeffs must be (M, 16, 3)");
        if (rot_buf.ndim != 2 || (size_t)rot_buf.shape[0] != n_add || rot_buf.shape[1] != 4)
            throw std::runtime_error("rotations must be (M, 4)");
        if (scale_buf.ndim != 2 || (size_t)scale_buf.shape[0] != n_add || scale_buf.shape[1] != 3)
            throw std::runtime_error("scales must be (M, 3)");
        if ((size_t)opac_buf.size != n_add)
            throw std::runtime_error("opacities must have M elements");
        const int32_t* gid_ptr = (gid_buf.size == (py::ssize_t)n_add) ? static_cast<const int32_t*>(gid_buf.ptr) : nullptr;
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> st_arr;
        const uint32_t* st_ptr = nullptr;
        if (!splat_states_obj.is_none()) {
            st_arr = splat_states_obj.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
            if ((size_t)st_arr.size() == n_add)
                st_ptr = st_arr.data();
        }
        return trainer.append_splats_preserving_optimizer(
            buffers,
            static_cast<const float*>(xyz_buf.ptr),
            static_cast<const float*>(sh_buf.ptr),
            static_cast<const float*>(rot_buf.ptr),
            static_cast<const float*>(scale_buf.ptr),
            static_cast<const float*>(opac_buf.ptr),
            gid_ptr,
            n_add,
            st_ptr
        );
    }

    // Device readback of the Adam moments and densification statistics (test/diagnostic)
    py::dict get_optimizer_moments() {
        py::dict d;
        d["g_xyz_ws"] = buffer_to_array<float, float>(buffers.g_xyz_ws);
        d["g_sh_coeffs_1"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_1);
        d["g_sh_coeffs_2"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_2);
        d["g_rotations"] = buffer_to_array<float, float>(buffers.g_rotations);
        d["g_scales_opacs"] = buffer_to_array<float, float>(buffers.g_scales_opacs);
        d["default_grad"] = buffer_to_array<float, float>(buffers.default_grad);
        d["default_radii"] = buffer_to_array<float, float>(buffers.default_radii);
        return d;
    }

    py::dict snapshot_optimizer_state() {
        py::dict snap;
        snap["num_splats"] = buffers.num_splats;
        snap["cap_max"] = config.cap_max;
        snap["growth_cap"] = trainer.get_growth_cap();
        snap["cap_speculative_max"] = config.cap_speculative_max;
        snap["grow_factor"] = config.grow_factor;
        snap["refine_every"] = config.refine_every;
        snap["structural_reset_opa"] = config.structural_reset_opa;
        snap["feature_lr_final_scale"] = config.feature_lr_final_scale;
        snap["opacity_lr_final_scale"] = config.opacity_lr_final_scale;
        snap["scale_lr_final_scale"] = config.scale_lr_final_scale;
        snap["max_steps"] = config.max_steps;
        snap["xyz_ws"] = buffer_to_array<float, float>(buffers.xyz_ws, 3, (size_t)-1, "mean");
        snap["sh_coeffs"] = buffer_to_array<float, float>(buffers.sh_coeffs, 16, 3, "sh");
        snap["rotations"] = buffer_to_array<float, float>(buffers.rotations, 4);
        snap["scales_opacs"] = buffer_to_array<float, float>(buffers.scales_opacs, 4);
        snap["splat_group_ids"] = get_splat_group_ids();
        snap["splat_entity_ids"] = get_splat_entity_ids();
        snap["splat_uids"] = get_splat_uids();
        snap["next_splat_uid"] = trainer.get_next_splat_uid();
        if (!trainer.get_splat_states().empty()) {
            snap["splat_states"] = get_splat_states();
        }
        if (trainer.has_local_cell_credits()) {
            snap["use_local_cell_credits"] = true;
            snap["splat_cell_ids"] = get_splat_cell_ids();
            snap["cell_credits"] = get_local_cell_credits();
            const auto& elig = trainer.get_cell_eligible();
            snap["cell_eligible"] = py::array_t<int32_t>({elig.size()}, {sizeof(int32_t)}, elig.data());
        }
        if (buffers.splat_anchor.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.splat_anchor.deviceSize() > 0) {
            snap["splat_anchor"] = buffer_to_array<float, float>(buffers.splat_anchor, 8, (size_t)-1, "mean");
        }
        if (buffers.g_xyz_ws.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.g_xyz_ws.deviceSize() > 0) {
            snap["g_xyz_ws"] = buffer_to_array<float, float>(buffers.g_xyz_ws);
            snap["g_sh_coeffs_1"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_1);
            snap["g_sh_coeffs_2"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_2);
            snap["g_rotations"] = buffer_to_array<float, float>(buffers.g_rotations);
            snap["g_scales_opacs"] = buffer_to_array<float, float>(buffers.g_scales_opacs);
        }
        if (buffers.default_grad.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_grad.deviceSize() > 0) {
            snap["default_grad"] = buffer_to_array<float, float>(buffers.default_grad);
        }
        if (buffers.default_radii.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_radii.deviceSize() > 0) {
            snap["default_radii"] = buffer_to_array<float, float>(buffers.default_radii);
        }
        return snap;
    }

    void restore_optimizer_state(py::dict snap) {
        size_t n = snap["num_splats"].cast<size_t>();
        buffers.num_splats = n;
        array_to_buffer(snap["xyz_ws"].cast<py::array_t<float>>(), buffers.xyz_ws);
        auto sh_arr = snap["sh_coeffs"].cast<py::array_t<float>>();
        auto sh_buf = sh_arr.request();
        buffers.sh_coeffs.assign((float*)sh_buf.ptr, (float*)sh_buf.ptr + sh_buf.size);
        buffers.reorderSH(buffers.sh_coeffs);
        trainer.copyToDevice(buffers.sh_coeffs);
        array_to_buffer(snap["rotations"].cast<py::array_t<float>>(), buffers.rotations);
        array_to_buffer(snap["scales_opacs"].cast<py::array_t<float>>(), buffers.scales_opacs);
        if (snap.contains(py::str("splat_group_ids"))) {
            set_splat_group_ids(snap["splat_group_ids"].cast<py::array_t<int32_t>>());
        }
        if (snap.contains(py::str("splat_entity_ids"))) {
            set_splat_entity_ids(snap["splat_entity_ids"].cast<py::array_t<int32_t>>());
        }
        if (snap.contains(py::str("splat_uids"))) {
            auto u_arr = snap["splat_uids"].cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
            std::vector<uint64_t> u_vec(u_arr.data(), u_arr.data() + u_arr.size());
            uint64_t next_u = snap.contains(py::str("next_splat_uid"))
                ? snap["next_splat_uid"].cast<uint64_t>()
                : (u_vec.empty() ? 1u : (*std::max_element(u_vec.begin(), u_vec.end()) + 1u));
            trainer.set_splat_uids_vec(std::move(u_vec), next_u);
        }
        if (snap.contains(py::str("splat_states"))) {
            set_splat_states(snap["splat_states"].cast<py::array_t<uint32_t>>());
        } else {
            trainer.set_splat_states_vec({});
            buffers.splat_state.assign(std::max<size_t>(1, n), 0xFFFFFFFFu);
            trainer.copyToDevice(buffers.splat_state);
        }
        if (snap.contains(py::str("use_local_cell_credits")) && snap["use_local_cell_credits"].cast<bool>()) {
            auto cids = snap["splat_cell_ids"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            auto creds = snap["cell_credits"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> elig;
            const int32_t* elig_ptr = nullptr;
            if (snap.contains(py::str("cell_eligible"))) {
                elig = snap["cell_eligible"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                if (elig.size() == creds.size())
                    elig_ptr = elig.data();
            }
            trainer.set_local_cell_credits(
                buffers, cids.data(), static_cast<size_t>(cids.size()),
                creds.data(), static_cast<size_t>(creds.size()), elig_ptr
            );
        } else {
            trainer.clear_local_cell_credits();
        }
        if (snap.contains(py::str("splat_anchor"))) {
            set_splat_anchors(snap["splat_anchor"].cast<py::array_t<float>>());
        } else {
            trainer.set_splat_anchors(buffers, nullptr, 0);
        }
        if (snap.contains(py::str("g_xyz_ws"))) {
            array_to_buffer(snap["g_xyz_ws"].cast<py::array_t<float>>(), buffers.g_xyz_ws);
            array_to_buffer(snap["g_sh_coeffs_1"].cast<py::array_t<float>>(), buffers.g_sh_coeffs_1);
            array_to_buffer(snap["g_sh_coeffs_2"].cast<py::array_t<float>>(), buffers.g_sh_coeffs_2);
            array_to_buffer(snap["g_rotations"].cast<py::array_t<float>>(), buffers.g_rotations);
            array_to_buffer(snap["g_scales_opacs"].cast<py::array_t<float>>(), buffers.g_scales_opacs);
        }
        if (snap.contains(py::str("default_grad"))) {
            array_to_buffer(snap["default_grad"].cast<py::array_t<float>>(), buffers.default_grad);
        }
        if (snap.contains(py::str("default_radii"))) {
            array_to_buffer(snap["default_radii"].cast<py::array_t<float>>(), buffers.default_radii);
        }
        if (snap.contains(py::str("growth_cap"))) {
            trainer.set_growth_cap(snap["growth_cap"].cast<size_t>());
        }
        if (snap.contains(py::str("cap_max"))) {
            set_densification_budget(
                snap["cap_max"].cast<int>(),
                snap.contains(py::str("cap_speculative_max")) ? snap["cap_speculative_max"].cast<int>() : -1,
                snap.contains(py::str("grow_factor")) ? snap["grow_factor"].cast<float>() : -1.0f,
                snap.contains(py::str("refine_every")) ? snap["refine_every"].cast<int>() : -1
            );
        }
        if (snap.contains(py::str("structural_reset_opa")))
            config.structural_reset_opa = snap["structural_reset_opa"].cast<float>();
        if (snap.contains(py::str("feature_lr_final_scale")))
            config.feature_lr_final_scale = snap["feature_lr_final_scale"].cast<float>();
        if (snap.contains(py::str("opacity_lr_final_scale")))
            config.opacity_lr_final_scale = snap["opacity_lr_final_scale"].cast<float>();
        if (snap.contains(py::str("scale_lr_final_scale")))
            config.scale_lr_final_scale = snap["scale_lr_final_scale"].cast<float>();
        if (snap.contains(py::str("max_steps")))
            config.max_steps = snap["max_steps"].cast<int>();
        if (config.strategy == TrainerConfig::Strategy::MCMC) {
            const size_t alloc_size = std::max(n, config.mcmc_alloc_cap());
            trainer.resizeAndCopyDeviceBuffer(buffers.xyz_ws, 3 * alloc_size, false);
            trainer.resizeAndCopyDeviceBuffer(buffers.sh_coeffs, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), false);
            trainer.resizeAndCopyDeviceBuffer(buffers.rotations, 4 * alloc_size, false);
            trainer.resizeAndCopyDeviceBuffer(buffers.scales_opacs, 4 * alloc_size, false);
            if (buffers.g_xyz_ws.deviceBuffer.buffer != VK_NULL_HANDLE) {
                trainer.resizeAndCopyDeviceBuffer(buffers.g_xyz_ws, 2 * 3 * alloc_size, true);
                trainer.resizeAndCopyDeviceBuffer(buffers.g_sh_coeffs_1, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), true);
                trainer.resizeAndCopyDeviceBuffer(buffers.g_sh_coeffs_2, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), true);
                trainer.resizeAndCopyDeviceBuffer(buffers.g_rotations, 2 * 4 * alloc_size, true);
                trainer.resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2 * 4 * alloc_size, true);
            }
            if (trainer.has_local_cell_credits()) {
                trainer.resizeAndCopyDeviceBuffer(buffers.splat_cell_ids, alloc_size, false);
            }
        }
    }

    // AGO-0 opt-in zero-perturbation per-Gaussian telemetry
    void set_ago_telemetry_enabled(bool enabled) {
        size_t cap = buffers.num_splats;
        if (config.strategy == TrainerConfig::Strategy::MCMC)
            cap = std::max(cap, config.mcmc_alloc_cap());
        trainer.set_ago_telemetry_enabled(enabled, cap);
    }

    bool get_ago_telemetry_enabled() const {
        return trainer.get_ago_telemetry_enabled();
    }

    void reset_ago_telemetry() {
        size_t cap = buffers.num_splats;
        if (config.strategy == TrainerConfig::Strategy::MCMC)
            cap = std::max(cap, config.mcmc_alloc_cap());
        trainer.reset_ago_telemetry(cap);
    }

    py::dict get_ago_telemetry() {
        const size_t n = buffers.num_splats;
        std::vector<float> raw = trainer.get_ago_telemetry_host(n);
        auto make_col = [&](size_t col) -> py::array_t<float> {
            py::array_t<float> arr({static_cast<py::ssize_t>(n)});
            float* dst = arr.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                dst[i] = raw[8 * i + col];
            }
            return arr;
        };
        py::dict d;
        d["grad_norm"] = make_col(0);
        d["update_norm"] = make_col(1);
        d["predicted_gain"] = make_col(2);
        d["abs_proj_grad"] = make_col(3);
        d["update_count"] = make_col(4);
        d["contrib_sum"] = make_col(5);
        d["last_step_updated"] = make_col(6);
        d["last_predicted_gain"] = make_col(7);
        d["num_gaussians"] = n;
        d["step"] = trainer.get_ago_last_step();
        return d;
    }

    static py::dict _ago_summary_to_dict(const VulkanGSTrainer::AgoActiveSetSummary& s) {
        py::dict d;
        d["active_count"] = static_cast<int64_t>(s.active_count);
        d["num_gaussians"] = static_cast<int64_t>(s.num_gaussians);
        d["active_fraction"] = static_cast<double>(s.active_fraction);
        d["total_predicted_gain"] = static_cast<double>(s.total_predicted_gain);
        d["mean_predicted_gain"] = static_cast<double>(s.mean_predicted_gain);
        d["max_predicted_gain"] = static_cast<double>(s.max_predicted_gain);
        d["eligible_count"] = static_cast<int64_t>(s.eligible_count);
        d["gain_threshold"] = static_cast<double>(s.gain_threshold);
        d["fell_back_to_full"] = s.fell_back_to_full;
        return d;
    }

    py::dict select_ago_active_set_gpu(
        float gain_threshold = 0.0f,
        float newborn_warmup_steps = 2.0f,
        bool exclude_frozen = true
    ) {
        const auto s = trainer.select_ago_active_set_gpu(
            config, buffers, gain_threshold, newborn_warmup_steps, exclude_frozen
        );
        return _ago_summary_to_dict(s);
    }

    py::dict select_ago_adaptive_active_set_gpu(
        float target_gain_ratio = 0.5f,
        float newborn_warmup_steps = 2.0f,
        bool exclude_frozen = true
    ) {
        const auto s = trainer.select_ago_adaptive_active_set_gpu(
            config, buffers, target_gain_ratio, newborn_warmup_steps, exclude_frozen
        );
        return _ago_summary_to_dict(s);
    }

    py::dict select_ago_exact_k_gpu(
        float target_fraction = 0.5f,
        float newborn_warmup_steps = 10.0f,
        bool exclude_frozen = true
    ) {
        const auto s = trainer.select_ago_exact_k_gpu(
            config, buffers, target_fraction, newborn_warmup_steps, exclude_frozen
        );
        return _ago_summary_to_dict(s);
    }

    void select_ago_active_set_gpu_async(
        float target_fraction = 0.5f,
        float newborn_warmup_steps = 10.0f,
        bool exclude_frozen = true,
        uint32_t max_budget_splats = 0u
    ) {
        trainer.select_ago_active_set_gpu_async(
            config, buffers, target_fraction, newborn_warmup_steps, exclude_frozen, max_budget_splats
        );
    }

    py::dict get_ago_cutoff_indirect_state() {
        const auto st = trainer.get_ago_cutoff_indirect_state();
        py::dict d;
        d["group_count_x"] = st.group_count_x;
        d["group_count_y"] = st.group_count_y;
        d["group_count_z"] = st.group_count_z;
        d["cutoff_bin"] = st.cutoff_bin;
        d["boundary_quota"] = st.boundary_quota;
        d["selected_count"] = st.selected_count;
        d["active_set_state"] = st.active_set_state;
        d["selection_mode"] = st.selection_mode;
        d["eligible_count"] = st.eligible_count;
        d["newborn_count"] = st.newborn_count;
        d["total_predicted_gain"] = st.total_predicted_gain;
        d["max_predicted_gain"] = st.max_predicted_gain;
        return d;
    }

    void set_external_strategy_mode(bool enabled) {
        config.external_strategy_mode = enabled;
    }

    bool get_external_strategy_mode() const {
        return config.external_strategy_mode;
    }

    void set_ago_param_mask(uint32_t mask) {
        config.ago_param_mask = mask;
    }

    uint32_t get_ago_param_mask() const {
        return config.ago_param_mask;
    }

    vksplat::ago::CompletionStatus ago_execute_external_op(
        vksplat::ago::OperationType op,
        int32_t group_id = -1
    ) {
        return trainer.ago_execute_external_op(config, buffers, op, group_id);
    }

    uint64_t ago_offload_snapshot_to_ram() {
        return trainer.ago_offload_snapshot_to_ram(buffers);
    }

    bool ago_restore_snapshot_from_ram(uint64_t snapshot_id) {
        return trainer.ago_restore_snapshot_from_ram(buffers, snapshot_id);
    }

    bool ago_drop_ram_snapshot(uint64_t snapshot_id) {
        return trainer.ago_drop_ram_snapshot(snapshot_id);
    }

    size_t ago_ram_snapshot_bytes(uint64_t snapshot_id) const {
        return trainer.ago_ram_snapshot_bytes(snapshot_id);
    }

    vksplat::ago::SubmitResult ago_gpu_submit_action(const vksplat::ago::AGOAction& action) {
        return trainer.ago_gpu_submit_action(action);
    }

    std::vector<vksplat::ago::CompletionRecord> ago_gpu_poll_completions(size_t max_count = 16u) {
        return trainer.ago_gpu_poll_completions(max_count);
    }

    uint32_t ago_gpu_available_credits() const {
        return const_cast<VulkanGSTrainer&>(trainer).ago_gpu_executor().available_credits();
    }

    uint32_t ago_gpu_inflight_count() const {
        return const_cast<VulkanGSTrainer&>(trainer).ago_gpu_executor().inflight_count();
    }

    void ago_gpu_shutdown() {
        trainer.ago_gpu_shutdown();
    }

    void ago_scheduler_init(
        const vksplat::ago::SchedulerConfig& cfg = {},
        bool use_gpu_executor = false
    ) {
        vksplat::ago::ActionExecutorInterface* exec =
            use_gpu_executor ? static_cast<vksplat::ago::ActionExecutorInterface*>(&trainer.ago_gpu_executor())
                             : nullptr;
        ago_scheduler_ = std::make_unique<vksplat::ago::AgoScheduler>(cfg, exec);
    }

    vksplat::ago::TickResult ago_scheduler_tick() {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        return ago_scheduler_->tick();
    }

    bool ago_scheduler_enqueue_action(const vksplat::ago::AGOAction& action) {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        return ago_scheduler_->enqueue_action(action);
    }

    bool ago_scheduler_push_evidence(const vksplat::ago::EvidenceMessage& msg) {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        if (msg.critical ||
            msg.kind == vksplat::ago::EvidenceKind::POSE_REVISION ||
            msg.kind == vksplat::ago::EvidenceKind::CONTRADICTION ||
            msg.kind == vksplat::ago::EvidenceKind::STRUCTURAL_COMPLETION) {
            return ago_scheduler_->push_critical_evidence(msg);
        }
        return ago_scheduler_->push_telemetry_evidence(msg);
    }

    void ago_scheduler_set_budget(const vksplat::ago::DeadlineBudget& budget) {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        ago_scheduler_->publish_deadline_budget(budget);
    }

    vksplat::ago::SchedulerSummary ago_scheduler_summary() {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        return ago_scheduler_->latest_summary();
    }

    std::vector<vksplat::ago::EventTraceEntry> ago_scheduler_trace() {
        if (!ago_scheduler_) {
            ago_scheduler_init();
        }
        std::vector<vksplat::ago::EventTraceEntry> out;
        ago_scheduler_->event_trace().snapshot(out);
        return out;
    }

    py::dict reproject_temporal_depth_gpu(
        py::array_t<float, py::array::c_style | py::array::forcecast> src_depths,
        py::array_t<float, py::array::c_style | py::array::forcecast> src_c2ws,
        py::array_t<float, py::array::c_style | py::array::forcecast> dst_c2w,
        py::array_t<float, py::array::c_style | py::array::forcecast> K_src,
        py::array_t<float, py::array::c_style | py::array::forcecast> K_dst,
        float max_depth_m = 10.0f,
        py::object dst_hw_obj = py::none(),
        float min_depth_m = 0.05f,
        float min_cos_angle = 0.0f
    ) {
        auto dbuf = src_depths.request();
        if (dbuf.ndim != 3) {
            throw std::invalid_argument("src_depths must have shape (H_hist, H_src, W_src)");
        }
        const size_t num_views = static_cast<size_t>(dbuf.shape[0]);
        const int src_h = static_cast<int>(dbuf.shape[1]);
        const int src_w = static_cast<int>(dbuf.shape[2]);

        int dst_h = src_h;
        int dst_w = src_w;
        if (!dst_hw_obj.is_none()) {
            auto seq = dst_hw_obj.cast<py::sequence>();
            if (py::len(seq) != 2) {
                throw std::invalid_argument("dst_hw must be a (H_dst, W_dst) pair");
            }
            dst_h = seq[0].cast<int>();
            dst_w = seq[1].cast<int>();
        }

        auto scbuf = src_c2ws.request();
        if (scbuf.ndim != 3 || static_cast<size_t>(scbuf.shape[0]) != num_views ||
            scbuf.shape[1] != 4 || scbuf.shape[2] != 4) {
            throw std::invalid_argument("src_c2ws must have shape (H_hist, 4, 4)");
        }
        auto dcbuf = dst_c2w.request();
        if (dcbuf.ndim != 2 || dcbuf.shape[0] != 4 || dcbuf.shape[1] != 4) {
            throw std::invalid_argument("dst_c2w must have shape (4, 4)");
        }

        auto extract_single_k = [](const py::buffer_info& b, float out[4], const char* name) {
            const float* p = static_cast<const float*>(b.ptr);
            if (b.ndim == 2 && b.shape[0] == 3 && b.shape[1] == 3) {
                out[0] = p[0];
                out[1] = p[4];
                out[2] = p[2];
                out[3] = p[5];
            } else if (b.ndim == 1 && b.shape[0] == 4) {
                out[0] = p[0];
                out[1] = p[1];
                out[2] = p[2];
                out[3] = p[3];
            } else {
                throw std::invalid_argument(std::string(name) + " must have shape (3, 3) or (4,)");
            }
            if (!(out[0] > 0.0f && out[1] > 0.0f && std::isfinite(out[0]) && std::isfinite(out[1]))) {
                throw std::invalid_argument(std::string(name) + " focal lengths fx, fy must be finite and > 0");
            }
        };

        float kd[4] = {};
        extract_single_k(K_dst.request(), kd, "K_dst");

        auto ksbuf = K_src.request();
        const bool ks_shared = (ksbuf.ndim == 2 && ksbuf.shape[0] == 3 && ksbuf.shape[1] == 3) ||
                               (ksbuf.ndim == 1 && ksbuf.shape[0] == 4);
        const bool ks_per_view = (ksbuf.ndim == 3 && static_cast<size_t>(ksbuf.shape[0]) == num_views &&
                                  ksbuf.shape[1] == 3 && ksbuf.shape[2] == 3) ||
                                 (ksbuf.ndim == 2 && static_cast<size_t>(ksbuf.shape[0]) == num_views &&
                                  ksbuf.shape[1] == 4);
        if (!ks_shared && !ks_per_view) {
            throw std::invalid_argument("K_src must have shape (3, 3), (4,), (H_hist, 3, 3), or (H_hist, 4)");
        }
        float ks_shared_vals[4] = {};
        if (ks_shared) {
            extract_single_k(ksbuf, ks_shared_vals, "K_src");
        }

        // Load row-major 4x4 dst_c2w into column-major glm::dmat4 and invert in float64
        const float* dptr = static_cast<const float*>(dcbuf.ptr);
        glm::dmat4 dst_c2w_d(1.0);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                dst_c2w_d[c][r] = static_cast<double>(dptr[4 * r + c]);
            }
        }
        const glm::dmat4 dst_w2c_d = glm::inverse(dst_c2w_d);

        std::vector<float> view_params(16 * num_views, 0.0f);
        const float* sptr = static_cast<const float*>(scbuf.ptr);
        for (size_t v = 0; v < num_views; ++v) {
            const float* sv = sptr + 16 * v;
            glm::dmat4 src_c2w_d(1.0);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    src_c2w_d[c][r] = static_cast<double>(sv[4 * r + c]);
                }
            }
            const glm::dmat4 t_s2d = dst_w2c_d * src_c2w_d;
            float* vp = &view_params[16 * v];
            // Store rows 0, 1, 2 of T_src_to_dst
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    vp[4 * r + c] = static_cast<float>(t_s2d[c][r]);
                }
            }
            float kv[4] = {};
            if (ks_shared) {
                memcpy(kv, ks_shared_vals, sizeof(kv));
            } else {
                extract_k_params(ksbuf, v, kv, "K_src");
                if (!(kv[0] > 0.0f && kv[1] > 0.0f && std::isfinite(kv[0]) && std::isfinite(kv[1]))) {
                    throw std::invalid_argument("K_src focal lengths fx, fy must be finite and > 0");
                }
            }
            vp[12] = 1.0f / kv[0];  // inv_fx_s
            vp[13] = 1.0f / kv[1];  // inv_fy_s
            vp[14] = kv[2];         // cx_s
            vp[15] = kv[3];         // cy_s
        }

        auto res = trainer.reproject_temporal_depth_gpu(
            static_cast<const float*>(dbuf.ptr),
            num_views,
            src_h,
            src_w,
            view_params.data(),
            dst_h,
            dst_w,
            kd[0],
            kd[1],
            kd[2],
            kd[3],
            max_depth_m,
            min_depth_m,
            min_cos_angle
        );

        py::array_t<float> depth_arr({dst_h, dst_w});
        py::array_t<float> weight_arr({dst_h, dst_w});
        py::array_t<uint32_t> support_arr({dst_h, dst_w});
        const size_t n_dst = static_cast<size_t>(dst_h) * static_cast<size_t>(dst_w);
        memcpy(depth_arr.mutable_data(), res.depth.data(), n_dst * sizeof(float));
        memcpy(weight_arr.mutable_data(), res.weight.data(), n_dst * sizeof(float));
        memcpy(support_arr.mutable_data(), res.support.data(), n_dst * sizeof(uint32_t));

        py::dict out;
        out["depth"] = std::move(depth_arr);
        out["weight"] = std::move(weight_arr);
        out["support_count"] = std::move(support_arr);
        out["kernel_ms"] = res.kernel_ms;
        out["total_ms"] = res.total_ms;
        return out;
    }

    py::dict get_ago_step_gain_summary() const {
        const auto s = trainer.get_ago_step_gain_summary(buffers.num_splats);
        return _ago_summary_to_dict(s);
    }

    // D1 opt-in native differentiable physical expected-depth supervision
    void set_train_depth(
        size_t frame_idx,
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_m,
        py::object weight_map = py::none()
    ) {
        auto d_buf = depth_m.request();
        if (d_buf.ndim != 2) {
            throw std::invalid_argument("set_train_depth: depth_m must be a 2D (H, W) float32 array");
        }
        const int h = static_cast<int>(d_buf.shape[0]);
        const int w = static_cast<int>(d_buf.shape[1]);
        const float* w_ptr = nullptr;
        py::array_t<float, py::array::c_style | py::array::forcecast> w_arr;
        if (!weight_map.is_none()) {
            w_arr = weight_map.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto w_buf = w_arr.request();
            if (w_buf.ndim != 2 || w_buf.shape[0] != h || w_buf.shape[1] != w) {
                throw std::invalid_argument("set_train_depth: weight_map must match depth_m shape (H, W)");
            }
            w_ptr = static_cast<const float*>(w_buf.ptr);
        }
        trainer.set_train_depth(frame_idx, static_cast<const float*>(d_buf.ptr), w_ptr, h, w);
    }

    void clear_train_depths() {
        current_train_depth_buf = nullptr;
        trainer.clear_train_depths();
    }

    void set_depth_loss_weight(float lambda_depth, float huber_delta = 0.15f) {
        trainer.set_depth_loss_weight(lambda_depth, huber_delta);
    }

    float get_depth_loss_weight() const {
        return trainer.get_depth_loss_weight();
    }

    // Explicit optimization active-set API (independent of evidence-based freezing)
    void set_active_set(py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices) {
        auto buf = indices.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("set_active_set: indices must be a 1D integer array");
        }
        const size_t n = buffers.num_splats;
        const int64_t* ptr = static_cast<const int64_t*>(buf.ptr);
        const size_t count = static_cast<size_t>(buf.size);
        std::vector<uint32_t> active;
        active.reserve(count);
        std::vector<uint8_t> seen(n, 0);
        for (size_t i = 0; i < count; ++i) {
            const int64_t idx = ptr[i];
            if (idx < 0 || static_cast<size_t>(idx) >= n) {
                throw std::out_of_range(
                    "set_active_set: index " + std::to_string(idx) +
                    " out of range for num_gaussians=" + std::to_string(n)
                );
            }
            const uint32_t u = static_cast<uint32_t>(idx);
            if (!seen[u]) {
                seen[u] = 1;
                active.push_back(u);
            }
        }
        trainer.set_active_indices_gpu(active, n);
    }

    void clear_active_set() {
        trainer.invalidateActiveIndexDispatch();
    }

    py::array_t<uint32_t> get_active_set() {
        std::vector<uint32_t> active = trainer.get_active_indices_host();
        return py::array_t<uint32_t>(
            {static_cast<py::ssize_t>(active.size())},
            {static_cast<py::ssize_t>(sizeof(uint32_t))},
            active.data()
        );
    }

    // AGO-1 exact state snapshot & restore for byte-identical counterfactual gain probes
    py::dict snapshot_ago_state() {
        py::dict snap = snapshot_optimizer_state();
        const size_t n = buffers.num_splats;
        snap["num_gaussians"] = n;
        snap["step"] = trainer.get_ago_last_step();

        // Friendly parameter aliases alongside raw packed arrays
        auto xyz_arr = snap["xyz_ws"].cast<py::array_t<float>>();
        auto rot_arr = snap["rotations"].cast<py::array_t<float>>();
        auto sh_arr = snap["sh_coeffs"].cast<py::array_t<float>>();
        auto so_arr = snap["scales_opacs"].cast<py::array_t<float>>();
        snap["means"] = xyz_arr;
        snap["quats"] = rot_arr;

        // Slice scales (N, 3), opacities (N,), sh0 (N, 1, 3), shN (N, 15, 3)
        auto so_buf = so_arr.unchecked<2>();
        py::array_t<float> scales_arr({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
        py::array_t<float> opacs_arr({static_cast<py::ssize_t>(n)});
        auto sc_mut = scales_arr.mutable_unchecked<2>();
        auto op_mut = opacs_arr.mutable_unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            sc_mut(i, 0) = so_buf(i, 0);
            sc_mut(i, 1) = so_buf(i, 1);
            sc_mut(i, 2) = so_buf(i, 2);
            op_mut(i) = so_buf(i, 3);
        }
        snap["scales"] = scales_arr;
        snap["opacities"] = opacs_arr;

        auto sh_buf = sh_arr.unchecked<3>();
        py::array_t<float> sh0_arr({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(1), static_cast<py::ssize_t>(3)});
        py::array_t<float> shN_arr({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(15), static_cast<py::ssize_t>(3)});
        auto sh0_mut = sh0_arr.mutable_unchecked<3>();
        auto shN_mut = shN_arr.mutable_unchecked<3>();
        for (size_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                sh0_mut(i, 0, c) = sh_buf(i, 0, c);
            }
            for (int k = 0; k < 15; ++k) {
                for (int c = 0; c < 3; ++c) {
                    shN_mut(i, k, c) = sh_buf(i, k + 1, c);
                }
            }
        }
        snap["sh0"] = sh0_arr;
        snap["shN"] = shN_arr;

        snap["has_optimizer_moments"] = snap.contains(py::str("g_xyz_ws"));
        snap["has_default_grad"] = snap.contains(py::str("default_grad"));
        snap["has_default_radii"] = snap.contains(py::str("default_radii"));

        if (buffers.group_state.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.group_state.deviceSize() > 0) {
            snap["group_state"] = buffer_to_array<uint32_t, uint32_t>(buffers.group_state);
        }

        if (trainer.get_use_active_index_dispatch() &&
            trainer.get_active_set_generation() == trainer.get_map_generation()) {
            std::vector<uint32_t> active = trainer.get_active_indices_host();
            snap["active_set"] = py::array_t<uint32_t>({static_cast<py::ssize_t>(active.size())},
                                                       {static_cast<py::ssize_t>(sizeof(uint32_t))},
                                                       active.data());
        } else {
            snap["active_set"] = py::none();
        }

        if (trainer.get_ago_telemetry_enabled() && n > 0) {
            std::vector<float> tel = trainer.get_ago_telemetry_host(n);
            snap["ago_telemetry"] = py::array_t<float>({static_cast<py::ssize_t>(tel.size())},
                                                       {static_cast<py::ssize_t>(sizeof(float))},
                                                       tel.data());
        } else {
            snap["ago_telemetry"] = py::none();
        }

        return snap;
    }

    void restore_ago_state(py::dict snap) {
        size_t n = snap.contains(py::str("num_gaussians"))
            ? snap["num_gaussians"].cast<size_t>()
            : snap["num_splats"].cast<size_t>();
        int step = snap.contains(py::str("step")) ? snap["step"].cast<int>() : 0;

        {
            DeviceGuard guard(&trainer);
            buffers.num_splats = n;
            trainer.set_ago_last_step(step);

            // Upload primary Gaussian parameters
            if (snap.contains(py::str("xyz_ws"))) {
                array_to_buffer(snap["xyz_ws"].cast<py::array_t<float>>(), buffers.xyz_ws);
            } else {
                array_to_buffer(snap["means"].cast<py::array_t<float>>(), buffers.xyz_ws);
            }

            if (snap.contains(py::str("sh_coeffs"))) {
                auto sh_arr = snap["sh_coeffs"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto sh_buf = sh_arr.request();
                buffers.sh_coeffs.assign(static_cast<float*>(sh_buf.ptr),
                                         static_cast<float*>(sh_buf.ptr) + sh_buf.size);
            } else {
                auto sh0_arr = snap["sh0"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto shN_arr = snap["shN"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto sh0_b = sh0_arr.unchecked<3>();
                auto shN_b = shN_arr.unchecked<3>();
                buffers.sh_coeffs.resize(n * 16 * 3);
                for (size_t i = 0; i < n; ++i) {
                    for (int c = 0; c < 3; ++c)
                        buffers.sh_coeffs[i * 48 + c] = sh0_b(i, 0, c);
                    for (int k = 0; k < 15; ++k)
                        for (int c = 0; c < 3; ++c)
                            buffers.sh_coeffs[i * 48 + (k + 1) * 3 + c] = shN_b(i, k, c);
                }
            }
            buffers.reorderSH(buffers.sh_coeffs);
            trainer.copyToDevice(buffers.sh_coeffs);

            if (snap.contains(py::str("rotations"))) {
                array_to_buffer(snap["rotations"].cast<py::array_t<float>>(), buffers.rotations);
            } else {
                array_to_buffer(snap["quats"].cast<py::array_t<float>>(), buffers.rotations);
            }

            if (snap.contains(py::str("scales_opacs"))) {
                array_to_buffer(snap["scales_opacs"].cast<py::array_t<float>>(), buffers.scales_opacs);
            } else {
                auto sc_arr = snap["scales"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto op_arr = snap["opacities"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                buffers.assignScalesOpacs(buffers.scales_opacs, n, sc_arr.data(), op_arr.data());
                trainer.copyToDevice(buffers.scales_opacs);
            }

            // Restore or zero all 5 Adam moment buffers
            if (snap.contains(py::str("g_xyz_ws"))) {
                array_to_buffer(snap["g_xyz_ws"].cast<py::array_t<float>>(), buffers.g_xyz_ws);
                array_to_buffer(snap["g_sh_coeffs_1"].cast<py::array_t<float>>(), buffers.g_sh_coeffs_1);
                array_to_buffer(snap["g_sh_coeffs_2"].cast<py::array_t<float>>(), buffers.g_sh_coeffs_2);
                array_to_buffer(snap["g_rotations"].cast<py::array_t<float>>(), buffers.g_rotations);
                array_to_buffer(snap["g_scales_opacs"].cast<py::array_t<float>>(), buffers.g_scales_opacs);
            } else {
                auto clear_if_alloc = [&](Buffer<float>& buf) {
                    if (buf.deviceBuffer.buffer != VK_NULL_HANDLE && buf.deviceSize() > 0) {
                        trainer.clearDeviceBuffer(buf, buf.deviceSize());
                    }
                };
                clear_if_alloc(buffers.g_xyz_ws);
                clear_if_alloc(buffers.g_sh_coeffs_1);
                clear_if_alloc(buffers.g_sh_coeffs_2);
                clear_if_alloc(buffers.g_rotations);
                clear_if_alloc(buffers.g_scales_opacs);
            }

            // Restore or zero strategy running_grad/running_count & radii buffers
            if (snap.contains(py::str("default_grad"))) {
                array_to_buffer(snap["default_grad"].cast<py::array_t<float>>(), buffers.default_grad);
            } else if (buffers.default_grad.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_grad.deviceSize() > 0) {
                trainer.clearDeviceBuffer(buffers.default_grad, buffers.default_grad.deviceSize());
            }
            if (snap.contains(py::str("default_radii"))) {
                array_to_buffer(snap["default_radii"].cast<py::array_t<float>>(), buffers.default_radii);
            } else if (buffers.default_radii.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_radii.deviceSize() > 0) {
                trainer.clearDeviceBuffer(buffers.default_radii, buffers.default_radii.deviceSize());
            }

            // Restore group/entity/UID/state/anchor/cell metadata
            if (snap.contains(py::str("splat_group_ids"))) {
                set_splat_group_ids(snap["splat_group_ids"].cast<py::array_t<int32_t>>());
            }
            if (snap.contains(py::str("group_state"))) {
                array_to_buffer(snap["group_state"].cast<py::array_t<uint32_t>>(), buffers.group_state);
            }
            if (snap.contains(py::str("splat_entity_ids"))) {
                set_splat_entity_ids(snap["splat_entity_ids"].cast<py::array_t<int32_t>>());
            }
            if (snap.contains(py::str("splat_uids"))) {
                auto u_arr = snap["splat_uids"].cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
                std::vector<uint64_t> u_vec(u_arr.data(), u_arr.data() + u_arr.size());
                uint64_t next_u = snap.contains(py::str("next_splat_uid"))
                    ? snap["next_splat_uid"].cast<uint64_t>()
                    : (u_vec.empty() ? 1u : (*std::max_element(u_vec.begin(), u_vec.end()) + 1u));
                trainer.set_splat_uids_vec(std::move(u_vec), next_u);
            }
            if (snap.contains(py::str("splat_states"))) {
                set_splat_states(snap["splat_states"].cast<py::array_t<uint32_t>>());
            } else {
                trainer.set_splat_states_vec({});
                buffers.splat_state.assign(std::max<size_t>(1, n), 0xFFFFFFFFu);
                trainer.copyToDevice(buffers.splat_state);
            }
            if (snap.contains(py::str("use_local_cell_credits")) && snap["use_local_cell_credits"].cast<bool>()) {
                auto cids = snap["splat_cell_ids"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto creds = snap["cell_credits"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> elig;
                const int32_t* elig_ptr = nullptr;
                if (snap.contains(py::str("cell_eligible"))) {
                    elig = snap["cell_eligible"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                    if (elig.size() == creds.size())
                        elig_ptr = elig.data();
                }
                trainer.set_local_cell_credits(
                    buffers, cids.data(), static_cast<size_t>(cids.size()),
                    creds.data(), static_cast<size_t>(creds.size()), elig_ptr
                );
            } else {
                trainer.clear_local_cell_credits();
            }
            if (snap.contains(py::str("splat_anchor"))) {
                set_splat_anchors(snap["splat_anchor"].cast<py::array_t<float>>());
            } else {
                trainer.set_splat_anchors(buffers, nullptr, 0);
            }

            // Restore budget/schedule scalars
            if (snap.contains(py::str("growth_cap"))) {
                trainer.set_growth_cap(snap["growth_cap"].cast<size_t>());
            }
            if (snap.contains(py::str("cap_max"))) {
                set_densification_budget(
                    snap["cap_max"].cast<int>(),
                    snap.contains(py::str("cap_speculative_max")) ? snap["cap_speculative_max"].cast<int>() : -1,
                    snap.contains(py::str("grow_factor")) ? snap["grow_factor"].cast<float>() : -1.0f,
                    snap.contains(py::str("refine_every")) ? snap["refine_every"].cast<int>() : -1
                );
            }
            if (snap.contains(py::str("structural_reset_opa")))
                config.structural_reset_opa = snap["structural_reset_opa"].cast<float>();
            if (snap.contains(py::str("feature_lr_final_scale")))
                config.feature_lr_final_scale = snap["feature_lr_final_scale"].cast<float>();
            if (snap.contains(py::str("opacity_lr_final_scale")))
                config.opacity_lr_final_scale = snap["opacity_lr_final_scale"].cast<float>();
            if (snap.contains(py::str("scale_lr_final_scale")))
                config.scale_lr_final_scale = snap["scale_lr_final_scale"].cast<float>();
            if (snap.contains(py::str("max_steps")))
                config.max_steps = snap["max_steps"].cast<int>();

            if (config.strategy == TrainerConfig::Strategy::MCMC) {
                const size_t alloc_size = std::max(n, config.mcmc_alloc_cap());
                trainer.resizeAndCopyDeviceBuffer(buffers.xyz_ws, 3 * alloc_size, false);
                trainer.resizeAndCopyDeviceBuffer(buffers.sh_coeffs, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), false);
                trainer.resizeAndCopyDeviceBuffer(buffers.rotations, 4 * alloc_size, false);
                trainer.resizeAndCopyDeviceBuffer(buffers.scales_opacs, 4 * alloc_size, false);
                if (buffers.g_xyz_ws.deviceBuffer.buffer != VK_NULL_HANDLE) {
                    trainer.resizeAndCopyDeviceBuffer(buffers.g_xyz_ws, 2 * 3 * alloc_size, true);
                    trainer.resizeAndCopyDeviceBuffer(buffers.g_sh_coeffs_1, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), true);
                    trainer.resizeAndCopyDeviceBuffer(buffers.g_sh_coeffs_2, 16 * 3 * _CEIL_ROUND(alloc_size, SH_REORDER_SIZE), true);
                    trainer.resizeAndCopyDeviceBuffer(buffers.g_rotations, 2 * 4 * alloc_size, true);
                    trainer.resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2 * 4 * alloc_size, true);
                }
                if (trainer.has_local_cell_credits()) {
                    trainer.resizeAndCopyDeviceBuffer(buffers.splat_cell_ids, alloc_size, false);
                }
            }
        }

        // Restore or clear AGO telemetry buffer
        if (snap.contains(py::str("ago_telemetry")) && !snap["ago_telemetry"].is_none()) {
            auto tel_arr = snap["ago_telemetry"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            trainer.set_ago_telemetry_host(tel_arr.data(), static_cast<size_t>(tel_arr.size()));
        } else {
            trainer.set_ago_telemetry_host(nullptr, 0);
        }

        // Bump map_generation_ cleanly and restore active_set if one was active
        trainer.bumpMapGeneration();
        if (snap.contains(py::str("active_set")) && !snap["active_set"].is_none()) {
            auto act_arr = snap["active_set"].cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
            std::vector<uint32_t> act_vec(act_arr.data(), act_arr.data() + act_arr.size());
            trainer.set_active_indices_gpu(act_vec, n);
        }
    }

    size_t num_train() { return trainer.num_train(); }
    size_t num_val() { return trainer.num_val(); }

    py::array_t<uint8_t> get_train_image(size_t &idx) {
        auto& im = trainer.get_train_image(idx);
        if (im.buffer.empty() && im.buffer.deviceBuffer.buffer != VK_NULL_HANDLE)
            trainer.copyFromDevice(im.buffer);  // async uploads keep no host copy
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }
    py::array_t<uint8_t> get_val_image(size_t &idx) {
        auto& im = trainer.get_val_image(idx);
        if (im.buffer.empty() && im.buffer.deviceBuffer.buffer != VK_NULL_HANDLE)
            trainer.copyFromDevice(im.buffer);  // async uploads keep no host copy
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }

    void set_uniforms(
        uint32_t active_sh,
        py::array_t<float> world_view_transform,
        uint32_t image_height, uint32_t image_width,
        float fx, float fy, float cx, float cy,
        bool is_fisheye
    ) {
        auto transform_buf = world_view_transform.request();
        if (transform_buf.ndim != 2 || transform_buf.shape[0] != 4 || transform_buf.shape[1] != 4)
            throw std::runtime_error("world_view_transform must be (4, 4) array");

        uniforms.active_sh = active_sh;
        uniforms.image_height = image_height;
        uniforms.image_width = image_width;
        uniforms.camera_model = is_fisheye ? 2 : 0;
        uniforms.fx = fx;
        uniforms.fy = fy;
        uniforms.cx = cx;
        uniforms.cy = cy;
        uniforms.grid_height = _CEIL_DIV(image_height, TILE_HEIGHT);
        uniforms.grid_width = _CEIL_DIV(image_width, TILE_WIDTH);

        // row/column major matrix conversion
        const float* transform_ptr = static_cast<float*>(transform_buf.ptr);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                uniforms.world_view_transform[4*i+j] = transform_ptr[4*j+i];

        uniforms.step = 0;
    }

    py::dict get_uniforms() {
        py::dict result;
        
        result["active_sh"] = uniforms.active_sh + 0;
        result["image_height"] = uniforms.image_height + 0;
        result["image_width"] = uniforms.image_width + 0;
        result["fx"] = uniforms.fx * 1.0f;
        result["fy"] = uniforms.fy * 1.0f;
        result["cx"] = uniforms.cx * 1.0f;
        result["cy"] = uniforms.cy * 1.0f;
        
        auto world_view_array = py::array_t<float>({4, 4});
        auto world_view_buf = world_view_array.request();
        float* world_view_ptr = static_cast<float*>(world_view_buf.ptr);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                world_view_ptr[4*i+j] = uniforms.world_view_transform[4*j+i];
        result["world_view_transform"] = world_view_array;
        
        auto K_array = py::array_t<float>({3, 3});
        auto K_buf = K_array.request();
        float* K_ptr = static_cast<float*>(K_buf.ptr);
        std::fill(K_ptr, K_ptr + 9, 0.0f);
        K_ptr[0] = uniforms.fx;  // K[0,0] = fx
        K_ptr[4] = uniforms.fy;  // K[1,1] = fy
        K_ptr[2] = uniforms.cx;  // K[0,2] = cx
        K_ptr[5] = uniforms.cy;  // K[1,2] = cy
        K_ptr[8] = 1.0f;  // K[2,2] = 1
        result["K"] = K_array;
        
        return result;
    }

    void set_gauss_params(
        py::array_t<float> xyz_ws,
        py::array_t<float> rotations,
        py::array_t<float> scales,
        py::array_t<float> opacities,
        py::array_t<float> sh_coeffs
    ) {
        // Validate input arrays
        auto xyz_buf = xyz_ws.request();
        auto sh_buf = sh_coeffs.request();
        auto rot_buf = rotations.request();
        auto scale_buf = scales.request();
        auto opacity_buf = opacities.request();
        
        if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3)
            throw std::runtime_error("xyz_ws must be (N, 3) array");
        if (sh_buf.ndim != 3 || sh_buf.shape[1] != 16 || sh_buf.shape[2] != 3)
            throw std::runtime_error("sh_coeffs must be (N, 16, 3) array");
        if (rot_buf.ndim != 2 || rot_buf.shape[1] != 4)
            throw std::runtime_error("rotations must be (N, 4) array");
        if (scale_buf.ndim != 2 || scale_buf.shape[1] != 3)
            throw std::runtime_error("scales must be (N, 3) array");
        if (opacity_buf.ndim != 2 || opacity_buf.shape[1] != 1)
            throw std::runtime_error("opacities must be (N, 1) array");
        
        buffers.num_splats = xyz_buf.shape[0];

        if ((size_t)sh_buf.shape[0] != buffers.num_splats ||
            (size_t)rot_buf.shape[0] != buffers.num_splats || 
            (size_t)scale_buf.shape[0] != buffers.num_splats ||
            (size_t)opacity_buf.shape[0] != buffers.num_splats
        ) throw std::runtime_error("All input arrays must have the same number of elements N");

        // Prepare inputs
        buffers.xyz_ws.assign((float*)(xyz_buf.ptr), (float*)(xyz_buf.ptr) + xyz_buf.size);
        buffers.rotations.assign((float*)(rot_buf.ptr), (float*)(rot_buf.ptr) + rot_buf.size);
        buffers.assignScalesOpacs(buffers.scales_opacs, buffers.num_splats,
            (float*)(scale_buf.ptr), (float*)(opacity_buf.ptr));
        buffers.sh_coeffs.assign((float*)(sh_buf.ptr), (float*)(sh_buf.ptr) + sh_buf.size);

        buffers.reorderSH(buffers.sh_coeffs);

        if (config.strategy == TrainerConfig::Strategy::MCMC) {
            const size_t alloc_cap = std::max(buffers.num_splats, config.mcmc_alloc_cap());
            trainer.resizeDeviceBuffer(buffers.xyz_ws, 3 * alloc_cap);
            trainer.resizeDeviceBuffer(buffers.sh_coeffs, 12 * 4 * alloc_cap);
            trainer.resizeDeviceBuffer(buffers.rotations, 4 * alloc_cap);
            trainer.resizeDeviceBuffer(buffers.scales_opacs, 4 * alloc_cap);
        }
        trainer.copyToDevice(buffers.xyz_ws);
        trainer.copyToDevice(buffers.sh_coeffs);
        trainer.copyToDevice(buffers.rotations);
        trainer.copyToDevice(buffers.scales_opacs);
        trainer.bumpMapGeneration();
    }

    void set_pixel_state(
        py::array_t<float> pixel_state,
        py::array_t<int> n_contributors
    ) {
        auto pixel_state_buf = pixel_state.request();
        if (pixel_state_buf.ndim != 3 ||
            pixel_state_buf.shape[0] != uniforms.image_height ||
            pixel_state_buf.shape[1] != uniforms.image_width ||
            pixel_state_buf.shape[2] != 4)
            throw std::runtime_error("pixel_state must be (H, W, 4) array");

        buffers.pixel_state.assign((float*)(pixel_state_buf.ptr), (float*)(pixel_state_buf.ptr) + pixel_state_buf.size);
        trainer.copyToDevice(buffers.pixel_state);

        auto n_contributors_buf = n_contributors.request();
        if (n_contributors_buf.ndim != 3 ||
            n_contributors_buf.shape[0] != uniforms.image_height ||
            n_contributors_buf.shape[1] != uniforms.image_width ||
            n_contributors_buf.shape[2] != 1)
            throw std::runtime_error("n_contributors must be (H, W, 1) array");

        buffers.n_contributors.assign((int32_t*)(n_contributors_buf.ptr), (int32_t*)(n_contributors_buf.ptr) + n_contributors_buf.size);
        trainer.copyToDevice(buffers.n_contributors);
    }

    void set_pixel_state_grad(
        py::array_t<float> v_pixel_state
    ) {
        auto v_pixel_state_buf = v_pixel_state.request();
        if (v_pixel_state_buf.ndim != 3 ||
            v_pixel_state_buf.shape[0] != uniforms.image_height ||
            v_pixel_state_buf.shape[1] != uniforms.image_width ||
            v_pixel_state_buf.shape[2] != 4)
            throw std::runtime_error("v_pixel_state must be (H, W, 4) array");

        buffers.v_pixel_state.assign((float*)(v_pixel_state_buf.ptr), (float*)(v_pixel_state_buf.ptr) + v_pixel_state_buf.size);
        trainer.copyToDevice(buffers.v_pixel_state);
    }

    void projection_forward() {
        uniforms.num_splats = (uint32_t)buffers.num_splats;
        try {
            size_t alloc_reserve = config.strategy == TrainerConfig::Strategy::MCMC ?
                config.mcmc_alloc_cap() : 0;
            trainer.executeProjectionForward(uniforms, buffers, alloc_reserve);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeProjectionForward failed");
        }
    }
    
    void process_tiles() {
        auto deviceGuard = DeviceGuard(&trainer);

        // calculate index buffer offset (prefix sum)
        try {
            trainer.executeCalculateIndexBufferOffset(buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
        }
        if (buffers.num_indices == 0)
            return;

        // generate keys
        try {
            trainer.executeGenerateKeys(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
        }

        // sorting
        try {
            trainer.executeSort(uniforms, buffers, -1);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeSort failed");
        }

        // compute tile ranges
        try {
            trainer.executeComputeTileRanges(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
        }

    }
    
    void rasterize_forward() {
        try {
            if (uniforms.depth_loss_scale > 0.0f) {
                trainer.executeRasterizeForwardDepth(uniforms, buffers);
            } else {
                trainer.executeRasterizeForward(uniforms, buffers);
            }
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
        }
    }
    
    void rasterize_backward() {
        set_freeze_flags();
        try {
            trainer.executeRasterizeBackward(uniforms, buffers, current_train_depth_buf);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
        }
    }

    void forward() {
        auto deviceGuard = DeviceGuard(&trainer);
        projection_forward();
        process_tiles();
        rasterize_forward();
    }

    void backward_optimize(int step) {
        auto deviceGuard = DeviceGuard(&trainer);

        rasterize_backward();

        try {
            trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step+1);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeFusedProjectionBackwardOptimizerStep failed");
        }

    }

    void compute_pixel_loss_grad(size_t train_idx, int step) {
        auto deviceGuard = DeviceGuard(&trainer);
        if (!trainer.gateActive() && buffers.num_indices == 0)
            return;
        try {
            trainer.executeComputeSSIMGradient(config, uniforms, buffers, train_idx, step);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeSSIMGradient failed");
        }
    }

    void post_backward_step(int step) {
        if (config.external_strategy_mode) {
            return;
        }
        if (config.strategy == TrainerConfig::Strategy::Default) {
            try {
                trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
            } catch (const std::runtime_error& err) {
                _THROW_ERROR(std::string(err.what()) + ". trainer.executeDefaultPostBackward failed");
            }
        }
        else if (config.strategy == TrainerConfig::Strategy::MCMC) {
            try {
                trainer.executeMCMCPostBackward(config, uniforms, buffers, step);
            } catch (const std::runtime_error& err) {
                _THROW_ERROR(std::string(err.what()) + ". trainer.executeMCMCPostBackward failed");
            }
        }
    }

    py::dict get_train_image_path(size_t idx) {
        auto image = trainer.get_train_image(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty())
            result["mask_path"] = image.mask_path;
        return result;
    }

    py::dict get_val_image_path(size_t idx) {
        auto image = trainer.get_val_image(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty())
            result["mask_path"] = image.mask_path;
        return result;
    }

    void set_train_image(size_t train_idx) {
        trainer.get_train_camera(train_idx, uniforms);
        uniforms.use_tile_list = 0;
        current_train_depth_buf = nullptr;
    }

    // Tells the raster backward which frozen groups exist so that it skips their gradient atomics. The group
    // table is a small host mirror; the device tables are made resident here (inside the open batch if any).
    void set_freeze_flags() {
        uint32_t flags = 0;
        if (config.freeze_skip_backward) {
            for (uint32_t state : buffers.group_state) {
                if (state == 1u) flags |= 1u;
                else if (state == 2u && config.mcmc_protect_structural_groups) flags |= 2u;
                else if (state == 3u) flags |= 2u;
            }
            if (!trainer.get_splat_states().empty())
                flags |= 4u;
        }
        uniforms.freeze_flags = flags;
        uniforms.num_groups = (uint32_t)buffers.group_state.size();
        if (flags) {
            auto deviceGuard = DeviceGuard(&trainer);
            trainer.ensureGroupBuffersOnDevice(buffers, buffers.num_splats);
            uniforms.num_groups = (uint32_t)buffers.group_state.deviceSize();
        }
    }

    // per-step uniforms shared by the synchronous and the gated path
    void begin_step(size_t train_idx, int step) {
        set_train_image(train_idx);
        const int kShDegreeInterval = 1000;
        uniforms.active_sh = (config.max_sh_degree_override >= 0)
            ? std::min(step / kShDegreeInterval, config.max_sh_degree_override)
            : std::min(step / kShDegreeInterval, 3);
        uniforms.step = step;
        // sentinel: every sparse_full_every-th step is dense so converged tiles are re-verified
        uniforms.use_tile_list = config.sparse_tile_quantile > 0.0f && config.sparse_full_every > 1 &&
                                 step % config.sparse_full_every != 0;
        uniforms.ago_telemetry_enabled = trainer.get_ago_telemetry_enabled() ? 1u : 0u;
        if (trainer.frame_has_depth_loss(train_idx)) {
            const float n_pix = static_cast<float>(std::max<uint32_t>(1u, uniforms.image_height * uniforms.image_width));
            uniforms.depth_loss_scale = trainer.get_depth_loss_weight() / n_pix;
            uniforms.depth_huber_delta = trainer.get_depth_huber_delta();
            uniforms.depth_inv_scale = 1.0f / trainer.get_world_to_normalized_scale();
            current_train_depth_buf = trainer.get_train_depth_buffer(train_idx);
        } else {
            uniforms.depth_loss_scale = 0.0f;
            uniforms.depth_huber_delta = 0.15f;
            uniforms.depth_inv_scale = 1.0f;
            current_train_depth_buf = nullptr;
        }
    }

    // Steps with host-visible work (densify, prune, reset, Morton sort) and calls made inside an open
    // command batch: the intersection count is read back mid-forward (one extra fence per step).
    void train_step_sync(size_t train_idx, int step) {
        begin_step(train_idx, step);

        auto deviceGuard = DeviceGuard(&trainer);

        forward();
        compute_pixel_loss_grad(train_idx, step);
        backward_optimize(step);

        // post_backward_step(step+1);
        post_backward_step(step);
    }

    // One step recorded into the open gated batch: no host readback and no fence; see step_gate.slang.
    void train_step_gated(size_t train_idx, int step, uint32_t step_in_batch) {
        begin_step(train_idx, step);
        trainer.gateBeginStep(step_in_batch);
        projection_forward();
        const int backward_impl = trainer.sampleBackwardImpl();
        trainer.executeTilesDeferred(uniforms, buffers, trainer.buildGatePlan(uniforms, buffers, backward_impl));
        rasterize_forward();
        compute_pixel_loss_grad(train_idx, step);
        backward_optimize(step);
        post_backward_step(step);
    }

    // Runs `count` consecutive deferrable steps starting at indices[begin] in as few submits as possible
    // (one, unless a descriptor/buffer change forces a split or a step overflows the intersection buffers).
    // An overflowing step and every later step of the batch are skipped on the GPU without touching the
    // parameters or optimizer state; the buffers are grown and the skipped steps are recorded again, so the
    // result is exactly that of running the steps one by one.
    void train_steps_gated(const std::vector<int>& indices, int first_step, size_t begin, size_t count) {
        size_t done = 0;
        for (int attempt = 0; done < count; ++attempt) {
            if (attempt > 16)
                _THROW_ERROR_ALWAYS("gated training steps keep overflowing the intersection buffers");
            const size_t remaining = count - done;
            if (!isect_capacity_forced)
                trainer.ensureIsectCapacity(buffers, std::max<size_t>(
                    {size_t(1) << 18, 16 * buffers.num_splats, last_isects + (last_isects >> 1)}));
            isect_capacity_forced = false;
            {
                auto deviceGuard = DeviceGuard(&trainer);
                trainer.gateBeginBatch();
                try {
                    for (size_t j = 0; j < remaining; ++j)
                        train_step_gated(
                            (size_t)indices[begin + done + j], first_step + (int)(begin + done + j), (uint32_t)j);
                } catch (...) {
                    trainer.gateEndBatch();
                    throw;
                }
                trainer.gateEndBatch();
            }  // submit + the one fence of the batch
            const auto result = trainer.gateResult();
            auto& w = PerfTimer::work_counters;
            for (uint32_t i = 0; i < result.steps_done; ++i) {
                w.forward_passes++;
                w.tile_intersections_total += result.isects[i];
                w.tile_intersections_last = result.isects[i];
                w.tile_intersections_max = std::max<uint64_t>(w.tile_intersections_max, result.isects[i]);
                last_isects = result.isects[i];
            }
            if (result.steps_done > 0)
                buffers.num_indices = last_isects;
            done += result.steps_done;
            if (result.overflow) {
                w.gate_overflows++;
                w.gate_steps_redone += remaining - result.steps_done;
                trainer.ensureIsectCapacity(buffers, result.needed + (result.needed >> 1));
            }
            else if (result.steps_done != remaining) {
                trainer.waitIdle();
                const auto retry_res = trainer.gateResult();
                if (retry_res.overflow) {
                    w.gate_overflows++;
                    w.gate_steps_redone += remaining - retry_res.steps_done;
                    done += retry_res.steps_done - result.steps_done;
                    trainer.ensureIsectCapacity(buffers, retry_res.needed + (retry_res.needed >> 1));
                } else if (retry_res.steps_done == remaining) {
                    for (uint32_t i = result.steps_done; i < retry_res.steps_done; ++i) {
                        w.forward_passes++;
                        w.tile_intersections_total += retry_res.isects[i];
                        w.tile_intersections_last = retry_res.isects[i];
                        w.tile_intersections_max = std::max<uint64_t>(w.tile_intersections_max, retry_res.isects[i]);
                        last_isects = retry_res.isects[i];
                    }
                    if (retry_res.steps_done > 0)
                        buffers.num_indices = last_isects;
                    done += retry_res.steps_done - result.steps_done;
                } else {
                    _THROW_ERROR_ALWAYS(
                        "gated batch stopped without an overflow (steps_done=" +
                        std::to_string(retry_res.steps_done) + "/" + std::to_string(remaining) + ")"
                    );
                }
            }
        }
    }

    void train_steps(const std::vector<int>& indices, int first_step) {
        if (config.sparse_tile_quantile > 0.0f && !indices.empty()) {
            size_t tiles = 0;
            for (int idx : indices)
                tiles = std::max(tiles, trainer.train_tile_count((size_t)idx));
            trainer.ensureSparseTileBuffers(tiles);
        }
        size_t i = 0;
        while (i < indices.size()) {
            const int step = first_step + (int)i;
            if (buffers.num_splats == 0 || trainer.isCommandBatchInProgress() || !trainer.stepIsDeferrable(config, step)) {
                train_step_sync((size_t)indices[i], step);
                ++i;
                continue;
            }
            size_t count = 1;
            while (i + count < indices.size() && count < VulkanGSTrainer::kGateMaxSteps &&
                   trainer.stepIsDeferrable(config, step + (int)count))
                ++count;
            train_steps_gated(indices, first_step, i, count);
            i += count;
        }
    }

    void train_step(size_t train_idx, int step) {
        train_steps({(int)train_idx}, step);
    }

    // Test hook: shrink the intersection buffers to `capacity` so the next gated batch overflows.
    void force_isect_capacity(size_t capacity) {
        trainer.setIsectCapacityExact(buffers, capacity);
        isect_capacity_forced = true;
    }

    void render_train(size_t idx) {
        trainer.get_train_camera(idx, uniforms);
        forward();
    }

    void render_val(size_t idx) {
        trainer.get_val_camera(idx, uniforms);
        forward();
    }

    py::dict forward_rgbd_dict() {
        const size_t h = uniforms.image_height;
        const size_t w = uniforms.image_width;
        const size_t num_pixels = h * w;

        py::array_t<float> depth_expected({h, w});
        py::array_t<float> depth_first_surface({h, w});
        py::array_t<float> alpha({h, w});
        py::array_t<float> rgb({h, w, static_cast<size_t>(3)});

        float* d_exp_ptr = depth_expected.mutable_data();
        float* d_first_ptr = depth_first_surface.mutable_data();
        float* alpha_ptr = alpha.mutable_data();
        float* rgb_ptr = rgb.mutable_data();

        if (buffers.num_splats == 0 || num_pixels == 0) {
            std::fill(d_exp_ptr, d_exp_ptr + num_pixels, 0.0f);
            std::fill(d_first_ptr, d_first_ptr + num_pixels, 0.0f);
            std::fill(alpha_ptr, alpha_ptr + num_pixels, 0.0f);
            std::fill(rgb_ptr, rgb_ptr + 3 * num_pixels, 0.0f);
        } else {
            {
                py::gil_scoped_release release;
                {
                    auto deviceGuard = DeviceGuard(&trainer);
                    projection_forward();
                    process_tiles();
                    trainer.executeRasterizeForwardDepth(uniforms, buffers);
                }
                trainer.copyFromDevice(buffers.pixel_state);
                trainer.copyFromDevice(buffers.pixel_depth);
            }
            // Convert normalized-space camera Z depth into world-space camera Z units matching c2w.
            const float inv_scale = 1.0f / trainer.get_world_to_normalized_scale();
            const float* ps = buffers.pixel_state.data();
            const float* pd = buffers.pixel_depth.data();
            for (size_t i = 0; i < num_pixels; ++i) {
                d_exp_ptr[i] = pd[4 * i + 0] * inv_scale;
                d_first_ptr[i] = pd[4 * i + 1] * inv_scale;
                alpha_ptr[i] = pd[4 * i + 2];
                rgb_ptr[3 * i + 0] = ps[4 * i + 0];
                rgb_ptr[3 * i + 1] = ps[4 * i + 1];
                rgb_ptr[3 * i + 2] = ps[4 * i + 2];
            }
        }

        py::dict out;
        out["depth_expected"] = depth_expected;
        out["depth_first_surface"] = depth_first_surface;
        out["alpha"] = alpha;
        out["rgb"] = rgb;
        out["depth_eval_source"] = "native_rendered";
        return out;
    }

    py::dict render_val_depth(size_t idx) {
        if (idx >= trainer.num_val()) {
            throw std::out_of_range("render_val_depth: val index out of range");
        }
        trainer.get_val_camera(idx, uniforms);
        return forward_rgbd_dict();
    }

    py::dict render_pose_rgbd(
        py::array_t<float, py::array::c_style | py::array::forcecast> c2w,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height
    ) {
        auto buf = c2w.request();
        if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
            throw std::invalid_argument("render_pose_rgbd: c2w must be a (4, 4) float32 array");
        }
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("render_pose_rgbd: width and height must be positive");
        }
        trainer.pose_to_uniforms(height, width, fx, fy, cx, cy, static_cast<const float*>(buf.ptr), uniforms);
        return forward_rgbd_dict();
    }

    size_t get_vram_usage() {
        return buffers.getTotalAllocSize();
    }

    size_t get_peak_vram_usage() {
        return trainer.getPeakAllocSize();
    }

    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown() {
        return PerfTimer::get_summary();
    }

    py::dict get_gpu_stage_timestamps_ms() {
        auto ms = [](PerfTimer::TrainStage s) -> double {
            return PerfTimer::get_stage_total_time_s(s) * 1000.0;
        };
        const double forward_ms =
            ms(PerfTimer::ProjectionForward) +
            ms(PerfTimer::CalculateIndexBufferOffset) +
            ms(PerfTimer::GenerateKeys) +
            ms(PerfTimer::SortRTS) +
            ms(PerfTimer::ComputeTileRanges) +
            ms(PerfTimer::RasterizeForward);
        const double backward_ms =
            ms(PerfTimer::CopyTrainImageToDevice) +
            ms(PerfTimer::ComputeSSIMGradient) +
            ms(PerfTimer::SelectSparseTiles) +
            ms(PerfTimer::RasterizeBackward);
        const double optimizer_ms =
            ms(PerfTimer::FusedProjectionBackwardOptimizerStep) +
            ms(PerfTimer::DefaultPostBackward) +
            ms(PerfTimer::MCMCPostBackward) +
            ms(PerfTimer::MortonSorting);
        const double rayguard_ms = ms(PerfTimer::RayGuard);
        const double appear_init_ms = ms(PerfTimer::AppearanceInit);
        const double total_gpu_ms = forward_ms + backward_ms + optimizer_ms + rayguard_ms + appear_init_ms;

        py::dict d;
        d["forward_ms"] = forward_ms;
        d["backward_ms"] = backward_ms;
        d["optimizer_ms"] = optimizer_ms;
        d["rayguard_ms"] = rayguard_ms;
        d["appear_init_ms"] = appear_init_ms;
        d["total_gpu_ms"] = total_gpu_ms;
        d["timestamp_queries_enabled"] = trainer.isTimestampQueriesEnabled();
        return d;
    }

    void reset_gpu_stage_timestamps() {
        PerfTimer::reset_stage_timers();
    }

    py::dict compact_mode0_active_set() {
        const size_t n_total = buffers.num_splats;
        const auto& splat_sts = trainer.get_splat_states();
        const auto& gids = trainer.get_splat_group_ids();
        const auto& gstate = buffers.group_state;
        const bool protect_locked = config.mcmc_protect_structural_groups;

        std::vector<uint32_t> active_indices;
        active_indices.reserve(n_total);
        size_t n_frozen_mode0 = 0;
        for (size_t i = 0; i < n_total; ++i) {
            const uint32_t raw_st = (i < splat_sts.size()) ? splat_sts[i] : 0xFFFFFFFFu;
            const uint32_t base_st = (raw_st == 0xFFFFFFFFu) ? 0xFFFFFFFFu : (raw_st & 0xFFFFu);
            const int32_t gid = (i < gids.size()) ? gids[i] : -1;
            const uint32_t gs = (gid >= 0 && static_cast<size_t>(gid) < gstate.size()) ? gstate[gid] : 0u;
            bool frozen = false;
            if (gs == 3u) {
                frozen = true;
            } else if (base_st == 4u) {
                frozen = true;
            } else if (base_st == 0xFFFFFFFFu && gs == 2u && protect_locked) {
                frozen = true;
            }
            if (frozen) {
                ++n_frozen_mode0;
            } else {
                active_indices.push_back(static_cast<uint32_t>(i));
            }
        }
        const size_t n_active_trainable = active_indices.size();
        const double active_fraction = (n_total > 0)
            ? (static_cast<double>(n_active_trainable) / static_cast<double>(n_total))
            : 0.0;

        trainer.set_active_indices_gpu(active_indices, n_total);
        if (n_total > 0) {
            set_freeze_flags();
        }
        const bool gpu_active_buffer_bound = trainer.get_use_active_index_dispatch();
        const size_t active_dispatch_splats = gpu_active_buffer_bound
            ? static_cast<size_t>(trainer.get_active_splat_count())
            : n_total;

        py::dict d;
        d["n_total"] = n_total;
        d["n_frozen_mode0"] = n_frozen_mode0;
        d["n_active_trainable"] = n_active_trainable;
        d["total_splats"] = n_total;
        d["frozen_splats"] = n_frozen_mode0;
        d["active_splats"] = n_active_trainable;
        d["active_dispatch_splats"] = active_dispatch_splats;
        d["active_fraction"] = active_fraction;
        d["gpu_active_buffer_bound"] = gpu_active_buffer_bound;
        d["map_generation"] = trainer.get_map_generation();
        d["active_set_generation"] = trainer.get_active_set_generation();
        return d;
    }

    uint64_t get_map_generation() const {
        return trainer.get_map_generation();
    }

    uint64_t get_active_set_generation() const {
        return trainer.get_active_set_generation();
    }

    static void extract_k_params(const py::buffer_info& b, size_t v, float out[4], const char* name) {
        const float* p = static_cast<const float*>(b.ptr);
        if (b.ndim == 3 && b.shape[1] == 3 && b.shape[2] == 3) {
            const float* kv = p + 9 * v;
            out[0] = kv[0];  // fx
            out[1] = kv[4];  // fy
            out[2] = kv[2];  // cx
            out[3] = kv[5];  // cy
        } else if (b.ndim == 2 && b.shape[1] == 4) {
            const float* kv = p + 4 * v;
            out[0] = kv[0];
            out[1] = kv[1];
            out[2] = kv[2];
            out[3] = kv[3];
        } else {
            throw std::invalid_argument(std::string(name) + " must have shape (V, 3, 3) or (V, 4)");
        }
    }

    static std::vector<float> parse_fallback_rgb(const py::object& fallback_rgb_obj, size_t n_points) {
        std::vector<float> fallback_vec(3 * n_points, 0.5f);
        if (!fallback_rgb_obj.is_none()) {
            auto fb_arr = fallback_rgb_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto fbuf = fb_arr.request();
            const float* fptr = static_cast<const float*>(fbuf.ptr);
            if (fbuf.ndim == 1 && fbuf.shape[0] == 3) {
                for (size_t i = 0; i < n_points; ++i) {
                    fallback_vec[3 * i + 0] = std::clamp(fptr[0], 0.0f, 1.0f);
                    fallback_vec[3 * i + 1] = std::clamp(fptr[1], 0.0f, 1.0f);
                    fallback_vec[3 * i + 2] = std::clamp(fptr[2], 0.0f, 1.0f);
                }
            } else if (fbuf.ndim == 2 && static_cast<size_t>(fbuf.shape[0]) == n_points && fbuf.shape[1] == 3) {
                for (size_t i = 0; i < 3 * n_points; ++i)
                    fallback_vec[i] = std::clamp(fptr[i], 0.0f, 1.0f);
            } else {
                throw std::invalid_argument("fallback_rgb must have shape (3,) or (N, 3)");
            }
        }
        return fallback_vec;
    }

    py::dict project_and_fit_sh_vulkan(
        py::array_t<float, py::array::c_style | py::array::forcecast> means_world,
        py::object normals_world_obj,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb_views_u8,
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_views_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> k_rgb_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> k_depth_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> w2c_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> cam_o_f32,
        py::object fallback_rgb_obj = py::none(),
        int sh_degree = 1,
        float reg_lambda = 0.25f,
        float depth_tol_m = 0.12f,
        float min_cos_angle = 0.15f
    ) {
        auto mbuf = means_world.request();
        if (mbuf.ndim != 2 || mbuf.shape[1] != 3)
            throw std::invalid_argument("means_world must have shape (N, 3)");
        const size_t n_points = static_cast<size_t>(mbuf.shape[0]);

        py::array_t<float, py::array::c_style | py::array::forcecast> normals_arr;
        const float* normals_ptr = nullptr;
        if (!normals_world_obj.is_none()) {
            normals_arr = normals_world_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto nbuf = normals_arr.request();
            if (nbuf.ndim != 2 || static_cast<size_t>(nbuf.shape[0]) != n_points || nbuf.shape[1] != 3)
                throw std::invalid_argument("normals_world must have shape (N, 3)");
            normals_ptr = static_cast<const float*>(nbuf.ptr);
        }

        std::vector<float> fallback_vec = parse_fallback_rgb(fallback_rgb_obj, n_points);

        auto rbuf = rgb_views_u8.request();
        auto dbuf = depth_views_f32.request();
        if (rbuf.ndim != 4 || (rbuf.shape[3] != 3 && rbuf.shape[3] != 4))
            throw std::invalid_argument("rgb_views_u8 must have shape (V, H_rgb, W_rgb, 3 or 4)");
        if (dbuf.ndim != 3)
            throw std::invalid_argument("depth_views_f32 must have shape (V, H_d, W_d)");
        const size_t n_views = static_cast<size_t>(rbuf.shape[0]);
        if (static_cast<size_t>(dbuf.shape[0]) != n_views)
            throw std::invalid_argument("rgb_views_u8 and depth_views_f32 must have matching view count V");

        auto krbuf = k_rgb_f32.request();
        auto kdbuf = k_depth_f32.request();
        auto w2cbuf = w2c_f32.request();
        auto obuf = cam_o_f32.request();

        if (static_cast<size_t>(krbuf.shape[0]) != n_views ||
            static_cast<size_t>(kdbuf.shape[0]) != n_views ||
            static_cast<size_t>(w2cbuf.shape[0]) != n_views ||
            static_cast<size_t>(obuf.shape[0]) != n_views) {
            throw std::invalid_argument("Camera parameter arrays must all have leading dimension V");
        }
        if (obuf.ndim != 2 || obuf.shape[1] != 3)
            throw std::invalid_argument("cam_o_f32 must have shape (V, 3)");

        std::vector<float> view_params(24 * n_views, 0.0f);
        const float* w2c_ptr = static_cast<const float*>(w2cbuf.ptr);
        const float* o_ptr = static_cast<const float*>(obuf.ptr);
        const bool w2c_4x4 = (w2cbuf.ndim == 3 && w2cbuf.shape[1] == 4 && w2cbuf.shape[2] == 4);
        const bool w2c_3x4 = (w2cbuf.ndim == 3 && w2cbuf.shape[1] == 3 && w2cbuf.shape[2] == 4);
        if (!w2c_4x4 && !w2c_3x4)
            throw std::invalid_argument("w2c_f32 must have shape (V, 4, 4) or (V, 3, 4)");

        const size_t w2c_stride = w2c_4x4 ? 16 : 12;
        for (size_t v = 0; v < n_views; ++v) {
            float* dst = &view_params[24 * v];
            const float* wv = w2c_ptr + w2c_stride * v;
            std::memcpy(dst + 0, wv + 0, 4 * sizeof(float));
            std::memcpy(dst + 4, wv + 4, 4 * sizeof(float));
            std::memcpy(dst + 8, wv + 8, 4 * sizeof(float));
            dst[12] = o_ptr[3 * v + 0];
            dst[13] = o_ptr[3 * v + 1];
            dst[14] = o_ptr[3 * v + 2];
            dst[15] = 0.0f;
            extract_k_params(krbuf, v, dst + 16, "k_rgb_f32");
            extract_k_params(kdbuf, v, dst + 20, "k_depth_f32");
        }

        auto res = trainer.project_and_fit_sh_vulkan(
            static_cast<const float*>(mbuf.ptr),
            normals_ptr,
            n_points,
            static_cast<const uint8_t*>(rbuf.ptr),
            n_views,
            static_cast<int>(rbuf.shape[1]),
            static_cast<int>(rbuf.shape[2]),
            static_cast<int>(rbuf.shape[3]),
            static_cast<const float*>(dbuf.ptr),
            static_cast<int>(dbuf.shape[1]),
            static_cast<int>(dbuf.shape[2]),
            view_params.data(),
            fallback_vec.data(),
            sh_degree,
            reg_lambda,
            depth_tol_m,
            min_cos_angle
        );

        py::array_t<float> sh_out({(py::ssize_t)n_points, (py::ssize_t)16, (py::ssize_t)3});
        if (!res.sh_coeffs.empty())
            std::memcpy(sh_out.mutable_data(), res.sh_coeffs.data(), res.sh_coeffs.size() * sizeof(float));

        py::array_t<float> rgb_out({(py::ssize_t)n_points, (py::ssize_t)3});
        if (!res.rgb_init.empty())
            std::memcpy(rgb_out.mutable_data(), res.rgb_init.data(), res.rgb_init.size() * sizeof(float));

        py::array_t<int32_t> obs_out({(py::ssize_t)n_points});
        if (!res.obs_count.empty())
            std::memcpy(obs_out.mutable_data(), res.obs_count.data(), res.obs_count.size() * sizeof(int32_t));

        py::array_t<float> wsum_out({(py::ssize_t)n_points});
        if (!res.weight_sum.empty())
            std::memcpy(wsum_out.mutable_data(), res.weight_sum.data(), res.weight_sum.size() * sizeof(float));

        py::dict d;
        d["sh_coeffs"] = sh_out;
        d["rgb_init"] = rgb_out;
        d["obs_count"] = obs_out;
        d["weight_sum"] = wsum_out;
        d["wls_sh_fit_rmse"] = res.wls_sh_fit_rmse;
        d["kernel_wall_ms"] = res.kernel_wall_ms;
        d["total_wall_ms"] = res.total_wall_ms;
        d["backend"] = "vulkan_slang";
        return d;
    }

    py::dict append_and_init_surfels_vulkan(
        py::array_t<float, py::array::c_style | py::array::forcecast> means_world,
        py::object normals_world_obj,
        py::array_t<float, py::array::c_style | py::array::forcecast> scales_norm,
        py::array_t<float, py::array::c_style | py::array::forcecast> quats_norm,
        py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
        py::object group_ids_obj,
        py::object splat_states_obj,
        py::object fallback_rgb_obj,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb_views_u8,
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_views_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> k_rgbs_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> k_depths_f32,
        py::array_t<float, py::array::c_style | py::array::forcecast> poses_c2w_f32,
        py::object dp_transform_obj = py::none(),
        float sh_reg = 1e-2f,
        float depth_tol = 0.12f,
        int sh_degree = 1,
        float min_cos_angle = 0.15f
    ) {
        auto mbuf = means_world.request();
        if (mbuf.ndim != 2 || mbuf.shape[1] != 3)
            throw std::invalid_argument("means_world must have shape (M, 3)");
        const size_t n_add = static_cast<size_t>(mbuf.shape[0]);

        auto sbuf = scales_norm.request();
        auto qbuf = quats_norm.request();
        auto obuf = opacities.request();
        if (sbuf.ndim != 2 || static_cast<size_t>(sbuf.shape[0]) != n_add || sbuf.shape[1] != 3)
            throw std::invalid_argument("scales_norm must have shape (M, 3)");
        if (qbuf.ndim != 2 || static_cast<size_t>(qbuf.shape[0]) != n_add || qbuf.shape[1] != 4)
            throw std::invalid_argument("quats_norm must have shape (M, 4)");
        if (static_cast<size_t>(obuf.size) != n_add)
            throw std::invalid_argument("opacities must have M elements");

        py::array_t<float, py::array::c_style | py::array::forcecast> normals_arr;
        const float* normals_ptr = nullptr;
        if (!normals_world_obj.is_none()) {
            normals_arr = normals_world_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto nbuf = normals_arr.request();
            if (nbuf.ndim != 2 || static_cast<size_t>(nbuf.shape[0]) != n_add || nbuf.shape[1] != 3)
                throw std::invalid_argument("normals_world must have shape (M, 3)");
            normals_ptr = static_cast<const float*>(nbuf.ptr);
        }

        py::array_t<int32_t, py::array::c_style | py::array::forcecast> gid_arr;
        const int32_t* gid_ptr = nullptr;
        if (!group_ids_obj.is_none()) {
            gid_arr = group_ids_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            if (static_cast<size_t>(gid_arr.size()) == n_add)
                gid_ptr = gid_arr.data();
        }

        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> st_arr;
        const uint32_t* st_ptr = nullptr;
        if (!splat_states_obj.is_none()) {
            st_arr = splat_states_obj.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
            if (static_cast<size_t>(st_arr.size()) == n_add)
                st_ptr = st_arr.data();
        }

        std::vector<float> fallback_vec = parse_fallback_rgb(fallback_rgb_obj, n_add);

        // Compute normalized coordinates xyz_norm = dp_transform @ [means_world, 1]
        float dp_rm[12];
        if (dp_transform_obj.is_none()) {
            glm::mat4 dp = trainer.get_dataparser_transform();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    dp_rm[4 * r + c] = dp[c][r];
        } else {
            auto dp_arr = dp_transform_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto dpbuf = dp_arr.request();
            const bool is_4x4 = (dpbuf.ndim == 2 && dpbuf.shape[0] == 4 && dpbuf.shape[1] == 4);
            const bool is_3x4 = (dpbuf.ndim == 2 && dpbuf.shape[0] == 3 && dpbuf.shape[1] == 4);
            if (!is_4x4 && !is_3x4)
                throw std::invalid_argument("dp_transform must have shape (4, 4) or (3, 4)");
            std::memcpy(dp_rm, dpbuf.ptr, 12 * sizeof(float));
        }
        const float* mptr = static_cast<const float*>(mbuf.ptr);
        std::vector<float> xyz_norm(3 * n_add, 0.0f);
        for (size_t i = 0; i < n_add; ++i) {
            const float x = mptr[3 * i + 0];
            const float y = mptr[3 * i + 1];
            const float z = mptr[3 * i + 2];
            xyz_norm[3 * i + 0] = dp_rm[0] * x + dp_rm[1] * y + dp_rm[2] * z + dp_rm[3];
            xyz_norm[3 * i + 1] = dp_rm[4] * x + dp_rm[5] * y + dp_rm[6] * z + dp_rm[7];
            xyz_norm[3 * i + 2] = dp_rm[8] * x + dp_rm[9] * y + dp_rm[10] * z + dp_rm[11];
        }

        // Parse RGB-D views and c2w poses -> view_params (w2c + cam_o + K_rgb + K_depth)
        auto rbuf = rgb_views_u8.request();
        auto dbuf = depth_views_f32.request();
        const size_t n_views = (rbuf.ndim > 0) ? static_cast<size_t>(rbuf.shape[0]) : 0;
        int h_rgb = 0, w_rgb = 0, rgb_ch = 3, h_d = 0, w_d = 0;
        std::vector<float> view_params;
        if (n_views > 0) {
            if (rbuf.ndim != 4 || (rbuf.shape[3] != 3 && rbuf.shape[3] != 4))
                throw std::invalid_argument("rgb_views must have shape (V, H_rgb, W_rgb, 3 or 4)");
            if (dbuf.ndim != 3 || static_cast<size_t>(dbuf.shape[0]) != n_views)
                throw std::invalid_argument("depth_views must have shape (V, H_d, W_d)");
            h_rgb = static_cast<int>(rbuf.shape[1]);
            w_rgb = static_cast<int>(rbuf.shape[2]);
            rgb_ch = static_cast<int>(rbuf.shape[3]);
            h_d = static_cast<int>(dbuf.shape[1]);
            w_d = static_cast<int>(dbuf.shape[2]);

            auto krbuf = k_rgbs_f32.request();
            auto kdbuf = k_depths_f32.request();
            auto c2wbuf = poses_c2w_f32.request();
            if (static_cast<size_t>(krbuf.shape[0]) != n_views ||
                static_cast<size_t>(kdbuf.shape[0]) != n_views ||
                static_cast<size_t>(c2wbuf.shape[0]) != n_views) {
                throw std::invalid_argument("K_rgbs, K_depths, and poses_c2w must all have leading dimension V");
            }
            const bool c2w_4x4 = (c2wbuf.ndim == 3 && c2wbuf.shape[1] == 4 && c2wbuf.shape[2] == 4);
            const bool c2w_3x4 = (c2wbuf.ndim == 3 && c2wbuf.shape[1] == 3 && c2wbuf.shape[2] == 4);
            if (!c2w_4x4 && !c2w_3x4)
                throw std::invalid_argument("poses_c2w must have shape (V, 4, 4) or (V, 3, 4)");
            const size_t c2w_stride = c2w_4x4 ? 16 : 12;
            const float* c2w_ptr = static_cast<const float*>(c2wbuf.ptr);

            view_params.assign(24 * n_views, 0.0f);
            for (size_t v = 0; v < n_views; ++v) {
                float* dst = &view_params[24 * v];
                const float* cv = c2w_ptr + c2w_stride * v;
                glm::mat4 c2w_m(1.0f);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c)
                        c2w_m[c][r] = cv[4 * r + c];
                glm::mat4 w2c_m = glm::inverse(c2w_m);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c)
                        dst[4 * r + c] = w2c_m[c][r];
                dst[12] = cv[3];
                dst[13] = cv[7];
                dst[14] = cv[11];
                dst[15] = 0.0f;
                extract_k_params(krbuf, v, dst + 16, "K_rgbs");
                extract_k_params(kdbuf, v, dst + 20, "K_depths");
            }
        }

        auto res = trainer.append_and_init_surfels_vulkan(
            buffers,
            mptr,
            normals_ptr,
            xyz_norm.data(),
            static_cast<const float*>(sbuf.ptr),
            static_cast<const float*>(qbuf.ptr),
            static_cast<const float*>(obuf.ptr),
            gid_ptr,
            st_ptr,
            fallback_vec.data(),
            n_add,
            n_views > 0 ? static_cast<const uint8_t*>(rbuf.ptr) : nullptr,
            n_views,
            h_rgb,
            w_rgb,
            rgb_ch,
            n_views > 0 ? static_cast<const float*>(dbuf.ptr) : nullptr,
            h_d,
            w_d,
            n_views > 0 ? view_params.data() : nullptr,
            sh_degree,
            sh_reg,
            depth_tol,
            min_cos_angle
        );

        py::dict d;
        d["n_appended"] = res.n_appended;
        d["kernel_ms"] = res.kernel_ms;
        d["total_ms"] = res.total_ms;
        d["mean_wls_rmse"] = res.mean_wls_rmse;
        d["n_valid_sh"] = res.n_valid_sh;
        d["gpu_resident"] = true;
        return d;
    }

    std::map<std::string, uint64_t> get_work_counters() {
        // the device counter is read first so that its own fence is already part of fence_waits_total
        const uint64_t sparse_tiles = PerfTimer::work_counters.backward_sparse_passes > 0 ? trainer.sparseTilesDispatched() : 0;
        auto counters = PerfTimer::get_work_counters();
        counters["backward_tiles_dispatched"] += sparse_tiles;
        return counters;
    }

    std::map<std::string, size_t> get_vram_breakdown() {
        return buffers.getVramBreakdown();
    }

    py::dict get_gpu_memory_stats() {
        auto stats = trainer.getGpuMemoryStats(buffers.getTotalAllocSize());
        py::dict d;
        d["device_buffers_mb"] = stats.device_buffers_mb;
        d["peak_device_buffers_mb"] = stats.peak_device_buffers_mb;
        d["pipeline_buffers_mb"] = static_cast<double>(buffers.getTotalAllocSize()) / (1024.0 * 1024.0);
        d["staging_buffers_mb"] = stats.staging_buffers_mb;
        d["vram_used_mb"] = stats.vram_used_mb;
        d["vram_budget_mb"] = stats.vram_budget_mb;
        d["source"] = stats.source;
        return d;
    }

    void set_local_cell_credits(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> splat_cell_ids,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_credits
    ) {
        auto cid_buf = splat_cell_ids.request();
        auto cred_buf = cell_credits.request();
        if (cid_buf.ndim != 1)
            throw std::invalid_argument("splat_cell_ids must be a 1D int32 array");
        if (cred_buf.ndim != 1)
            throw std::invalid_argument("cell_credits must be a 1D int32 array");
        trainer.set_local_cell_credits(
            buffers,
            static_cast<const int32_t*>(cid_buf.ptr),
            static_cast<size_t>(cid_buf.size),
            static_cast<const int32_t*>(cred_buf.ptr),
            static_cast<size_t>(cred_buf.size)
        );
    }

    void clear_local_cell_credits() {
        trainer.clear_local_cell_credits();
    }

    py::array_t<int32_t> get_local_cell_credits() {
        std::vector<int32_t> creds = trainer.get_local_cell_credits(buffers);
        return py::array_t<int32_t>({creds.size()}, {sizeof(int32_t)}, creds.data());
    }

    py::array_t<int32_t> get_splat_cell_ids() {
        const auto& cids = trainer.get_splat_cell_ids();
        return py::array_t<int32_t>({cids.size()}, {sizeof(int32_t)}, cids.data());
    }

    size_t get_adc_host_readbacks() const {
        return trainer.get_adc_host_readbacks();
    }

    void write_ply(std::string filename) {
        trainer.writePLY(filename, buffers);
    }

    void cleanup() {
        trainer.cleanupBuffers(buffers);
        trainer.cleanup();
    }
};

#define DEF_BUFFER_0(key, buffer_name, dtype, ...) \
    def_property_readonly(#key, [] (VkSplat &self) { \
        return self.buffer_to_array<dtype, dtype>(self.buffers.buffer_name, __VA_ARGS__); \
    })

#define DEF_BUFFER(key, dtype, ...) \
    DEF_BUFFER_0(key, key, dtype, __VA_ARGS__)

#define DEF_ARRAY(key, dtype) \
    def("_set_"#key, [] (VkSplat &self, py::array_t<dtype> array) { \
        self.array_to_buffer(array, self.buffers.key); \
    }, "Directly set `"#key"` buffer (use at own risk)", py::arg(#key))

#define DEF_BUFFER_ARRAY(key, dtype, ...) \
    DEF_BUFFER(key, dtype, __VA_ARGS__) \
    .DEF_ARRAY(key, dtype)

PYBIND11_MODULE(vksplat, m) {
    m.doc() = "Vulkan Gaussian Rasterization Python Bindings";

    // -------------------------------------------------------------------------
    // Native C++ Non-Blocking AGO Control Plane Bindings (P1 + P3)
    // Registered before VkSplat so default arguments (e.g. SchedulerConfig) convert cleanly.
    // -------------------------------------------------------------------------
    {
        using namespace vksplat::ago;

        py::enum_<ServiceClass>(m, "AgoServiceClass")
            .value("P0_CONSISTENCY", ServiceClass::P0_CONSISTENCY)
            .value("P1_LIVE_MAP", ServiceClass::P1_LIVE_MAP)
            .value("P2_REFINEMENT", ServiceClass::P2_REFINEMENT)
            .value("P3_PREDICTIVE", ServiceClass::P3_PREDICTIVE)
            .value("P4_MAINTENANCE", ServiceClass::P4_MAINTENANCE);

        py::enum_<ActiveSetState>(m, "AgoActiveSetState")
            .value("EMPTY", ActiveSetState::EMPTY)
            .value("PARTIAL", ActiveSetState::PARTIAL)
            .value("FULL", ActiveSetState::FULL)
            .value("INVALID", ActiveSetState::INVALID);

        py::enum_<OperationType>(m, "AgoOperationType")
            .value("REFINE", OperationType::REFINE)
            .value("FREEZE", OperationType::FREEZE)
            .value("REOPEN", OperationType::REOPEN)
            .value("INSERT", OperationType::INSERT)
            .value("CLONE", OperationType::CLONE)
            .value("SPLIT", OperationType::SPLIT)
            .value("MERGE", OperationType::MERGE)
            .value("PRUNE", OperationType::PRUNE)
            .value("RELOCATE", OperationType::RELOCATE)
            .value("REANCHOR", OperationType::REANCHOR)
            .value("RETRACT", OperationType::RETRACT)
            .value("TRANSFORM", OperationType::TRANSFORM)
            .value("PARTITION_MOVE", OperationType::PARTITION_MOVE)
            .value("LOD_CHANGE", OperationType::LOD_CHANGE)
            .value("SERIALIZE", OperationType::SERIALIZE)
            .value("RESTORE", OperationType::RESTORE)
            .value("PREFETCH", OperationType::PREFETCH)
            .value("EVICT", OperationType::EVICT);

        py::enum_<EvidenceKind>(m, "AgoEvidenceKind")
            .value("FRAME_OBSERVATION", EvidenceKind::FRAME_OBSERVATION)
            .value("POSE_REVISION", EvidenceKind::POSE_REVISION)
            .value("CONTRADICTION", EvidenceKind::CONTRADICTION)
            .value("COVERAGE_UPDATE", EvidenceKind::COVERAGE_UPDATE)
            .value("TELEMETRY_SUMMARY", EvidenceKind::TELEMETRY_SUMMARY)
            .value("STRUCTURAL_COMPLETION", EvidenceKind::STRUCTURAL_COMPLETION)
            .value("DEADLINE_BUDGET_UPDATE", EvidenceKind::DEADLINE_BUDGET_UPDATE)
            .value("SHUTDOWN_SIGNAL", EvidenceKind::SHUTDOWN_SIGNAL);

        py::enum_<CompletionStatus>(m, "AgoCompletionStatus")
            .value("OK", CompletionStatus::OK)
            .value("REJECTED_STALE", CompletionStatus::REJECTED_STALE)
            .value("REJECTED_NO_CREDITS", CompletionStatus::REJECTED_NO_CREDITS)
            .value("REJECTED_UNSUPPORTED", CompletionStatus::REJECTED_UNSUPPORTED)
            .value("REJECTED_INVALID", CompletionStatus::REJECTED_INVALID)
            .value("FAILED_EXECUTION", CompletionStatus::FAILED_EXECUTION)
            .value("CANCELLED", CompletionStatus::CANCELLED);

        py::enum_<SubmitResult>(m, "AgoSubmitResult")
            .value("OK", SubmitResult::OK)
            .value("NO_CREDITS", SubmitResult::NO_CREDITS)
            .value("INVALID_ACTION", SubmitResult::INVALID_ACTION)
            .value("UNSUPPORTED_OP", SubmitResult::UNSUPPORTED_OP)
            .value("DEVICE_ERROR", SubmitResult::DEVICE_ERROR)
            .value("REJECTED_NO_EXECUTOR", SubmitResult::REJECTED_NO_EXECUTOR);

        m.attr("AGO_PARAM_POS")   = TrainerConfig::AGO_PARAM_POS;
        m.attr("AGO_PARAM_ROT")   = TrainerConfig::AGO_PARAM_ROT;
        m.attr("AGO_PARAM_SCALE") = TrainerConfig::AGO_PARAM_SCALE;
        m.attr("AGO_PARAM_OPAC")  = TrainerConfig::AGO_PARAM_OPAC;
        m.attr("AGO_PARAM_SH")    = TrainerConfig::AGO_PARAM_SH;
        m.attr("AGO_PARAM_ALL")   = TrainerConfig::AGO_PARAM_ALL;

        py::enum_<TraceEventType>(m, "AgoTraceEventType")
            .value("EVIDENCE_CRITICAL", TraceEventType::EVIDENCE_CRITICAL)
            .value("EVIDENCE_TELEMETRY", TraceEventType::EVIDENCE_TELEMETRY)
            .value("COMPLETION_IN", TraceEventType::COMPLETION_IN)
            .value("ACTION_ADMITTED", TraceEventType::ACTION_ADMITTED)
            .value("ACTION_REJECTED_STALE", TraceEventType::ACTION_REJECTED_STALE)
            .value("ACTION_REJECTED_BUDGET", TraceEventType::ACTION_REJECTED_BUDGET)
            .value("ACTION_REJECTED_CREDITS", TraceEventType::ACTION_REJECTED_CREDITS)
            .value("ACTION_REJECTED_UNSUPPORTED", TraceEventType::ACTION_REJECTED_UNSUPPORTED)
            .value("ACTION_SUBMIT_FAILED", TraceEventType::ACTION_SUBMIT_FAILED)
            .value("TICK_SUMMARY", TraceEventType::TICK_SUMMARY);

        py::class_<EvidenceMessage>(m, "AgoEvidenceMessage")
            .def(py::init<>())
            .def_readwrite("kind", &EvidenceMessage::kind)
            .def_readwrite("region_id", &EvidenceMessage::region_id)
            .def_readwrite("map_version", &EvidenceMessage::map_version)
            .def_readwrite("pose_version", &EvidenceMessage::pose_version)
            .def_readwrite("evidence_version", &EvidenceMessage::evidence_version)
            .def_readwrite("timestamp_ms", &EvidenceMessage::timestamp_ms)
            .def_readwrite("magnitude", &EvidenceMessage::magnitude)
            .def_readwrite("flags", &EvidenceMessage::flags)
            .def_readwrite("critical", &EvidenceMessage::critical);

        py::class_<AGOAction>(m, "AgoAction")
            .def(py::init<>())
            .def_readwrite("action_id", &AGOAction::action_id)
            .def_readwrite("region_id", &AGOAction::region_id)
            .def_readwrite("service_class", &AGOAction::service_class)
            .def_readwrite("op", &AGOAction::op)
            .def_readwrite("required_map_version", &AGOAction::required_map_version)
            .def_readwrite("required_pose_version", &AGOAction::required_pose_version)
            .def_readwrite("required_evidence_version", &AGOAction::required_evidence_version)
            .def_readwrite("expected_gain", &AGOAction::expected_gain)
            .def_readwrite("estimated_cost_ms", &AGOAction::estimated_cost_ms)
            .def_readwrite("max_splats", &AGOAction::max_splats)
            .def_readwrite("target_fraction", &AGOAction::target_fraction)
            .def_readwrite("requires_exclusive_region", &AGOAction::requires_exclusive_region)
            .def_readwrite("requires_gpu", &AGOAction::requires_gpu)
            .def_readwrite("enqueued_at_ms", &AGOAction::enqueued_at_ms)
            .def_readwrite("evidence_ingested_at_ms", &AGOAction::evidence_ingested_at_ms)
            .def_readwrite("deadline_ms", &AGOAction::deadline_ms);

        py::class_<CompletionRecord>(m, "AgoCompletionRecord")
            .def(py::init<>())
            .def_readonly("action_id", &CompletionRecord::action_id)
            .def_readonly("region_id", &CompletionRecord::region_id)
            .def_readonly("op", &CompletionRecord::op)
            .def_readonly("service_class", &CompletionRecord::service_class)
            .def_readonly("status", &CompletionRecord::status)
            .def_readonly("active_set_state", &CompletionRecord::active_set_state)
            .def_readonly("affected_splats", &CompletionRecord::affected_splats)
            .def_readonly("realized_gain", &CompletionRecord::realized_gain)
            .def_readonly("actual_cost_ms", &CompletionRecord::actual_cost_ms)
            .def_readonly("completed_at_ms", &CompletionRecord::completed_at_ms)
            .def_readonly("event_to_map_correction_ms", &CompletionRecord::event_to_map_correction_ms)
            .def_readonly("was_exclusive", &CompletionRecord::was_exclusive);

        py::class_<DeadlineBudget>(m, "AgoDeadlineBudget")
            .def(py::init<>())
            .def_readwrite("now_ms", &DeadlineBudget::now_ms)
            .def_readwrite("next_deadline_ms", &DeadlineBudget::next_deadline_ms)
            .def_readwrite("reserved_ms", &DeadlineBudget::reserved_ms)
            .def_readwrite("inflight_ms", &DeadlineBudget::inflight_ms)
            .def_readwrite("safety_margin_ms", &DeadlineBudget::safety_margin_ms)
            .def("available_ms", &DeadlineBudget::available_ms);

        py::class_<AdmissionConfig>(m, "AgoAdmissionConfig")
            .def(py::init<>())
            .def_readwrite("max_inflight_actions", &AdmissionConfig::max_inflight_actions)
            .def_readwrite("maintenance_floor_interval_ticks", &AdmissionConfig::maintenance_floor_interval_ticks)
            .def_readwrite("min_cost_epsilon_ms", &AdmissionConfig::min_cost_epsilon_ms);

        py::class_<SchedulerConfig>(m, "AgoSchedulerConfig")
            .def(py::init<>())
            .def_readwrite("max_completions_per_tick", &SchedulerConfig::max_completions_per_tick)
            .def_readwrite("max_critical_evidence_per_tick", &SchedulerConfig::max_critical_evidence_per_tick)
            .def_readwrite("max_telemetry_evidence_per_tick", &SchedulerConfig::max_telemetry_evidence_per_tick)
            .def_readwrite("max_admissions_per_tick", &SchedulerConfig::max_admissions_per_tick)
            .def_readwrite("max_pending_actions", &SchedulerConfig::max_pending_actions)
            .def_readwrite("critical_queue_capacity", &SchedulerConfig::critical_queue_capacity)
            .def_readwrite("telemetry_queue_capacity", &SchedulerConfig::telemetry_queue_capacity)
            .def_readwrite("max_regions", &SchedulerConfig::max_regions)
            .def_readwrite("trace_capacity", &SchedulerConfig::trace_capacity)
            .def_readwrite("admission", &SchedulerConfig::admission);

        py::class_<TickResult>(m, "AgoTickResult")
            .def_readonly("tick_index", &TickResult::tick_index)
            .def_readonly("completions_processed", &TickResult::completions_processed)
            .def_readonly("critical_evidence_processed", &TickResult::critical_evidence_processed)
            .def_readonly("telemetry_evidence_processed", &TickResult::telemetry_evidence_processed)
            .def_readonly("actions_admitted", &TickResult::actions_admitted)
            .def_readonly("stale_actions_pruned", &TickResult::stale_actions_pruned)
            .def_readonly("budget_rejections", &TickResult::budget_rejections)
            .def_readonly("credit_rejections", &TickResult::credit_rejections)
            .def_readonly("tick_duration_us", &TickResult::tick_duration_us);

        py::class_<SchedulerSummary>(m, "AgoSchedulerSummary")
            .def_readonly("tick_count", &SchedulerSummary::tick_count)
            .def_readonly("map_version", &SchedulerSummary::map_version)
            .def_readonly("pose_version", &SchedulerSummary::pose_version)
            .def_readonly("evidence_version", &SchedulerSummary::evidence_version)
            .def_readonly("inflight_count", &SchedulerSummary::inflight_count)
            .def_readonly("queued_actions", &SchedulerSummary::queued_actions)
            .def_readonly("dirty_regions", &SchedulerSummary::dirty_regions)
            .def_readonly("total_admitted", &SchedulerSummary::total_admitted)
            .def_readonly("total_completed", &SchedulerSummary::total_completed)
            .def_readonly("total_rejected_stale", &SchedulerSummary::total_rejected_stale)
            .def_readonly("total_rejected_budget", &SchedulerSummary::total_rejected_budget)
            .def_readonly("total_rejected_credits", &SchedulerSummary::total_rejected_credits)
            .def_readonly("total_rejected_unsupported", &SchedulerSummary::total_rejected_unsupported)
            .def_readonly("total_failed", &SchedulerSummary::total_failed)
            .def_readonly("total_dropped_telemetry", &SchedulerSummary::total_dropped_telemetry)
            .def_readonly("total_critical_backpressure", &SchedulerSummary::total_critical_backpressure)
            .def_readonly("last_tick_us", &SchedulerSummary::last_tick_us)
            .def_readonly("available_budget_ms", &SchedulerSummary::available_budget_ms)
            .def_readonly("last_event_to_map_correction_ms", &SchedulerSummary::last_event_to_map_correction_ms)
            .def_readonly("max_event_to_map_correction_ms", &SchedulerSummary::max_event_to_map_correction_ms)
            .def_readonly("mean_event_to_map_correction_ms", &SchedulerSummary::mean_event_to_map_correction_ms)
            .def_readonly("event_to_map_correction_samples", &SchedulerSummary::event_to_map_correction_samples);

        py::class_<EventTraceEntry>(m, "AgoEventTraceEntry")
            .def_readonly("tick", &EventTraceEntry::tick)
            .def_readonly("event_type", &EventTraceEntry::event_type)
            .def_readonly("action_id", &EventTraceEntry::action_id)
            .def_readonly("region_id", &EventTraceEntry::region_id)
            .def_readonly("service_class", &EventTraceEntry::service_class)
            .def_readonly("op", &EventTraceEntry::op)
            .def_readonly("status", &EventTraceEntry::status)
            .def_readonly("value", &EventTraceEntry::value);

        py::class_<NonBlockingGpuExecutor>(m, "NonBlockingGpuExecutor")
            .def(py::init<>())
            .def("is_initialized", &NonBlockingGpuExecutor::is_initialized)
            .def("available_credits", &NonBlockingGpuExecutor::available_credits)
            .def("inflight_count", &NonBlockingGpuExecutor::inflight_count)
            .def("try_submit", &NonBlockingGpuExecutor::try_submit, py::arg("action"))
            .def("shutdown", &NonBlockingGpuExecutor::shutdown, py::arg("drain_timeout_ms") = 5000u);

        py::class_<SimulatedAgoExecutor>(m, "SimulatedAgoExecutor")
            .def(py::init<uint32_t, uint32_t>(),
                 py::arg("ring_slots") = 4u,
                 py::arg("default_latency_ticks") = 1u)
            .def("advance_ticks", &SimulatedAgoExecutor::advance_ticks, py::arg("ticks") = 1u)
            .def("set_stalled", &SimulatedAgoExecutor::set_stalled, py::arg("stalled"))
            .def("set_fail_next_completion", &SimulatedAgoExecutor::set_fail_next_completion, py::arg("fail"))
            .def("set_unsupported_op_mask", &SimulatedAgoExecutor::set_unsupported_op_mask, py::arg("mask"))
            .def("set_default_latency_ticks", &SimulatedAgoExecutor::set_default_latency_ticks, py::arg("ticks"))
            .def("available_credits", &SimulatedAgoExecutor::available_credits)
            .def("inflight_count", &SimulatedAgoExecutor::inflight_count)
            .def("inflight_cost_ms", &SimulatedAgoExecutor::inflight_cost_ms)
            .def("total_submitted", &SimulatedAgoExecutor::total_submitted)
            .def("total_completed", &SimulatedAgoExecutor::total_completed)
            .def("submitted_order", &SimulatedAgoExecutor::submitted_order)
            .def("clear_history", &SimulatedAgoExecutor::clear_history);

        py::class_<AgoScheduler>(m, "AgoScheduler")
            .def(py::init([](const SchedulerConfig& cfg, SimulatedAgoExecutor* exec) {
                return std::make_unique<AgoScheduler>(cfg, exec);
            }), py::arg("config") = SchedulerConfig{}, py::arg("executor") = nullptr,
                py::keep_alive<1, 3>())
            .def("push_critical_evidence", &AgoScheduler::push_critical_evidence, py::arg("msg"))
            .def("push_telemetry_evidence", &AgoScheduler::push_telemetry_evidence, py::arg("msg"))
            .def("publish_deadline_budget", &AgoScheduler::publish_deadline_budget, py::arg("budget"))
            .def("enqueue_action", &AgoScheduler::enqueue_action, py::arg("action"))
            .def("tick", &AgoScheduler::tick)
            .def("try_consume_summary", [](AgoScheduler& self) -> py::object {
                SchedulerSummary out{};
                if (self.try_consume_summary(out)) {
                    return py::cast(out);
                }
                return py::none();
            })
            .def("latest_summary", &AgoScheduler::latest_summary)
            .def("trace_snapshot", [](const AgoScheduler& self) {
                std::vector<EventTraceEntry> out;
                self.event_trace().snapshot(out);
                return out;
            })
            .def("set_region_frozen", &AgoScheduler::set_region_frozen, py::arg("region_id"), py::arg("frozen"))
            .def("set_region_retracted", &AgoScheduler::set_region_retracted, py::arg("region_id"), py::arg("retracted"))
            .def("map_version", &AgoScheduler::map_version)
            .def("pose_version", &AgoScheduler::pose_version)
            .def("evidence_version", &AgoScheduler::evidence_version)
            .def("bump_map_version", &AgoScheduler::bump_map_version, py::arg("delta") = 1u)
            .def("bump_pose_version", &AgoScheduler::bump_pose_version, py::arg("delta") = 1u)
            .def("region_count", [](const AgoScheduler& self) { return self.dirty_tracker().region_count(); })
            .def("dirty_queue_size", [](const AgoScheduler& self) { return self.dirty_tracker().dirty_queue_size(); });
    }
    
    py::class_<VkSplat>(m, "VkSplat")
        .def(py::init<>())
        .def("initialize", &VkSplat::initialize, "Initialize the trainer with SPIRV code")
        .def("set_train_config", &VkSplat::set_train_config, "Set training configuration")
        .def_property_readonly("num_train", &VkSplat::num_train)
        .def_property_readonly("num_val", &VkSplat::num_val)
        .def("get_train_image_path", &VkSplat::get_train_image_path)
        .def("get_val_image_path", &VkSplat::get_val_image_path)
        .def("get_train_image", &VkSplat::get_train_image)
        .def("get_val_image", &VkSplat::get_val_image)
        .def("set_uniforms", &VkSplat::set_uniforms, "Set trainer uniforms (image/camera info)",
             py::arg("active_sh"), py::arg("world_view_transform"),
             py::arg("image_height"), py::arg("image_width"),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("is_fisheye"))
        .def("get_uniforms", &VkSplat::get_uniforms, "Get trainer uniforms as a dict")
        .def("set_gauss_params", &VkSplat::set_gauss_params, "Set Gaussian parameters",
             py::arg("xyz_ws"), py::arg("sh_coeffs"), py::arg("rotations"), py::arg("scales"), 
             py::arg("opacities"))
        .def("set_pixel_state", &VkSplat::set_pixel_state, "Set output pixels and n_contributors",
             py::arg("pixel_state"), py::arg("n_contributors"))
        .def("set_pixel_state_grad", &VkSplat::set_pixel_state_grad, "Set gradient of output pixels",
             py::arg("v_pixel_state"))
        .def("projection_forward", &VkSplat::projection_forward,
             "Execute the forward projection trainer", py::call_guard<py::gil_scoped_release>())
        .def("process_tiles", &VkSplat::process_tiles,
             "Process tiles", py::call_guard<py::gil_scoped_release>())
        .def("rasterize_forward", &VkSplat::rasterize_forward,
             "Execute the forward rasterization trainer", py::call_guard<py::gil_scoped_release>())
        .def("rasterize_backward", &VkSplat::rasterize_backward,
             "Execute the backward rasterization trainer", py::call_guard<py::gil_scoped_release>())
        .def("forward", &VkSplat::forward, "Execute forward all the way", py::call_guard<py::gil_scoped_release>())
        .def("compute_pixel_loss_grad", &VkSplat::compute_pixel_loss_grad,
             "Execute compute gradient of pixel loss (the SSIM term only on steps with step % ssim_every == 0)",
             py::arg("train_idx"), py::arg("step") = 0, py::call_guard<py::gil_scoped_release>())
        .def("set_train_image", &VkSplat::set_train_image,
             "Train on the give image for one step.", py::call_guard<py::gil_scoped_release>())
        .def("train_step", &VkSplat::train_step,
             "Train on the give image for one step.", py::call_guard<py::gil_scoped_release>())
        .def("train_steps", &VkSplat::train_steps,
             "Run training steps on the given train slots (step numbers first_step, first_step+1, ...), "
             "batched into as few GPU submits as possible",
             py::arg("indices"), py::arg("first_step"), py::call_guard<py::gil_scoped_release>())
        .def("_force_isect_capacity", &VkSplat::force_isect_capacity,
             "Test hook: shrink the tile-intersection buffers so the next batched step overflows",
             py::arg("capacity"), py::call_guard<py::gil_scoped_release>())
        .def("train_steps_slots", &VkSplat::train_steps,
             "Alias of train_steps(list_of_train_slots, start_step)",
             py::arg("slots"), py::arg("start_step"), py::call_guard<py::gil_scoped_release>())
        .def("render_train", &VkSplat::render_train,
             "Render the train camera pose to buffer.", py::call_guard<py::gil_scoped_release>())
        .def("render_val", &VkSplat::render_val,
             "Render the eval camera pose to buffer.", py::call_guard<py::gil_scoped_release>())
        .def("render_val_depth", &VkSplat::render_val_depth,
             "Render eval camera pose with native Vulkan expected depth, first-surface depth, alpha, and RGB.",
             py::arg("idx"))
        .def("render_pose_rgbd", &VkSplat::render_pose_rgbd,
             "Render arbitrary camera pose (c2w world 4x4) with native Vulkan expected depth, first-surface depth, alpha, and RGB.",
             py::arg("c2w"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("width"), py::arg("height"))
        .def("post_backward_step", &VkSplat::post_backward_step,
             "Run this after each step for densification.", py::call_guard<py::gil_scoped_release>())
        .def("get_vram_usage", &VkSplat::get_vram_usage, "Get VRAM Usage")
        .def("get_peak_vram_usage", &VkSplat::get_peak_vram_usage, "Get peak VRAM Usage")
        .def("get_timing_breakdown", &VkSplat::get_timing_breakdown, "Get timing for each step since start")
        .def("get_gpu_stage_timestamps_ms", &VkSplat::get_gpu_stage_timestamps_ms,
             "Get aggregated Vulkan GPU timestamp durations in milliseconds by pipeline stage")
        .def("reset_gpu_stage_timestamps", &VkSplat::reset_gpu_stage_timestamps,
             "Reset Vulkan GPU stage timestamp accumulators")
        .def("compact_mode0_active_set", &VkSplat::compact_mode0_active_set,
             "Sync freeze flags and return Mode-0 frozen vs. active trainable Gaussian counts")
        .def("get_map_generation", &VkSplat::get_map_generation,
             "Current splat map mutation generation counter")
        .def("get_active_set_generation", &VkSplat::get_active_set_generation,
             "Map generation counter at which the GPU active-set index buffer was last compacted")
        .def_property_readonly("map_generation", &VkSplat::get_map_generation,
             "Current splat map mutation generation counter")
        .def_property_readonly("active_set_generation", &VkSplat::get_active_set_generation,
             "Map generation counter at which the GPU active-set index buffer was last compacted")
        .def("project_and_fit_sh_vulkan", &VkSplat::project_and_fit_sh_vulkan,
             "Native Slang/Vulkan multi-view RGB-D projection and regularized WLS degree-0/1 SH initializer",
             py::arg("means_world"), py::arg("normals_world"),
             py::arg("rgb_views_u8"), py::arg("depth_views_f32"),
             py::arg("k_rgb_f32"), py::arg("k_depth_f32"),
             py::arg("w2c_f32"), py::arg("cam_o_f32"),
             py::arg("fallback_rgb_f32") = py::none(),
             py::arg("sh_degree") = 1,
             py::arg("reg_lambda") = 0.25f,
             py::arg("depth_tol_m") = 0.12f,
             py::arg("min_cos_angle") = 0.15f)
        .def("append_and_init_surfels_vulkan", &VkSplat::append_and_init_surfels_vulkan,
             "Fused GPU-resident multi-view RGB-D SH initialization and optimizer-preserving surfel append",
             py::arg("means_world"), py::arg("normals_world"),
             py::arg("scales_norm"), py::arg("quats_norm"), py::arg("opacities"),
             py::arg("group_ids"), py::arg("splat_states"), py::arg("fallback_rgb"),
             py::arg("rgb_views"), py::arg("depth_views"),
             py::arg("K_rgbs"), py::arg("K_depths"), py::arg("poses_c2w"),
             py::arg("dp_transform") = py::none(),
             py::arg("sh_reg") = 1e-2f,
             py::arg("depth_tol") = 0.12f,
             py::arg("sh_degree") = 1,
             py::arg("min_cos_angle") = 0.15f)
        .def("get_work_counters", &VkSplat::get_work_counters,
             "Cumulative tile-intersection (num_indices) and backward-dispatch counters since initialize; no extra readback")
        .def("get_vram_breakdown", &VkSplat::get_vram_breakdown, "Get VRAM for each buffer at the current moment")
        .def("get_gpu_memory_stats", &VkSplat::get_gpu_memory_stats, "Get exact Vulkan buffer allocations and adapter VRAM usage in MB")
        .def("set_local_cell_credits", &VkSplat::set_local_cell_credits,
             "Set GPU-resident per-splat local cell IDs and per-cell growth credits",
             py::arg("splat_cell_ids"), py::arg("cell_credits"))
        .def("clear_local_cell_credits", &VkSplat::clear_local_cell_credits,
             "Disable and clear local cell growth credits")
        .def("get_local_cell_credits", &VkSplat::get_local_cell_credits,
             "Read back current remaining per-cell growth credits")
        .def("get_splat_cell_ids", &VkSplat::get_splat_cell_ids,
             "Return host-mirrored per-splat cell IDs aligned with current splat order")
        .def("get_adc_host_readbacks", &VkSplat::get_adc_host_readbacks,
             "Number of ADC host mask readback trim events triggered by global effective_cap")
        .def("write_ply", &VkSplat::write_ply, "Save PLY file", py::call_guard<py::gil_scoped_release>())
        .def("get_device_group_ids", &VkSplat::get_device_group_ids, "Read the device-resident per-splat group ids")
        .def("get_device_group_state", &VkSplat::get_device_group_state, "Read the device-resident group state table")
        .def("get_splat_capacity", &VkSplat::get_splat_capacity, "Allocated capacity (elements) of per-splat buffers")
        .def("get_splat_group_ids", &VkSplat::get_splat_group_ids, "Get per-splat planar group IDs after Morton permutation")
        .def("set_splat_group_ids", &VkSplat::set_splat_group_ids, "Set per-splat planar group IDs", py::arg("group_ids"))
        .def("set_group_state", &VkSplat::set_group_state, "Set state for a structural group (0=normal, 1=IMAGINED, 2=LOCKED, 3=RETRACTED)", py::arg("group_id"), py::arg("state"), py::call_guard<py::gil_scoped_release>())
        .def("set_group_states", &VkSplat::set_group_states, "Batched set_group_state: one device upload for all (group_id, state) pairs", py::arg("group_ids"), py::arg("states"))
        .def("get_splat_states", &VkSplat::get_splat_states, "Get per-splat evidence states (uint32)")
        .def("set_splat_states", &VkSplat::set_splat_states, "Set per-splat evidence states (uint32)", py::arg("states"))
        .def("set_splat_states_sparse", &VkSplat::set_splat_states_sparse, "Sparse update of per-splat evidence states", py::arg("rows"), py::arg("states"))
        .def("set_splat_surfel_2d_mask", &VkSplat::set_splat_surfel_2d_mask, "Mark per-splat SURFEL_2D primitive flags (bit 16) and zero normal scale", py::arg("mask"))
        .def("promote_surfels_to_3d", &VkSplat::promote_surfels_to_3d, "Promote selected SURFEL_2D rows to GAUSSIAN_3D with measured normal scale sigma_n", py::arg("rows"), py::arg("sigma_n"))
        .def("set_splat_anchors", &VkSplat::set_splat_anchors, "Set per-splat (N, 8) normalized anchors [p_xyz, w_depth, n_xyz, w_plane]", py::arg("anchors"))
        .def("set_splat_anchors_sparse", &VkSplat::set_splat_anchors_sparse, "Sparse update of per-splat (R, 8) normalized anchors", py::arg("rows"), py::arg("anchors"))
        .def("get_splat_anchors", &VkSplat::get_splat_anchors, "Read back per-splat (N, 8) normalized anchors")
        .def("reanchor_splats_sparse", &VkSplat::reanchor_splats_sparse, "Pre-step sparse re-anchoring of observed structural surfels onto measured depth + RGB",
             py::arg("rows"), py::arg("xyz_norm"), py::arg("scales_norm"), py::arg("quats_norm"), py::arg("sh_dc"), py::arg("states"))
        .def("decay_opacities_sparse", &VkSplat::decay_opacities_sparse,
             "GPU-resident sparse logit-space opacity decay for contradicted splats (deduplicates rows, validates parameters, protects unobserved/verified/locked states)",
             py::arg("rows"), py::arg("delta_logit") = 2.5f, py::arg("min_opacity") = 1e-4f,
             py::arg("reset_moments") = true, py::arg("respect_protected_states") = true)
        .def("supports_imagined_protection", [](VkSplat&) { return true; }, "True when the optimizer and MCMC kernels keep IMAGINED (state 1) group geometry/opacity frozen")
        .def("set_sparse_backward", &VkSplat::set_sparse_backward,
             "Sparse tile backward: quantile in [0,1) (0 = dense), halo tiles, every full_every-th step dense",
             py::arg("quantile"), py::arg("halo") = 1, py::arg("full_every") = 4)
        .def("force_backward_variant", &VkSplat::force_backward_variant, "Pin one rasterize-backward variant (0 PerPixel, 1 PerSplat, 2/3/4 Tensor); -1 restores the scheduler", py::arg("impl"))
        .def("backward_variants", &VkSplat::backward_variants, "Backward variants available on this device")
        .def("clamp_scales", &VkSplat::clamp_scales, "GPU clamp of scales without readback: ungrouped splats to max_scale on all axes; grouped splats (group_id >= 0) to tangent_cap on x, y and max_scale on z", py::arg("max_scale"), py::arg("tangent_cap"), py::call_guard<py::gil_scoped_release>())
        .def("update_group_sparse", &VkSplat::update_group_sparse, "Rewrite xyz/scales/quats of one group's splats (rows in ascending splat index) and reset their geometry Adam moments, on GPU without readback", py::arg("group_id"), py::arg("xyz_norm"), py::arg("scales_norm"), py::arg("quats_norm"))
        .def("compact_retracted", &VkSplat::compact_retracted, "Remove all splats of RETRACTED groups from every per-splat buffer; returns the old->new index map (int32, -1 = removed). One host-side keep-list build, GPU gather, no readback.")
        .def("set_async_frame_upload", &VkSplat::set_async_frame_upload, "Asynchronous keyframe upload (staging ring + fence-free transfer submit, default on); off = old blocking copy", py::arg("enabled"))
        .def("set_freeze_skip_backward", &VkSplat::set_freeze_skip_backward, "Enable/disable skipping the raster-backward gradient atomics of frozen (IMAGINED/protected LOCKED) splats", py::arg("enabled"))
        .def("retract_group_sparse", &VkSplat::retract_group_sparse, "Retract a structural group on GPU without full-map host readback", py::arg("group_id"), py::call_guard<py::gil_scoped_release>())
        .def("set_sh_coeffs_reordered", &VkSplat::set_sh_coeffs_reordered, "Reorder and upload ONLY sh_coeffs without touching other Gaussian buffers or optimizer moments", py::arg("sh_coeffs"))
        .def("get_scales_opacities_packed", &VkSplat::get_scales_opacities_packed, "Read packed (N, 4) scales+opacities in a single device transfer")
        .def("get_sh_dc", &VkSplat::get_sh_dc, "Read only the (N, 3) DC band from reordered sh_coeffs on device")
        .def("get_splat_entity_ids", &VkSplat::get_splat_entity_ids, "Get per-splat entity IDs")
        .def("set_splat_entity_ids", &VkSplat::set_splat_entity_ids, "Set per-splat entity IDs", py::arg("entity_ids"))
        .def("get_splat_uids", &VkSplat::get_splat_uids, "Get persistent monotonic uint64 per-splat identities tracked across sort/prune/relocate/compact")
        .def("observe_and_apply_rayguard_gpu", &VkSplat::observe_and_apply_rayguard_gpu,
             "GPU-resident incremental RayGuard: uploads only HxW depth map, accumulates UID-keyed evidence on GPU, applies sparse logit opacity decay, and reads back 16-byte stats",
             py::arg("depth_m"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("n2c_3x4"), py::arg("cam_o_norm"), py::arg("dp_inv_scale"), py::arg("current_step"),
             py::arg("delta_logit") = 3.5f, py::arg("min_opacity") = 0.002f,
             py::arg("reset_moments") = true, py::arg("respect_protected_states") = true,
             py::arg("min_rays") = 2, py::arg("min_baseline_m") = 0.20f,
             py::arg("min_parallax_deg") = 8.0f, py::arg("surface_protect_m") = 0.12f,
             py::arg("cooldown_steps") = 500)
        .def("invalidate_rayguard_contradictions_gpu", &VkSplat::invalidate_rayguard_contradictions_gpu,
             "Invalidate GPU-resident RayGuard contradiction counters after SLAM pose revision (preserves support_hits)")
        .def("reset_rayguard_evidence_gpu", &VkSplat::reset_rayguard_evidence_gpu,
             "Reset GPU-resident RayGuard evidence buffer")
        .def("get_mcmc_stats", &VkSplat::get_mcmc_stats, "Get MCMC relocation and add statistics")
        .def("set_growth_cap", &VkSplat::set_growth_cap,
             "Set evidence-derived densification ceiling for ADC and MCMC; does not remove existing splats",
             py::arg("max_splats"))
        .def("set_densification_budget", &VkSplat::set_densification_budget,
             "Dynamically update densification budget (cap_max, cap_speculative_max, grow_factor, refine_every)",
             py::arg("cap_max"), py::arg("cap_speculative_max") = -1,
             py::arg("grow_factor") = -1.0f, py::arg("refine_every") = -1)
        .def("get_densification_budget", &VkSplat::get_densification_budget,
             "Return active densification budget dict")
        .def("set_stabilization_schedule", &VkSplat::set_stabilization_schedule,
             "Configure structural opacity reset floor and exponential refinement LR decay over max_steps",
             py::arg("structural_reset_opa") = 0.25f,
             py::arg("lr_final_scale") = 1.0f,
             py::arg("max_steps") = -1)
        .def("get_stabilization_schedule", &VkSplat::get_stabilization_schedule,
             "Return active stabilization schedule dict")
        .def("set_selective_freeze_unfrozen", &VkSplat::set_selective_freeze_unfrozen, "Toggle selective plane freeze", py::arg("unfrozen"))
        .def("reset_anchored_snapshot", &VkSplat::reset_anchored_snapshot, "Reset anchored splat count and snapshot", py::arg("new_num_anchored"))
        .def("clear_train_frames", &VkSplat::clear_train_frames, "Clear all loaded training frames")
        .def("clear_val_frames", &VkSplat::clear_val_frames, "Clear all loaded validation frames")
        .def("append_train_frame", &VkSplat::append_train_frame, "Append a streaming RGB training frame and camera pose without resetting optimizer",
             py::arg("image_rgba"), py::arg("c2w_world"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("image_name") = "stream_frame.png")
        .def("append_val_frame", &VkSplat::append_val_frame, "Append an evaluator-only validation RGB frame and camera pose",
             py::arg("image_rgba"), py::arg("c2w_world"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("image_name") = "val_frame.png")
        .def("replace_train_frame", &VkSplat::replace_train_frame, "Replace an existing training frame slot in-place",
             py::arg("train_idx"), py::arg("image_rgba"), py::arg("c2w_world"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("image_name") = "stream_frame.png")
        .def("append_splats_preserving_optimizer", &VkSplat::append_splats_preserving_optimizer, "Append new Gaussians while preserving Adam optimizer state of existing splats",
             py::arg("xyz_ws"), py::arg("sh_coeffs"), py::arg("rotations"), py::arg("scales"), py::arg("opacities"), py::arg("group_ids"), py::arg("splat_states") = py::none())
        .def("get_optimizer_moments", &VkSplat::get_optimizer_moments, "Device readback of the Adam moments and densification statistics (diagnostic)")
        .def("snapshot_optimizer_state", &VkSplat::snapshot_optimizer_state, "Take a non-destructive snapshot of Gaussian parameters and Adam optimizer state")
        .def("restore_optimizer_state", &VkSplat::restore_optimizer_state, "Restore a non-destructive snapshot of Gaussian parameters and Adam optimizer state", py::arg("snap"))
        .def("set_ago_telemetry_enabled", &VkSplat::set_ago_telemetry_enabled,
             "Enable or disable opt-in zero-perturbation AGO-0 per-Gaussian telemetry", py::arg("enabled"))
        .def("get_ago_telemetry_enabled", &VkSplat::get_ago_telemetry_enabled,
             "Return whether AGO-0 per-Gaussian telemetry is enabled")
        .def("reset_ago_telemetry", &VkSplat::reset_ago_telemetry,
             "Zero all accumulated AGO-0 per-Gaussian telemetry counters on device")
        .def("get_ago_telemetry", &VkSplat::get_ago_telemetry,
             "Read back per-Gaussian AGO-0 telemetry dict (grad_norm, update_norm, predicted_gain, abs_proj_grad, update_count, contrib_sum, last_step_updated, last_predicted_gain, num_gaussians, step)")
        .def("select_ago_active_set_gpu", &VkSplat::select_ago_active_set_gpu,
             "GPU-resident AGO-2 active-set compaction directly into active_indices_buf_ with 16-byte summary readback",
             py::arg("gain_threshold") = 0.0f,
             py::arg("newborn_warmup_steps") = 2.0f,
             py::arg("exclude_frozen") = true)
        .def("select_ago_adaptive_active_set_gpu", &VkSplat::select_ago_adaptive_active_set_gpu,
             "Single-dispatch GPU-resident adaptive AGO-2 active-set selection using prev_mean_gain * target_gain_ratio",
             py::arg("target_gain_ratio") = 0.5f,
             py::arg("newborn_warmup_steps") = 2.0f,
             py::arg("exclude_frozen") = true)
        .def("select_ago_exact_k_gpu", &VkSplat::select_ago_exact_k_gpu,
             "2-pass GPU 256-bin log-gain histogram + exact-K active-set compaction guaranteeing active_count == round(target_fraction * eligible) with zero per-Gaussian host readback",
             py::arg("target_fraction") = 0.5f,
             py::arg("newborn_warmup_steps") = 10.0f,
             py::arg("exclude_frozen") = true)
        .def("select_ago_active_set_gpu_async", &VkSplat::select_ago_active_set_gpu_async,
             "Fully GPU-native histogram -> cutoff -> compaction in a single command batch with zero synchronous host readbacks",
             py::arg("target_fraction") = 0.5f,
             py::arg("newborn_warmup_steps") = 10.0f,
             py::arg("exclude_frozen") = true,
             py::arg("max_budget_splats") = 0u)
        .def("get_ago_cutoff_indirect_state", &VkSplat::get_ago_cutoff_indirect_state,
             "Diagnostic host readback of the 12-word GPU ago_cutoff_indirect_buf_ (tests/verification only)")
        .def("set_external_strategy_mode", &VkSplat::set_external_strategy_mode,
             "Enable or disable AGO-0 external strategy ownership mode (disables internal ADC/MCMC/reset_opa)",
             py::arg("enabled"))
        .def("get_external_strategy_mode", &VkSplat::get_external_strategy_mode,
             "Return whether AGO-0 external strategy ownership mode is active")
        .def("set_ago_param_mask", &VkSplat::set_ago_param_mask,
             "Set AGO-L0 parameter-specific Adam update bitmask (POS=1, ROT=2, SCALE=4, OPAC=8, SH=16, ALL=31)",
             py::arg("mask"))
        .def("get_ago_param_mask", &VkSplat::get_ago_param_mask,
             "Return active AGO-L0 parameter-specific Adam update bitmask")
        .def("ago_execute_external_op", &VkSplat::ago_execute_external_op,
             "Execute an explicit AGO-0/L0 external strategy op (FREEZE/REOPEN/RETRACT/PRUNE/REFINE; REJECTED_UNSUPPORTED otherwise)",
             py::arg("op"), py::arg("group_id") = -1)
        .def("ago_offload_snapshot_to_ram", &VkSplat::ago_offload_snapshot_to_ram,
             "Offload Gaussian params, Adam moments, and group states bit-for-bit to host RAM; returns snapshot_id")
        .def("ago_restore_snapshot_from_ram", &VkSplat::ago_restore_snapshot_from_ram,
             "Restore Gaussian params, Adam moments, and group states bit-for-bit from host RAM snapshot",
             py::arg("snapshot_id"))
        .def("ago_drop_ram_snapshot", &VkSplat::ago_drop_ram_snapshot,
             "Release a host RAM snapshot by snapshot_id",
             py::arg("snapshot_id"))
        .def("ago_ram_snapshot_bytes", &VkSplat::ago_ram_snapshot_bytes,
             "Return byte size of a stored host RAM snapshot",
             py::arg("snapshot_id"))
        .def("ago_gpu_submit_action", &VkSplat::ago_gpu_submit_action,
             "Submit an AGOAction to the real NonBlockingGpuExecutor (non-blocking)",
             py::arg("action"))
        .def("ago_gpu_poll_completions", &VkSplat::ago_gpu_poll_completions,
             "Poll completed GPU actions via vkGetFenceStatus (strictly non-blocking)",
             py::arg("max_count") = 16u)
        .def("ago_gpu_available_credits", &VkSplat::ago_gpu_available_credits,
             "Return available ring-slot credits in the NonBlockingGpuExecutor")
        .def("ago_gpu_inflight_count", &VkSplat::ago_gpu_inflight_count,
             "Return in-flight submission count in the NonBlockingGpuExecutor")
        .def("ago_gpu_shutdown", &VkSplat::ago_gpu_shutdown,
             "Safely drain and shut down the NonBlockingGpuExecutor")
        .def("ago_scheduler_init", &VkSplat::ago_scheduler_init,
             "Initialize integrated native C++ AgoScheduler",
             py::arg("config") = vksplat::ago::SchedulerConfig{},
             py::arg("use_gpu_executor") = false)
        .def("ago_scheduler_tick", &VkSplat::ago_scheduler_tick,
             "Execute one bounded non-blocking AgoScheduler tick")
        .def("ago_scheduler_enqueue_action", &VkSplat::ago_scheduler_enqueue_action,
             "Enqueue an AGOAction into the integrated AgoScheduler",
             py::arg("action"))
        .def("ago_scheduler_push_evidence", &VkSplat::ago_scheduler_push_evidence,
             "Push an EvidenceMessage into the integrated AgoScheduler",
             py::arg("msg"))
        .def("ago_scheduler_set_budget", &VkSplat::ago_scheduler_set_budget,
             "Publish a DeadlineBudget snapshot into the integrated AgoScheduler",
             py::arg("budget"))
        .def("ago_scheduler_summary", &VkSplat::ago_scheduler_summary,
             "Return the latest SchedulerSummary from the integrated AgoScheduler")
        .def("ago_scheduler_trace", &VkSplat::ago_scheduler_trace,
             "Return chronological EventTrace snapshot from the integrated AgoScheduler")
        .def("reproject_temporal_depth_gpu", &VkSplat::reproject_temporal_depth_gpu,
             "GPU temporal depth reprojection of historical depth maps into target viewpoint with atomic uint32 z-buffer, view-angle cosine weighting, and valid-support count",
             py::arg("src_depths"),
             py::arg("src_c2ws"),
             py::arg("dst_c2w"),
             py::arg("K_src"),
             py::arg("K_dst"),
             py::arg("max_depth_m") = 10.0f,
             py::arg("dst_hw") = py::none(),
             py::arg("min_depth_m") = 0.05f,
             py::arg("min_cos_angle") = 0.0f)
        .def("get_ago_step_gain_summary", &VkSplat::get_ago_step_gain_summary,
             "Return cached 16-byte GPU AGO-2 step-gain summary without per-Gaussian readback")
        .def("snapshot_ago_state", &VkSplat::snapshot_ago_state,
             "Take an exact byte-identical snapshot of Gaussian parameters, all Adam moments, strategy state, active set, and step for AGO-1 counterfactual probes")
        .def("restore_ago_state", &VkSplat::restore_ago_state,
             "Restore an exact snapshot from snapshot_ago_state() and bump map_generation cleanly", py::arg("snapshot"))
        .def("set_train_depth", &VkSplat::set_train_depth,
             "Attach a float32 (H, W) metric physical depth map (meters; <=0 invalid) and optional confidence weight map to training frame `frame_idx`",
             py::arg("frame_idx"), py::arg("depth_m"), py::arg("weight_map") = py::none())
        .def("clear_train_depths", &VkSplat::clear_train_depths,
             "Clear all attached training depth maps")
        .def("set_depth_loss_weight", &VkSplat::set_depth_loss_weight,
             "Set opt-in differentiable expected-depth loss weight (0.0 disables) and Huber delta in meters",
             py::arg("lambda_depth"), py::arg("huber_delta") = 0.15f)
        .def("get_depth_loss_weight", &VkSplat::get_depth_loss_weight,
             "Return current differentiable expected-depth loss weight")
        .def("set_active_set", &VkSplat::set_active_set,
             "Set explicit optimization active-set Gaussian indices (independent of evidence-based freezing)",
             py::arg("indices"))
        .def("clear_active_set", &VkSplat::clear_active_set,
             "Clear explicit optimization active-set and restore full-map updates")
        .def("get_active_set", &VkSplat::get_active_set,
             "Return current optimization active-set uint32 indices (empty if inactive)")
        .def("cleanup", &VkSplat::cleanup, "Cleanup Vulkan resources")
        .DEF_BUFFER_ARRAY(xyz_ws, float, 3, (size_t)-1, "mean")
        .DEF_BUFFER_ARRAY(sh_coeffs, float, 16, 3, "sh")
        .DEF_BUFFER_ARRAY(rotations, float, 4)
        .DEF_BUFFER_0(scales, scales_opacs, float, 4, (size_t)-1, "", 3, 0)
        .DEF_BUFFER_0(opacities, scales_opacs, float, 4, (size_t)-1, "", 1, 3)
        .DEF_ARRAY(scales_opacs, float)
        .DEF_BUFFER_ARRAY(tiles_touched, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(rect_tile_space, int64_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(radii, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(depths, float, 1)
        .DEF_BUFFER_ARRAY(inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(rgb, float, 3)
        .DEF_BUFFER_ARRAY(index_buffer_offset, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_1, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_1, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_2, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_2, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(tile_ranges, int32_t, 2)
        .DEF_BUFFER_ARRAY(pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(n_contributors, int32_t, self.uniforms.image_width, 1)
        .DEF_BUFFER_ARRAY(v_pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(v_xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(v_depths, float, 1)
        .DEF_BUFFER_ARRAY(v_inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(v_rgb, float, 3)
        .def("_set_num_splats", [] (VkSplat &self, size_t n) {
            self.buffers.num_splats = n;
            self.trainer.bumpMapGeneration();
        }, "Directly set buffers.num_splats (use at own risk)", py::arg("num_splats"))
        ;

    py::register_exception<std::runtime_error>(m, "RuntimeError");
}
