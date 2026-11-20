#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gs_renderer.h"
#include "buffer.h"
#include "gs_trainer.h"

#include <chrono>

namespace py = pybind11;

class VkSplat {
public:
    // VulkanGSRenderer trainer;
    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config;
    bool isect_capacity_forced = false;  // test hook: skip the proactive capacity growth once
    size_t last_isects = 0;  // intersection count of the last gated step (sizes the next batch)

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
            "morton_sort_compute_stats",
            "morton_sort_generate_keys",
            "morton_sort_apply_indices",
            "morton_sort_apply_indices_sh",
            "morton_sort_update_buffer",
            "morton_sort_update_buffer_sh",
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
        snap["xyz_ws"] = buffer_to_array<float, float>(buffers.xyz_ws, 3, (size_t)-1, "mean");
        snap["sh_coeffs"] = buffer_to_array<float, float>(buffers.sh_coeffs, 16, 3, "sh");
        snap["rotations"] = buffer_to_array<float, float>(buffers.rotations, 4);
        snap["scales_opacs"] = buffer_to_array<float, float>(buffers.scales_opacs, 4);
        snap["splat_group_ids"] = get_splat_group_ids();
        snap["splat_entity_ids"] = get_splat_entity_ids();
        if (!trainer.get_splat_states().empty()) {
            snap["splat_states"] = get_splat_states();
        }
        if (buffers.splat_anchor.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.splat_anchor.deviceSize() > 0) {
            snap["splat_anchor"] = buffer_to_array<float, float>(buffers.splat_anchor, 8, (size_t)-1, "mean");
        }
        if (buffers.g_xyz_ws.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.g_xyz_ws.deviceSize() > 0) {
            snap["g_xyz_ws"] = buffer_to_array<float, float>(buffers.g_xyz_ws, 6, (size_t)-1, "mean");
            snap["g_sh_coeffs_1"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_1, 48, (size_t)-1, "mean");
            snap["g_sh_coeffs_2"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_2, 48, (size_t)-1, "mean");
            snap["g_rotations"] = buffer_to_array<float, float>(buffers.g_rotations, 8, (size_t)-1, "mean");
            snap["g_scales_opacs"] = buffer_to_array<float, float>(buffers.g_scales_opacs, 8, (size_t)-1, "mean");
        }
        if (buffers.default_grad.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_grad.deviceSize() > 0) {
            snap["default_grad"] = buffer_to_array<float, float>(buffers.default_grad, 2, (size_t)-1, "mean");
        }
        if (buffers.default_radii.deviceBuffer.buffer != VK_NULL_HANDLE && buffers.default_radii.deviceSize() > 0) {
            snap["default_radii"] = buffer_to_array<float, float>(buffers.default_radii, 1, (size_t)-1, "mean");
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
        if (snap.contains(py::str("splat_states"))) {
            set_splat_states(snap["splat_states"].cast<py::array_t<uint32_t>>());
        } else {
            trainer.set_splat_states_vec({});
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
            trainer.executeRasterizeForward(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
        }
    }
    
    void rasterize_backward() {
        set_freeze_flags();
        try {
            trainer.executeRasterizeBackward(uniforms, buffers);
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

    size_t get_vram_usage() {
        return buffers.getTotalAllocSize();
    }

    size_t get_peak_vram_usage() {
        return trainer.getPeakAllocSize();
    }

    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown() {
        return PerfTimer::get_summary();
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
        .def("post_backward_step", &VkSplat::post_backward_step,
             "Run this after each step for densification.", py::call_guard<py::gil_scoped_release>())
        .def("get_vram_usage", &VkSplat::get_vram_usage, "Get VRAM Usage")
        .def("get_peak_vram_usage", &VkSplat::get_peak_vram_usage, "Get peak VRAM Usage")
        .def("get_timing_breakdown", &VkSplat::get_timing_breakdown, "Get timing for each step since start")
        .def("get_work_counters", &VkSplat::get_work_counters,
             "Cumulative tile-intersection (num_indices) and backward-dispatch counters since initialize; no extra readback")
        .def("get_vram_breakdown", &VkSplat::get_vram_breakdown, "Get VRAM for each buffer at the current moment")
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
        .def("set_splat_anchors", &VkSplat::set_splat_anchors, "Set per-splat (N, 8) normalized anchors [p_xyz, w_depth, n_xyz, w_plane]", py::arg("anchors"))
        .def("set_splat_anchors_sparse", &VkSplat::set_splat_anchors_sparse, "Sparse update of per-splat (R, 8) normalized anchors", py::arg("rows"), py::arg("anchors"))
        .def("reanchor_splats_sparse", &VkSplat::reanchor_splats_sparse, "Pre-step sparse re-anchoring of observed structural surfels onto measured depth + RGB",
             py::arg("rows"), py::arg("xyz_norm"), py::arg("scales_norm"), py::arg("quats_norm"), py::arg("sh_dc"), py::arg("states"))
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
        }, "Directly set buffers.num_splats (use at own risk)", py::arg("num_splats"))
        ;

    py::register_exception<std::runtime_error>(m, "RuntimeError");
}
