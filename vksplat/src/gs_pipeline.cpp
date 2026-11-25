#include "gs_pipeline.h"
#include "perf_timer.h"

#include <filesystem>
#include <fstream>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static const size_t MAX_UNIFORM_SIZE = 192;

static const uint32_t MAX_TIMESTAMP_QUERY_COUNT = 48;

#if defined(__SSE2__) || defined(_MSC_VER)
#define SSE2_AVAILABLE 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif


std::vector<uint32_t> loadSpirv(std::string spirv_path) {
    // Load the SPIR-V file
    #ifdef WIN32
    // replace "/" with "\\"
    size_t start_pos = 0;
    while((start_pos = spirv_path.find("/", start_pos)) != std::string::npos) {
        spirv_path.replace(start_pos, 1, "\\");
        start_pos += 1;
    }
    #endif

    std::ifstream file(spirv_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open file: " + spirv_path);

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))
        throw std::runtime_error("Failed to read file: " + spirv_path);
    
    return spirv_code;
}


VulkanGSPipeline::VulkanGSPipeline() :
    instance(VK_NULL_HANDLE),
    physical_device(VK_NULL_HANDLE),
    device(VK_NULL_HANDLE),
    command_queue(VK_NULL_HANDLE),
    command_pool(VK_NULL_HANDLE),
    command_buffer(VK_NULL_HANDLE),
    fence(VK_NULL_HANDLE),
    timestamp_query_pool(VK_NULL_HANDLE),
    queue_family_index(UINT32_MAX) {
}

VulkanGSPipeline::~VulkanGSPipeline() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}

void VulkanGSPipeline::initialize(int device_id) {

#if ENABLE_VULKAN_VALIDATION_LAYER
    do {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        enableValidationLayer = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp("VK_LAYER_KHRONOS_validation", layerProperties.layerName) == 0)
                { enableValidationLayer = true; break; }
        }

        if (!enableValidationLayer)
            fprintf(stderr, "WARNING: Vulkan validation layer not available");
    } while(0);
#endif

    createInstance();
    selectPhysicalDevice(device_id);
    createDevice();
    createCommandPool();
    createFence();
    createQueryPools();

    commandBatchInProgress = false;
}


void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
    #define _(name) { \
        destroyBuffer(buffers.name.deviceBuffer); \
        buffers.name.clear(); \
        buffers.name.shrink_to_fit(); \
    }
    _(xyz_ws)_(sh_coeffs)_(rotations)_(scales_opacs)
    _(tiles_touched)_(rect_tile_space)_(radii)_(xy_vs)_(depths)_(inv_cov_vs_opacity)_(rgb)
    _(index_buffer_offset)_(sorting_keys_1)_(sorting_keys_2)_(sorting_gauss_idx_1)_(sorting_gauss_idx_2)_(tile_ranges)
    _(pixel_state)_(pixel_depth)_(n_contributors)_(v_pixel_state)_(ref_image)
    _(v_xy_vs)_(v_depths)_(v_inv_cov_vs_opacity)_(v_rgb)
    _(g_xyz_ws)_(g_sh_coeffs_1)_(g_sh_coeffs_2)_(g_rotations)_(g_scales_opacs)
    _(default_grad)_(default_radii)_(default_dupli_mask)_(default_split_mask)_(default_keep_mask)
    _(mcmc_sample_probs)_(mcmc_sample_probs_cumsum)_(mcmc_index_map)_(mcmc_n_idx_buffer)
    _(group_ids)_(group_state)_(splat_state)_(splat_anchor)
    _(splat_cell_ids)_(cell_credits)_(cell_eligible)_(default_spec_remaining)
    _(_temp_gauss_attr)_(_temp_indices)_(_temp_sum)_(_temp_cumsum)
    _(_cumsum_blockSums)_(_cumsum_blockSums2)_(_sorting_histogram)_(_sorting_histogram_cumsum)
    #undef _
}

void VulkanGSPipeline::cleanup() {
    HOST_GUARD;

    if (stager.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, stager.buffer, nullptr);
        vkFreeMemory(device, stager.memory, nullptr);
        stager.buffer = VK_NULL_HANDLE;
        stager.memory = VK_NULL_HANDLE;
        stager.allocSize = 0;
    }

    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        destroyGateBuffers();
        append_pending = false;
        for (auto& slot : upload_slots) {
            slot.pending = false;
            destroyUploadStage(slot);
            if (slot.fence != VK_NULL_HANDLE) vkDestroyFence(device, slot.fence, nullptr);
            if (slot.cb != VK_NULL_HANDLE) vkFreeCommandBuffers(device, command_pool, 1, &slot.cb);
            slot.fence = VK_NULL_HANDLE;
            slot.cb = VK_NULL_HANDLE;
        }
        for (auto& b : retired_buffers)
            destroyBuffer(b);
        retired_buffers.clear();
        freeAppendStage();
        if (append_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, append_fence, nullptr);
            append_fence = VK_NULL_HANDLE;
        }
        if (append_cb != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &append_cb);
            append_cb = VK_NULL_HANDLE;
        }

        for (_ComputePipeline* pipeline : all_compute_pipelines)
            destroyComputePipeline(*pipeline);
        all_compute_pipelines.clear();
        
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (timestamp_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, timestamp_query_pool, nullptr);
            timestamp_query_pool = VK_NULL_HANDLE;
        }
        if (ago_timestamp_query_pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, ago_timestamp_query_pool_, nullptr);
            ago_timestamp_query_pool_ = VK_NULL_HANDLE;
        }
        pfn_cmd_begin_debug_label_ = nullptr;
        pfn_cmd_end_debug_label_ = nullptr;
        
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            command_buffer = VK_NULL_HANDLE;
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

}


void VulkanGSPipeline::createInstance() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = VK_NULL_HANDLE;
    app_info.pApplicationName = "Vulkan Gaussian Rasterization";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pNext = VK_NULL_HANDLE;
    create_info.pApplicationInfo = &app_info;

    if (enableValidationLayer) {
        static const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
        create_info.enabledLayerCount = 1u;
        create_info.ppEnabledLayerNames = &validation_layer_name;
    }

    // Enable VK_EXT_debug_utils when available for RGP / Perfetto GPU command labels.
    std::vector<const char*> instance_extensions;
    uint32_t inst_ext_count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, nullptr) == VK_SUCCESS && inst_ext_count > 0) {
        std::vector<VkExtensionProperties> inst_exts(inst_ext_count);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, inst_exts.data()) == VK_SUCCESS) {
            for (const auto& ext : inst_exts) {
                if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    break;
                }
            }
        }
    }
    create_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
    create_info.ppEnabledExtensionNames = instance_extensions.empty() ? nullptr : instance_extensions.data();
    
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        _THROW_ERROR("Failed to create instance");

    pfn_cmd_begin_debug_label_ = nullptr;
    pfn_cmd_end_debug_label_ = nullptr;
    if (!instance_extensions.empty()) {
        pfn_cmd_begin_debug_label_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT")
        );
        pfn_cmd_end_debug_label_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT")
        );
    }
}

void VulkanGSPipeline::selectPhysicalDevice(int device_id) {
    static constexpr int kANSIDefault = 0;
    static constexpr int kANSIRed = 91;
    static constexpr int kANSIGreen = 32;
    static constexpr int kANSIOrange = 93;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        const char* message = "Could not find any physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        throw std::runtime_error(message);
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    struct SelectedDevice {
        int idx = -1;
        VkPhysicalDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIdx = UINT32_MAX;
        DeviceInfo deviceInfo;
    };
    std::vector<SelectedDevice> viableDevices;
    std::vector<SelectedDevice> softViableDevices;

    const auto& [
        minMaxGroups, minMaxThreads, minSharedMemory
    ] = getDeviceRequirement();

    printf(
        "Device Requirement: subgroup>=%d, maxGroups>=[%u %u %u], maxThreads>=[%u %u %u], maxShared>=%u, I16|I64|F32Atomic \n",
        (int)SUBGROUP_SIZE,
        minMaxGroups[0], minMaxGroups[1], minMaxGroups[2],
        minMaxThreads[0], minMaxThreads[1], minMaxThreads[2],
        minSharedMemory
    );
    fflush(stdout);

    for (size_t i = 0; i < devices.size(); i++) {
        VkPhysicalDevice& device = devices[i];

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        auto& limits = deviceProperties.limits;
        uint32_t maxGroupsX = limits.maxComputeWorkGroupCount[0];
        uint32_t maxGroupsY = limits.maxComputeWorkGroupCount[1];
        uint32_t maxGroupsZ = limits.maxComputeWorkGroupCount[2];
        uint32_t maxThreadsX = limits.maxComputeWorkGroupSize[0];
        uint32_t maxThreadsY = limits.maxComputeWorkGroupSize[1];
        uint32_t maxThreadsZ = limits.maxComputeWorkGroupSize[2];
        bool validGroupCount[4] = {
            maxGroupsX >= minMaxGroups[0],
            maxGroupsY >= minMaxGroups[1],
            maxGroupsZ >= minMaxGroups[2],
        };
        validGroupCount[3] = (validGroupCount[0] && validGroupCount[1] && validGroupCount[2]);
        bool validGroupSize[4] = {
            maxThreadsX >= minMaxThreads[0],
            maxThreadsY >= minMaxThreads[1],
            maxThreadsZ >= minMaxThreads[2],
        };
        validGroupSize[3] = (validGroupSize[0] && validGroupSize[1] && validGroupSize[2]);
        bool validSharedSize = limits.maxComputeSharedMemorySize >= minSharedMemory;

        // check subgroup size support
        VkPhysicalDeviceSubgroupProperties subgroupProperties{};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties2KHR deviceProperties2{};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(device, &deviceProperties2);
        bool validSubgroupSize = subgroupProperties.subgroupSize >= SUBGROUP_SIZE;

        // check compute pipeline.support
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
        uint32_t queueFamilyIdx = (uint32_t)(-1);
        for (uint32_t i = 0; i < queue_families.size(); i++) {
            if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                (queue_families[i].timestampValidBits != 0)
            ) queueFamilyIdx = i;
        }
        bool validQueueFamily = ((int32_t)queueFamilyIdx != -1);

        // check feature support
        bool hasInt16 = true, hasInt64 = true, hasFloat32AtomicAdd = true;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_float_features.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &atomic_float_features;
        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);
        if (deviceFeatures2.features.shaderInt16 == VK_FALSE)
            hasInt16 = false;
        if (deviceFeatures2.features.shaderInt64 == VK_FALSE)
            hasInt64 = false;
        if (atomic_float_features.shaderBufferFloat32AtomicAdd == VK_FALSE)
            hasFloat32AtomicAdd = false;

        bool hasMemoryBudget = false;
        uint32_t ext_count = 0;
        if (vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr) == VK_SUCCESS && ext_count > 0) {
            std::vector<VkExtensionProperties> exts(ext_count);
            if (vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, exts.data()) == VK_SUCCESS) {
                for (const auto& ext : exts) {
                    if (std::strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                        hasMemoryBudget = true;
                        break;
                    }
                }
            }
        }

        DeviceVendor vendor = DeviceVendor::Unknown;
        if (deviceProperties.vendorID == 0x10DE)
            vendor = DeviceVendor::NVIDIA;
        else if (deviceProperties.vendorID == 0x1002 || deviceProperties.vendorID == 0x1022)
            vendor = DeviceVendor::AMD;
        else if (deviceProperties.vendorID == 0x8086)
            vendor = DeviceVendor::Intel_R_;
        else if (deviceProperties.vendorID == 0x13B5)
            vendor = DeviceVendor::ARM;
        else if (deviceProperties.vendorID == 0x5143)
            vendor = DeviceVendor::Qualcomm;

        bool softViable = validSubgroupSize && validGroupSize[3] && validQueueFamily && hasInt16;
        bool viable = softViable && validGroupCount[3] && validSharedSize && hasInt64 && hasFloat32AtomicAdd;
        SelectedDevice deviceInfo{
            (int)i, device, queueFamilyIdx,
            {
                subgroupProperties.subgroupSize,
                limits.maxComputeSharedMemorySize,
                maxGroupsX, maxGroupsY, maxGroupsZ,
                maxThreadsX, maxThreadsY, maxThreadsZ,
                hasInt16, hasInt64, hasFloat32AtomicAdd, hasMemoryBudget,
                vendor, deviceProperties.vendorID,
                deviceProperties.deviceName
            }
        };
        bool is_excluded_device = (device_id >= 0 && device_id != (int)i);
        if (viable && !is_excluded_device)
            viableDevices.push_back(deviceInfo);
        else if (softViable && !is_excluded_device)
            softViableDevices.push_back(deviceInfo);

        printf(
            "[%d] %s - \033[%dm%s\033[m\n "
            " subgroup=\033[%dm%u\033[m, "
            "maxGroups=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxThreads=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxShared=\033[%dm%u\033[m, "
            "\033[%dmI16\033[m|\033[%dmI64\033[m|\033[%dmF32Atomic\033[m\n",
            (int)i, deviceProperties.deviceName,
            viable ? kANSIGreen : softViable ? kANSIOrange : kANSIRed,
            viable ? "VIABLE" : softViable ? "POSSIBLY VIABLE" : "NOT VIABLE",
            validSubgroupSize ? kANSIDefault : kANSIRed, subgroupProperties.subgroupSize,
            validGroupCount[0] ? kANSIDefault : kANSIOrange, maxGroupsX,
            validGroupCount[1] ? kANSIDefault : kANSIOrange, maxGroupsY,
            validGroupCount[2] ? kANSIDefault : kANSIOrange, maxGroupsZ,
            validGroupSize[0] ? kANSIDefault : kANSIRed, maxThreadsX,
            validGroupSize[1] ? kANSIDefault : kANSIRed, maxThreadsY,
            validGroupSize[2] ? kANSIDefault : kANSIRed, maxThreadsZ,
            validSharedSize ? kANSIDefault : kANSIOrange, limits.maxComputeSharedMemorySize,
            hasInt16 ? kANSIDefault : kANSIRed,
            hasInt64 ? kANSIDefault : kANSIOrange,
            hasFloat32AtomicAdd ? kANSIDefault : kANSIOrange
        );
        if (softViable) {
            if (!hasInt64)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: To use this device, shaders must be compiled with USE_EMULATED_INT64=1.");
            if (!hasFloat32AtomicAdd)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: To use this device, shaders must be compiled with USE_EMULATED_F32_ATOMIC=1.");
            if (!validGroupCount[3])
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device may not work if you want to train a large scene.");
            if (!validSharedSize)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device does not have sufficient shared memory, which can involve undefined behavior.");
        }
        fflush(stdout);
    }
    
    SelectedDevice device;
    if (!viableDevices.empty())
        device = viableDevices[0];
    else if (!softViableDevices.empty())
        device = softViableDevices[0];
    else {
        const char* message = "Could not find a viable physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        if (device_id >= 0)
            printf("\033[%dmNote: Device [%d] is requested, but it %s.\033[m\n",
                kANSIOrange, device_id, device_id >= (int)devices.size() ? "does not exist" : "is not viable");
        throw std::runtime_error(message);
    }

    this->physical_device = device.device;
    this->queue_family_index = device.queueFamilyIdx;
    this->deviceInfo = device.deviceInfo;
    printf("Using device [\033[%dm%d\033[m]%s\n",
        viableDevices.empty() ? kANSIOrange : kANSIDefault,
        device.idx,
        viableDevices.empty() ? " (\033[93mPOSSIBLY VIABLE\033[m)" : ""
    );
    if (!deviceInfo.hasFloat32AtomicAdd)
        printf("\033[%dm%s\033[m\n", kANSIOrange, "WARNING: Float32AtomicAdd is not available. Make sure shaders are compiled with USE_EMULATED_F32_ATOMIC=1.");
    if (!deviceInfo.hasInt64)
        printf("\033[%dm%s\033[m\n", kANSIOrange, "WARNING: Int64 is not available. Make sure shaders are compiled with USE_EMULATED_INT64=1.");
    printf("\n");
    fflush(stdout);
}

std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>
VulkanGSPipeline::get_device_info() const {
    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> result;
    result["subgroup_size"] = deviceInfo.subgroupSize;
    result["shared_size"] = deviceInfo.sharedSize;
    result["max_groups"] = std::vector<uint32_t>{deviceInfo.maxGroupsX, deviceInfo.maxGroupsY, deviceInfo.maxGroupsZ};
    result["max_threads"] = std::vector<uint32_t>{deviceInfo.maxThreadsX, deviceInfo.maxThreadsY, deviceInfo.maxThreadsZ};
    result["has_int16"] = deviceInfo.hasInt16;
    result["has_int64"] = deviceInfo.hasInt64;
    result["has_float32_atomic_add"] = deviceInfo.hasFloat32AtomicAdd;
    result["vendor"] = deviceInfo.vendorId;
    result["name"] = deviceInfo.name;
    return result;
}

void VulkanGSPipeline::createDevice() {
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext = VK_NULL_HANDLE;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    VkPhysicalDeviceFeatures enabledFeatures = {};
    enabledFeatures.shaderInt16 = VK_TRUE;
    if (deviceInfo.hasInt64)
        enabledFeatures.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {};
    atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float_features.pNext = VK_NULL_HANDLE;

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
    subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
    subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
    subgroupSizeControlFeatures.pNext = VK_NULL_HANDLE;
    if (deviceInfo.hasFloat32AtomicAdd)
        subgroupSizeControlFeatures.pNext = &atomic_float_features;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &enabledFeatures;

    std::vector<const char*> device_extensions = {
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    };
    if (deviceInfo.hasMemoryBudget)
        device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pNext = &subgroupSizeControlFeatures;

    if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create device");
    }
    
    vkGetDeviceQueue(device, queue_family_index, 0, &command_queue);
}

VulkanGSPipeline::GpuMemoryStats VulkanGSPipeline::getGpuMemoryStats(size_t pipeline_buffers_bytes) const {
    constexpr double kInvMiB = 1.0 / (1024.0 * 1024.0);
    GpuMemoryStats stats;
    const size_t dev_bytes = std::max(current_vram, pipeline_buffers_bytes);
    const size_t staging_bytes = getStagingAllocBytes();
    stats.device_buffers_mb = static_cast<double>(dev_bytes) * kInvMiB;
    stats.peak_device_buffers_mb = static_cast<double>(std::max(peak_vram, dev_bytes)) * kInvMiB;
    stats.staging_buffers_mb = static_cast<double>(staging_bytes) * kInvMiB;

    uint64_t vram_used = 0;
    uint64_t vram_budget = 0;
    if (physical_device != VK_NULL_HANDLE && deviceInfo.hasMemoryBudget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
        budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mem_props2{};
        mem_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mem_props2.pNext = &budget_props;
        vkGetPhysicalDeviceMemoryProperties2(physical_device, &mem_props2);
        for (uint32_t h = 0; h < mem_props2.memoryProperties.memoryHeapCount; ++h) {
            if (mem_props2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                vram_used += budget_props.heapUsage[h];
                vram_budget += budget_props.heapBudget[h];
            }
        }
        if (vram_used > 0)
            stats.source = "VK_EXT_memory_budget";
    }

    if (vram_used == 0) {
        auto read_u64 = [](const std::filesystem::path& p) -> uint64_t {
            std::ifstream f(p);
            uint64_t v = 0;
            if (f >> v)
                return v;
            return 0;
        };
        auto read_hex_u32 = [](const std::filesystem::path& p) -> uint32_t {
            std::ifstream f(p);
            uint32_t v = 0;
            if (f >> std::hex >> v)
                return v;
            return 0;
        };
        uint64_t fallback_used = 0, fallback_total = 0;
        for (int card = 0; card < 16; ++card) {
            std::filesystem::path dev_dir = "/sys/class/drm/card" + std::to_string(card) + "/device";
            std::filesystem::path used_path = dev_dir / "mem_info_vram_used";
            if (!std::filesystem::exists(used_path))
                continue;
            uint64_t used = read_u64(used_path);
            uint64_t total = read_u64(dev_dir / "mem_info_vram_total");
            uint32_t vend = read_hex_u32(dev_dir / "vendor");
            if (deviceInfo.vendorId != 0 && vend == deviceInfo.vendorId && used > 0) {
                vram_used = used;
                vram_budget = total;
                stats.source = "sysfs_drm";
                break;
            }
            if (fallback_used == 0 && used > 0) {
                fallback_used = used;
                fallback_total = total;
            }
        }
        if (vram_used == 0 && fallback_used > 0) {
            vram_used = fallback_used;
            vram_budget = fallback_total;
            stats.source = "sysfs_drm";
        }
    }

    if (vram_used == 0) {
        vram_used = dev_bytes;
        stats.source = "device_buffers_fallback";
    } else {
        vram_used = std::max<uint64_t>(vram_used, dev_bytes);
    }
    stats.vram_used_mb = static_cast<double>(vram_used) * kInvMiB;
    stats.vram_budget_mb = static_cast<double>(vram_budget) * kInvMiB;
    return stats;
}

void VulkanGSPipeline::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;
    
    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create command pool");

    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;
    
    if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate command buffer");
    if (vkAllocateCommandBuffers(device, &alloc_info, &append_cb) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate append command buffer");
    for (auto& slot : upload_slots)
        if (vkAllocateCommandBuffers(device, &alloc_info, &slot.cb) != VK_SUCCESS)
            _THROW_ERROR("Failed to allocate upload command buffer");
}

void VulkanGSPipeline::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS)
        _THROW_ERROR("Failed to create fence");
    if (vkCreateFence(device, &fenceInfo, nullptr, &append_fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to create append fence");
    for (auto& slot : upload_slots)
        if (vkCreateFence(device, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to create upload fence");
}

void VulkanGSPipeline::createQueryPools() {
    // timestamp
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = MAX_TIMESTAMP_QUERY_COUNT;
    if (vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &timestamp_query_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create timestamp query pool");

    timestamp_queries_enabled = false;
    timestamp_period_ns_ = 0.0;
    if (timestamp_query_pool != VK_NULL_HANDLE && physical_device != VK_NULL_HANDLE) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        if (qf_count > 0) {
            vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qf_props.data());
        }
        bool valid_bits = (queue_family_index < qf_count) && (qf_props[queue_family_index].timestampValidBits > 0);
        timestamp_period_ns_ = static_cast<double>(props.limits.timestampPeriod);
        timestamp_queries_enabled = (props.limits.timestampPeriod > 0.0f) && valid_bits;
    }

    // Dedicated lightweight 64-query AGO timestamp pool (independent of PerfTimer).
    VkQueryPoolCreateInfo agoPoolInfo = {};
    agoPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    agoPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    agoPoolInfo.queryCount = kAgoTimestampQueryCount;
    if (vkCreateQueryPool(device, &agoPoolInfo, nullptr, &ago_timestamp_query_pool_) != VK_SUCCESS) {
        ago_timestamp_query_pool_ = VK_NULL_HANDLE;
    }
}

void VulkanGSPipeline::createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule *pShaderModule) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();
    
    if (vkCreateShaderModule(device, &create_info, nullptr, pShaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
}


void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress)
        _THROW_ERROR("Command batch already in progress");
    commandBatchInProgress = true;

    PerfTimer::hostToc();
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin command buffer for batch");

    vkCmdResetQueryPool(command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
    PerfTimer::popMarkers(this);
}

void VulkanGSPipeline::endCommandBatch(bool use_fence) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");

    if (timestampNumWritten > 0) {
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }
    
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        _THROW_ERROR("Failed to end command buffer for batch");
    }
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    
    if (vkQueueSubmit(command_queue, 1, &submit_info,
        use_fence ? fence : VK_NULL_HANDLE) != VK_SUCCESS) {
        _THROW_ERROR("Failed to submit batch");
    }
    
    commandBatchInProgress = false;

    if (use_fence) {
        PerfTimer::work_counters.fence_waits_total++;
      #if SSE2_AVAILABLE
      #if ENABLE_ASSERTION
        constexpr unsigned long long kTimeout = 0x100000000ull;
        auto time0 = __rdtsc();
      #endif
        while (vkGetFenceStatus(device, fence) != VK_SUCCESS) {
            _mm_pause();
          #if ENABLE_ASSERTION
            if (__rdtsc() - time0 >= kTimeout) {
                // _THROW_ERROR("Fence timed out");
                printf("\033[91m%s\033[m\n", "Timed out.");
                std::terminate();  // note that this is often in destructor
            }
          #endif
        }
      #else
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      #endif
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to reset fence");
    }
    else vkQueueWaitIdle(command_queue);

    PerfTimer::hostTic();

    if (timestampNumWritten > 0) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
        double timestampPeriod = deviceProperties.limits.timestampPeriod;

        std::vector<uint64_t> timestamps(timestampNumWritten);
        VkResult qr = vkGetQueryPoolResults(
            device, timestamp_query_pool,
            0, timestampNumWritten,
            sizeof(uint64_t) * timestampNumWritten,
            timestamps.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT
        );
        if (qr == VK_SUCCESS) {
            std::vector<double> times(timestampNumWritten);
            for (uint32_t i = 0; i < timestampNumWritten; i++)
                times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
            auto time_updates = PerfTimer::update(times);
            for (auto& callback : timerCallbacks)
                callback(time_updates);
        } else {
            PerfTimer::clearIncompleteBatch();
        }

        timestampNumWritten = 0;
    }
}

bool VulkanGSPipeline::writeTimestamp(int delta) {
    if (!commandBatchInProgress)
        _THROW_ERROR("writeTimestamp requires command batch in progress");
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        _THROW_ERROR("Too many timestamps written");
    if (delta != 1 && delta != -1)
        _THROW_ERROR("delta in writeTimestamp must be 1 or -1");
    if (delta == -1 && timestampStackDepth == 0)
        _THROW_ERROR("attempt to write exit timestamp while stack is empty");
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten
    );
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

bool VulkanGSPipeline::writeTimestampNoExcept(int delta) {
    if (!commandBatchInProgress)
        return false;
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        return false;
    if (delta != 1 && delta != -1)
        return false;
    if (delta == -1 && timestampStackDepth == 0)
        return false;
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten
    );
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

void VulkanGSPipeline::resetAgoTimestamps(uint32_t first_query, uint32_t query_count) {
    if (!timestamp_queries_enabled || ago_timestamp_query_pool_ == VK_NULL_HANDLE || !commandBatchInProgress) {
        return;
    }
    if (first_query >= kAgoTimestampQueryCount || query_count == 0u) {
        return;
    }
    const uint32_t clamped = std::min(query_count, kAgoTimestampQueryCount - first_query);
    for (uint32_t q = first_query; q < first_query + clamped; ++q) {
        ago_query_written_[q] = false;
    }
    vkCmdResetQueryPool(command_buffer, ago_timestamp_query_pool_, first_query, clamped);
}

void VulkanGSPipeline::writeAgoTimestamp(uint32_t query_index, VkPipelineStageFlagBits stage) {
    if (!timestamp_queries_enabled || ago_timestamp_query_pool_ == VK_NULL_HANDLE || !commandBatchInProgress) {
        return;
    }
    if (query_index >= kAgoTimestampQueryCount) {
        return;
    }
    vkCmdWriteTimestamp(command_buffer, stage, ago_timestamp_query_pool_, query_index);
    ago_query_written_[query_index] = true;
}

std::optional<double> VulkanGSPipeline::readTimestampDeltaMsOptional(uint32_t start_query, uint32_t end_query) const {
    if (!timestamp_queries_enabled || ago_timestamp_query_pool_ == VK_NULL_HANDLE ||
        timestamp_period_ns_ <= 0.0 ||
        start_query >= kAgoTimestampQueryCount || end_query >= kAgoTimestampQueryCount ||
        end_query <= start_query ||
        !ago_query_written_[start_query] || !ago_query_written_[end_query]) {
        return std::nullopt;
    }
    const uint32_t count = (end_query - start_query) + 1u;
    std::vector<uint64_t> values(count, 0u);
    const VkResult qr = vkGetQueryPoolResults(
        device,
        ago_timestamp_query_pool_,
        start_query,
        count,
        sizeof(uint64_t) * count,
        values.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );
    if (qr != VK_SUCCESS) {
        return std::nullopt;
    }
    const uint64_t t0 = values.front();
    const uint64_t t1 = values.back();
    if (t1 < t0) {
        return std::nullopt;
    }
    return static_cast<double>(t1 - t0) * timestamp_period_ns_ * 1e-6;
}

double VulkanGSPipeline::readTimestampDeltaMs(uint32_t start_query, uint32_t end_query) const {
    return readTimestampDeltaMsOptional(start_query, end_query).value_or(0.0);
}

void VulkanGSPipeline::beginDebugLabel(const char* name, float r, float g, float b, float a) {
    if (!pfn_cmd_begin_debug_label_ || !commandBatchInProgress || !name) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    pfn_cmd_begin_debug_label_(command_buffer, &label);
}

void VulkanGSPipeline::endDebugLabel() {
    if (!pfn_cmd_end_debug_label_ || !commandBatchInProgress) {
        return;
    }
    pfn_cmd_end_debug_label_(command_buffer);
}


VkAccessFlags toAccessMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkAccessFlags result = (VkAccessFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_SHADER_WRITE_BIT;
    return result;
}

VkPipelineStageFlags toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags result = (VkPipelineStageFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_HOST_BIT;
    return result;
}

void VulkanGSPipeline::memoryBarrier(
    VulkanGSPipeline::BarrierMask srcMask,
    VulkanGSPipeline::BarrierMask dstMask
) {
    if (!commandBatchInProgress)
        return;

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = toAccessMask(srcMask);
    barrier.dstAccessMask = toAccessMask(dstMask);
    
    vkCmdPipelineBarrier(
        command_buffer,
        toStageMask(srcMask), toStageMask(dstMask),
        0, // dependencyFlags
        1, &barrier, // memory barriers
        0, nullptr, // buffer barriers
        0, nullptr // image barriers
    );
}

void VulkanGSPipeline::bufferMemoryBarrier(
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>> &buffers,
    VulkanGSPipeline::BarrierMask dstMask
) {
    if (!commandBatchInProgress)
        return;

    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    VkPipelineStageFlags srcStageFlags = (VkPipelineStageFlags)0;
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = toAccessMask(srcMask);
        barrier.dstAccessMask = toAccessMask(dstMask);
        barrier.srcQueueFamilyIndex = queue_family_index;
        barrier.dstQueueFamilyIndex = queue_family_index;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = buffer.size;
        barriers.push_back(barrier);
        srcStageFlags |= toStageMask(srcMask);
    }
    if (barriers.empty())
        return;
    
    vkCmdPipelineBarrier(
        command_buffer,
        srcStageFlags, toStageMask(dstMask),
        0, // dependencyFlags
        0, nullptr, // memory barriers
        (uint32_t)barriers.size(), barriers.data(), // buffer barriers
        0, nullptr // image barriers
    );
}


// Compute pipeline

void VulkanGSPipeline::createComputeDescriptorSetLayout(_ComputePipeline &pipeline) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(pipeline.buffer_layouts.size());
    
    for (int i : pipeline.buffer_layouts) {
        VkDescriptorSetLayoutBinding binding;
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;
        bindings.push_back(binding);
    }
    
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pipeline.descriptor_set_layout) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor set layout");
}

void VulkanGSPipeline::createComputeDescriptorPool(_ComputePipeline &pipeline) {
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (uint32_t)(pipeline.buffer_layouts.size());
    
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &pipeline.descriptor_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor pool");
}

void VulkanGSPipeline::updateComputeDescriptorSet(_ComputePipeline &pipeline, const std::vector<_VulkanBuffer> &data_buffers) {
    if (pipeline.descriptor_set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pipeline.descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &pipeline.descriptor_set_layout;
        
        if (vkAllocateDescriptorSets(device, &alloc_info, &pipeline.descriptor_set) != VK_SUCCESS)
            _THROW_ERROR("Failed to allocate descriptor sets while updating");
    }
    
    size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkWriteDescriptorSet> descriptor_writes(num_buffers);
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    
    int idx = 0;
    for (int i : pipeline.buffer_layouts) {
        if (data_buffers[i].buffer == VK_NULL_HANDLE)
            _THROW_ERROR("Buffer " + std::to_string(i) + " is NULL");
        buffer_infos[idx].buffer = data_buffers[i].buffer;
        buffer_infos[idx].offset = 0;
        // buffer_infos[idx].range = data_buffers[i].size;
        buffer_infos[idx].range = data_buffers[i].allocSize;
        
        descriptor_writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[idx].dstSet = pipeline.descriptor_set;
        descriptor_writes[idx].dstBinding = i;
        descriptor_writes[idx].dstArrayElement = 0;
        descriptor_writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[idx].descriptorCount = 1;
        descriptor_writes[idx].pBufferInfo = &buffer_infos[idx];
        
        idx++;
    }
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    if (min_shared_memory > this->deviceInfo.sharedSize) {
        pipeline.shader = VK_NULL_HANDLE;
        return;
    }

    createShaderModule(loadSpirv(spirv_path), &pipeline.shader);
    createComputeDescriptorSetLayout(pipeline);

    // Create push constant range for uniforms
    VkPushConstantRange push_constant_range = {};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = (uint32_t)MAX_UNIFORM_SIZE;
    
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create pipeline set layout");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};
    req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    req.requiredSubgroupSize = SUBGROUP_SIZE;  // 32

    VkPipelineShaderStageCreateInfo compute_shader_stage_info = {};
    compute_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_shader_stage_info.module = pipeline.shader;
    compute_shader_stage_info.pName = "main";
    if (compatible_subgroup_size && (
        deviceInfo.subgroupSize != SUBGROUP_SIZE ||
        deviceInfo.vendor == DeviceVendor::Intel_R_
    ))
        compute_shader_stage_info.pNext = &req;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline.pipeline_layout;
    pipeline_info.stage = compute_shader_stage_info;
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
        _THROW_ERROR("Failed to create compute pipeline");

    createComputeDescriptorPool(pipeline);

    all_compute_pipelines.push_back(&pipeline);
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers,
    int gate_slot,
    const _VulkanBuffer* indirect
) {
    if (uniformSize > MAX_UNIFORM_SIZE)
        _THROW_ERROR("Maximum uniform size exceeded");

    if (pipeline.buffers != buffers) {
        HOST_GUARD;
        updateComputeDescriptorSet(pipeline, buffers);
        pipeline.buffers = buffers;
    }

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1, &pipeline.descriptor_set, 0, nullptr);
    
    // Push constants for uniforms
    if (uniformsPtr) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr
        );
    }
    
    if (indirect) {
        vkCmdDispatchIndirect(command_buffer, indirect->buffer, 0);
        return;
    }

    // Dispatch compute shader
    while (dims.size() < 3)
        dims.push_back({1, 1});
    uint32_t nGroupsX = (uint32_t)_CEIL_DIV(dims[0].first, dims[0].second);
    uint32_t nGroupsY = (uint32_t)_CEIL_DIV(dims[1].first, dims[1].second);
    uint32_t nGroupsZ = (uint32_t)_CEIL_DIV(dims[2].first, dims[2].second);
    if (gate_active && gate_slot >= 0) {
        // the gate wrote the arguments from the host plan (or from the GPU intersection count for the
        // sort and tile-range slots); a plan that disagrees with the site would silently mis-dispatch
        if (gate_slot != GATE_SORT && gate_slot != GATE_TILE_RANGES) {
            const uint32_t* b = gate_plan.base[gate_slot];
            if (b[0] != nGroupsX || b[1] != nGroupsY || b[2] != nGroupsZ)
                throw std::runtime_error("step gate plan does not match the dispatch of slot " + std::to_string(gate_slot));
        }
        vkCmdDispatchIndirect(command_buffer, gate_args.buffer,
            ((VkDeviceSize)gate_step * kGateSlots + (VkDeviceSize)gate_slot) * 16);
        return;
    }
    if (nGroupsX > deviceInfo.maxGroupsX ||
        nGroupsY > deviceInfo.maxGroupsY ||
        nGroupsZ > deviceInfo.maxGroupsZ
    ) _THROW_ERROR("Cannot launch compute kernel, too many groups: [" +
            std::to_string(nGroupsX) + " " +
            std::to_string(nGroupsY) + " " +
            std::to_string(nGroupsZ) + "] > [" +
            std::to_string(deviceInfo.maxGroupsX) + " " +
            std::to_string(deviceInfo.maxGroupsY) + " " +
            std::to_string(deviceInfo.maxGroupsZ) + "]"
        );
    vkCmdDispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::executeComputeIndirect(
    const DeviceBuffer& indirect_buf,
    VkDeviceSize indirect_offset,
    const void* push_data,
    size_t push_size,
    _ComputePipeline& pipeline,
    const std::vector<DeviceBuffer>& buffers
) {
    if (push_size > MAX_UNIFORM_SIZE) {
        _THROW_ERROR("Maximum uniform size exceeded");
    }
    if (indirect_buf.buffer == VK_NULL_HANDLE) {
        _THROW_ERROR("executeComputeIndirect requires a valid indirect buffer");
    }

    if (pipeline.buffers != buffers) {
        HOST_GUARD;
        updateComputeDescriptorSet(pipeline, buffers);
        pipeline.buffers = buffers;
    }

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout,
        0,
        1,
        &pipeline.descriptor_set,
        0,
        nullptr
    );
    if (push_data != nullptr && push_size > 0) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            static_cast<uint32_t>(push_size),
            push_data
        );
    }
    vkCmdDispatchIndirect(command_buffer, indirect_buf.buffer, indirect_offset);
}


void VulkanGSPipeline::destroyComputePipeline(_ComputePipeline &pipeline) {
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pipeline.descriptor_pool, nullptr);
        pipeline.descriptor_pool = VK_NULL_HANDLE;
    }
    if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, pipeline.descriptor_set_layout, nullptr);
        pipeline.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, pipeline.shader, nullptr);
        pipeline.shader = VK_NULL_HANDLE;
    }
}
