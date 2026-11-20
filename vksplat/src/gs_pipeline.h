#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>  // std::sort
#include <cstring>  // memcpy
#include <functional>
#include <array>
#include <variant>


#include <cassert>

#include "buffer.h"

union Uniform32_t {
    uint32_t u;
    float f;
};


class VulkanGSPipeline {
public:
    VulkanGSPipeline();
    ~VulkanGSPipeline();

    void initialize(int device_id);
    void cleanup();
    void cleanupBuffers(VulkanGSPipelineBuffers& buffers);

    void createBuffer(size_t size, _VulkanBuffer& buffer);
    void destroyBuffer(_VulkanBuffer &buffer);
    void resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink=true);
    template<typename T> _VulkanBuffer& resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink=true);
    // Grow-only resize with geometric capacity: the logical size becomes new_size, the allocation is
    // reallocated (contents not preserved) only when it is too small, to max(needed, 1.5 * old).
    template<typename T> _VulkanBuffer& reserveDeviceBuffer(Buffer<T>& buffer, size_t new_size);
    template<typename T> _VulkanBuffer& clearDeviceBuffer(Buffer<T>& buffer, size_t new_size);
    template<typename T> _VulkanBuffer& resizeAndCopyDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool clear);
    template<typename T> _VulkanBuffer& copyToDevice(Buffer<T>& buffer);
    template<typename T> void copyFromDevice(Buffer<T>& buffer);
    void uploadDeviceSlice(_VulkanBuffer& deviceBuffer, size_t dst_byte_offset, const void* src_data, size_t byte_size);
    void downloadDeviceSlice(const _VulkanBuffer& deviceBuffer, size_t src_byte_offset, void* dst_data, size_t byte_size);
    void copyFromDeviceToDevice(const _VulkanBuffer& src, _VulkanBuffer& dst);
    template<typename T> void copyFromDeviceToDevice(const Buffer<T>& src, Buffer<T>& dst);
    template<typename T> T readElement(const _VulkanBuffer& buffer, size_t index);
    template<typename T> void dumpRawBytesToFile(Buffer<T>& buffer, std::string filename);

    // Fence-free transfer path used by append: records into a dedicated command buffer and host staging
    // buffer, and is submitted without waiting. Queue order serializes it before the next submit.
    // Returns the mapped staging pointer; the previous append is waited on only if still running.
    uint8_t* beginAppendRecording(size_t staging_bytes);
    void submitAppendRecording();
    VkCommandBuffer appendCommandBuffer() const { return append_cb; }
    VkBuffer appendStagingBuffer() const { return append_stage.buffer; }
    // Asynchronous keyframe upload: copies `byte_size` bytes once into a persistent host-visible staging ring
    // slot and submits a dedicated transfer command buffer without a host wait (queue order plus a trailing
    // barrier serialize it before the next submit). A slot is reused only after its fence has signalled; a
    // wait is counted in `frame_upload_waits`. `dst` must already have capacity for `byte_size`.
    void uploadDeviceAsync(_VulkanBuffer& dst, const void* src, size_t byte_size);
    void waitUploads();
    bool async_frame_upload = true;
    // Destroys buffers replaced by an append once the GPU is done with them.
    void retireDeviceBuffer(const _VulkanBuffer& old_buffer) { retired_buffers.push_back(old_buffer); }

    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true);
    void waitIdle() { if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device); }
    bool isCommandBatchInProgress() const {
        return commandBatchInProgress;
    }
    bool writeTimestamp(int delta);
    bool writeTimestampNoExcept(int delta);
    // Batched steps: only the first step of a batch records the full stage breakdown, the others only the
    // timers the backward-variant scheduler needs (the query pool is small).
    bool timestamp_scheduling_only = false;

    // GPU step gate for batched training steps (see step_gate.slang). Dispatches that depend on the tile
    // intersection count, or that mutate state after it, are dispatched indirectly through `gate_args`, so a
    // step whose count exceeds the buffer capacity (or follows such a step) becomes a no-op on the GPU.
    static constexpr uint32_t kGateSlots = 8;
    static constexpr uint32_t kGateMaxSteps = 8;
    enum GateSlot {
        GATE_GENERATE_KEYS = 0, GATE_SORT = 1, GATE_TILE_RANGES = 2, GATE_RASTER_FWD = 3,
        GATE_RASTER_BWD = 4, GATE_OPTIMIZER = 5, GATE_POST = 6,
    };
    // group counts of the host-known gate slots (cross-checked against the dispatch site)
    struct GatePlan { uint32_t base[kGateSlots][4] = {}; };
    bool gateActive() const { return gate_active; }

    size_t getPeakAllocSize() const
        { return peak_vram; }

    enum BarrierMask {
        TRANSFER_READ,
        TRANSFER_WRITE,
        TRANSFER_READ_WRITE,
        COMPUTE_SHADER_READ,
        COMPUTE_SHADER_WRITE,
        COMPUTE_SHADER_READ_WRITE,
        TRANSFER_COMPUTE_SHADER_READ,
        TRANSFER_COMPUTE_SHADER_WRITE,
        TRANSFER_COMPUTE_SHADER_READ_WRITE,
        HOST_READ,
        HOST_WRITE,
        HOST_READ_WRITE,
    };

    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> get_device_info() const;

protected:

    struct DeviceRequirement {
        std::array<uint32_t, 3> minMaxGroups;
        std::array<uint32_t, 3> minMaxThreads;
        uint32_t minSharedMemory;
    };

    virtual DeviceRequirement getDeviceRequirement() = 0;

    bool enableValidationLayer = false;

    bool commandBatchInProgress = false;
    uint32_t timestampNumWritten = 0;
    uint32_t timestampStackDepth = 0;
    std::vector<std::function<void(const std::vector<std::pair<size_t, double>>&)>> timerCallbacks;

    void memoryBarrier(BarrierMask srcMask, BarrierMask dstMask);
    void bufferMemoryBarrier(const std::vector<std::pair<_VulkanBuffer, BarrierMask>> &buffers, BarrierMask dstMask);

    size_t current_vram = 0;
    size_t peak_vram = 0;

    // Vulkan objects
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue command_queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkCommandBuffer append_cb = VK_NULL_HANDLE;
    VkFence append_fence = VK_NULL_HANDLE;
    bool append_pending = false;
    struct _AppendStage {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint8_t* mapped = nullptr;
        size_t allocSize = 0;
    };
    _AppendStage append_stage;
    std::vector<_VulkanBuffer> retired_buffers;
    static constexpr int kUploadSlots = 3;
    struct _UploadSlot {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint8_t* mapped = nullptr;
        size_t allocSize = 0;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool pending = false;
    };
    _UploadSlot upload_slots[kUploadSlots];
    int upload_cursor = 0;
    void waitUploadSlot(_UploadSlot& slot);
    void destroyUploadStage(_UploadSlot& slot);
    void createHostStaging(size_t size, VkBuffer& buffer, VkDeviceMemory& memory, uint8_t*& mapped);
    void waitAppendFence();
    void freeAppendStage();
    VkQueryPool timestamp_query_pool;

    enum class DeviceVendor {
        Unknown,
        NVIDIA, AMD, Intel_R_, ARM, Qualcomm, 
    };
    struct DeviceInfo {
        uint32_t subgroupSize;
        uint32_t sharedSize;
        uint32_t maxGroupsX;
        uint32_t maxGroupsY;
        uint32_t maxGroupsZ;
        uint32_t maxThreadsX;
        uint32_t maxThreadsY;
        uint32_t maxThreadsZ;
        bool hasInt16;
        bool hasInt64;
        bool hasFloat32AtomicAdd;
        DeviceVendor vendor;
        uint32_t vendorId;
        std::string name;
    } deviceInfo;
    
    // Compute pipeline
    struct _ComputePipeline {
        VkShaderModule shader;
        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        std::vector<int> buffer_layouts;

        _ComputePipeline(
            std::vector<int> buffer_layouts
        ): shader(VK_NULL_HANDLE), descriptor_pool(VK_NULL_HANDLE),
            descriptor_set_layout(VK_NULL_HANDLE), descriptor_set(VK_NULL_HANDLE),
            pipeline_layout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE),
            buffer_layouts(buffer_layouts) {}

        _ComputePipeline(int num_buffers)
        : shader(VK_NULL_HANDLE), descriptor_pool(VK_NULL_HANDLE),
            descriptor_set_layout(VK_NULL_HANDLE), descriptor_set(VK_NULL_HANDLE),
            pipeline_layout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE)
        {
            buffer_layouts.resize(num_buffers);
            for (int i = 0; i < num_buffers; i++)
                buffer_layouts[i] = i;
        }

        std::vector<_VulkanBuffer> buffers;
    };

    struct _ComputePipelinePair {
        _ComputePipeline _cp0, _cp1;
        _ComputePipelinePair(int num_buffers)
            : _cp0(num_buffers), _cp1(num_buffers) {}
        _ComputePipeline& operator[](bool b)
            { return b ? _cp0 : _cp1; }
    };

    std::vector<_ComputePipeline*> all_compute_pipelines;

    uint32_t queue_family_index;

    // For CPU-GPU transfers
    struct _Stager {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t allocSize = 0;
        std::mutex mutex;
    };
    _Stager stager;

    void allocStagingBuffer(size_t size);
    
    void createInstance();
    void selectPhysicalDevice(int device_id);
    void createDevice();
    void createCommandPool();
    void createFence();
    void createQueryPools();
    void createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule *pShaderModule);

    void createComputeDescriptorSetLayout(_ComputePipeline &pipeline);
    void createComputeDescriptorPool(_ComputePipeline &pipeline);
    void updateComputeDescriptorSet(_ComputePipeline &pipeline, const std::vector<_VulkanBuffer> &buffers);
    void createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory=0, bool compatible_subgroup_size=true);
    // gate_slot >= 0 and gate active: dispatch indirectly from the step gate's arguments for that slot
    void executeCompute(
        std::vector<std::pair<size_t, size_t>> dims,
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline &pipeline,
        const std::vector<_VulkanBuffer> &buffers,
        int gate_slot = -1,
        const _VulkanBuffer* indirect = nullptr  // dispatch from this buffer (VkDispatchIndirectCommand at offset 0)
    );

    bool gate_active = false;
    uint32_t gate_step = 0;  // step index inside the current batch
    GatePlan gate_plan;
    _VulkanBuffer gate_args;     // kGateMaxSteps x kGateSlots VkDispatchIndirectCommand (16 B stride)
    _VulkanBuffer gate_status;   // see step_gate.slang
    _VulkanBuffer isect_count;   // (1,) tile intersection count read by sort / tile-range kernels
    VkBuffer gate_host_buffer = VK_NULL_HANDLE;
    VkDeviceMemory gate_host_memory = VK_NULL_HANDLE;
    const volatile uint32_t* gate_host_status = nullptr;
    static constexpr uint32_t kGateStatusWords = 4 + kGateMaxSteps;
    void createGateBuffers();
    void destroyGateBuffers();
    void fullBarrier();

    void _displayImage(Buffer<float>& pixels, int width, bool reverse_alpha);

private:
    void destroyComputePipeline(_ComputePipeline &pipeline);
};


class [[nodiscard]] DeviceGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
public:
    DeviceGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("DeviceGuard constructor\n");
        this->pipeline = pipeline;
        cbip = pipeline->isCommandBatchInProgress();
        if (!cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("DeviceGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
    }
    ~DeviceGuard() noexcept(false) {
        // printf("DeviceGuard destructor\n");
        if (!cbip) {
            pipeline->endCommandBatch();
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during DeviceGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }

    }
};

class [[nodiscard]] HostGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
public:
    HostGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("HostGuard constructor\n");
        this->pipeline = pipeline;
        cbip = pipeline->isCommandBatchInProgress();
        if (cbip) {
            pipeline->endCommandBatch();
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("HostGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
    }
    ~HostGuard() noexcept(false) {
        // printf("HostGuard destructor\n");
        if (cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during HostGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }

    }
};

// #define DeviceGuard(args) DeviceGuard(args, __FILE__, __LINE__)
// #define HostGuard(args) HostGuard(args, __FILE__, __LINE__)

#define DEVICE_GUARD auto deviceGuard = DeviceGuard(this)
#define HOST_GUARD auto hostGuard = HostGuard(this)
