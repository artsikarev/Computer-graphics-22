#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "application.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <imgui.h>
#include <cstddef>
#include <algorithm>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace application {
namespace {
float rotation_angle = 0.0f;
float rotation_speed = 1.0f;
float cube_position_x = 0.0f;
float cube_position_y = 0.0f;
float cube_position_z = 0.0f;
float cube_scale = 1.0f;
bool use_perspective = true;

// Положение камеры в мировом пространстве
float camera_x = 0.0f;
float camera_y = 0.0f;
float camera_z = 0.0f;

GLFWwindow* app_window = nullptr;

float camera_move_speed = 2.5f;

// Время предыдущего кадра
double previous_frame_time = 0.0;

// Углы поворота камеры, в радианах
float camera_yaw = -1.5707963f;   // -90 градусов: смотрим вдоль -Z
float camera_pitch = 0.0f;

// Угол обзора перспективной камеры
float camera_fov = 1.0471976f;    // 60 градусов

struct Mat4 {
    float m[16]{};
};

Mat4 identityMatrix() {
    Mat4 result{};
    result.m[0] = 1.0f;
    result.m[5] = 1.0f;
    result.m[10] = 1.0f;
    result.m[15] = 1.0f;
    return result;
}

Mat4 scaleMatrix(float scale) {
    Mat4 result{};
    result.m[0] = scale;
    result.m[5] = scale;
    result.m[10] = scale;
    result.m[15] = 1.0f;
    return result;
}


Mat4 multiplyMatrices(const Mat4& a, const Mat4& b) {
    Mat4 result{};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result.m[column * 4 + row] +=
                    a.m[k * 4 + row] * b.m[column * 4 + k];
            }
        }
    }

    return result;
}


Mat4 rotationY(float angle) {
    Mat4 result = identityMatrix();

    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.m[0] = c;
    result.m[2] = -s;
    result.m[8] = s;
    result.m[10] = c;

    return result;
}


Mat4 translationMatrix(float x, float y, float z) {
    Mat4 result = identityMatrix();

    result.m[12] = x;
    result.m[13] = y;
    result.m[14] = z;

    return result;
}


Mat4 perspectiveMatrix(float fov, float aspect, float near_plane, float far_plane);
Mat4 orthographicMatrix(float left, float right, float bottom, float top, float near_plane, float far_plane);

Mat4 createModel(float angle) {
    return multiplyMatrices(
        translationMatrix(
            cube_position_x,
            cube_position_y,
            cube_position_z - 2.0f
        ),
        multiplyMatrices(
            rotationY(angle),
            scaleMatrix(cube_scale)
        )
    );
}

struct Vec3 {
    float x;
    float y;
    float z;
};

struct LightingSettings {
    // Направление от поверхности к источнику света
    float light_direction[3] = { -1.0f, 1.0f, 1.0f };
    float light_color[3] = { 1.0f, 1.0f, 1.0f };
    float light_intensity = 1.0f;
    float point_position[3] = { 2.0f, 2.0f, 2.0f };
    float point_color[3] = { 1.0f, 0.4f, 0.2f };
    float point_intensity = 5.0f;
    bool point_enabled = true;
    float spot_position[3] = { 0.0f, 2.0f, 2.0f };
    float spot_direction[3] = { 0.0f, -1.0f, -1.0f };
    float spot_color[3] = { 0.3f, 0.6f, 1.0f };
    float spot_intensity = 8.0f;
    float spot_inner_angle = 15.0f;
    float spot_outer_angle = 25.0f;
    bool spot_enabled = true;

    // Фоновое освещение
    float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
    float ambient_intensity = 0.15f;

    // Материал куба
    float albedo[3] = { 0.2f, 0.7f, 1.0f };
    float specular_color[3] = { 1.0f, 1.0f, 1.0f };
    float shininess = 32.0f;
};

constexpr uint32_t MAX_POINT_LIGHTS = 8;
constexpr uint32_t MAX_SPOT_LIGHTS = 8;

struct alignas(16) PointLightGPU {
    alignas(16) float position[4];
    alignas(16) float color_intensity[4];
};

struct alignas(16) SpotLightGPU {
    alignas(16) float position[4];
    alignas(16) float direction[4];
    alignas(16) float color_intensity[4];
    alignas(16) float angles[4];
};

struct alignas(16) PointLightsBuffer {
    alignas(16) uint32_t count;
    uint32_t padding[3]{};
    PointLightGPU lights[MAX_POINT_LIGHTS]{};
};

struct alignas(16) SpotLightsBuffer {
    alignas(16) uint32_t count;
    uint32_t padding[3]{};
    SpotLightGPU lights[MAX_SPOT_LIGHTS]{};
};

static_assert(sizeof(PointLightGPU) == 32);
static_assert(sizeof(SpotLightGPU) == 64);
static_assert(offsetof(PointLightsBuffer, lights) == 16);
static_assert(offsetof(SpotLightsBuffer, lights) == 16);

struct alignas(16) LightingUniform {
    alignas(16) float light_direction[4];
    alignas(16) float light_color_intensity[4];
    alignas(16) float ambient_color_intensity[4];
    alignas(16) float albedo[4];
    alignas(16) float specular_color_shininess[4];
    alignas(16) float camera_position[4];

    alignas(16) float point_position[4];
    alignas(16) float point_color_intensity[4];
};

static_assert(sizeof(LightingUniform) == 128);

LightingSettings lighting;

Vec3 subtractVec3(Vec3 a, Vec3 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3 normalizeVec3(Vec3 v) {
    const float length = std::sqrt(
        v.x * v.x + v.y * v.y + v.z * v.z
    );

    if (length < 0.000001f) {
        return { 0.0f, 0.0f, 0.0f };
    }

    return {
        v.x / length,
        v.y / length,
        v.z / length
    };
}

Vec3 crossVec3(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float dotVec3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Mat4 lookAtMatrix(Vec3 eye, Vec3 target, Vec3 worldUp) {
    const Vec3 forward = normalizeVec3(
        subtractVec3(target, eye)
    );

    const Vec3 right = normalizeVec3(
        crossVec3(forward, worldUp)
    );

    const Vec3 up = crossVec3(right, forward);

    Mat4 result = identityMatrix();

    result.m[0] = right.x;
    result.m[1] = up.x;
    result.m[2] = -forward.x;

    result.m[4] = right.y;
    result.m[5] = up.y;
    result.m[6] = -forward.y;

    result.m[8] = right.z;
    result.m[9] = up.z;
    result.m[10] = -forward.z;

    result.m[12] = -dotVec3(right, eye);
    result.m[13] = -dotVec3(up, eye);
    result.m[14] = dotVec3(forward, eye);

    return result;
}

Mat4 createMvp(float angle, float aspect, bool perspective) {
    // 1. Модельная матрица: положение, поворот и размер куба
    const Mat4 model = multiplyMatrices(
        translationMatrix(
            cube_position_x,
            cube_position_y,
            cube_position_z - 2.0f
        ),
        multiplyMatrices(
            rotationY(angle),
            scaleMatrix(cube_scale)
        )
    );

    // 2. Направление взгляда камеры
    const Vec3 camera_position = {
        camera_x,
        camera_y,
        camera_z
    };

    const Vec3 camera_direction = {
        std::cos(camera_yaw) * std::cos(camera_pitch),
        std::sin(camera_pitch),
        std::sin(camera_yaw) * std::cos(camera_pitch)
    };

    const Vec3 camera_target = {
        camera_position.x + camera_direction.x,
        camera_position.y + camera_direction.y,
        camera_position.z + camera_direction.z
    };

    // 3. View-матрица: преобразуем мир относительно камеры
    const Mat4 view = lookAtMatrix(
        camera_position,
        camera_target,
        { 0.0f, 1.0f, 0.0f }
    );

    // 4. Projection-матрица
    Mat4 projection{};

    if (perspective) {
        projection = perspectiveMatrix(
            camera_fov,
            aspect,
            0.1f,
            100.0f
        );
    } else {
        const float half_height = 1.5f;
        const float half_width = half_height * aspect;

        projection = orthographicMatrix(
            -half_width,
            half_width,
            -half_height,
            half_height,
            0.1f,
            100.0f
        );
    }

    // Итог: сначала модель, затем камера, затем проекция
    return multiplyMatrices(
        projection,
        multiplyMatrices(view, model)
    );
}


Mat4 perspectiveMatrix(float fov, float aspect, float near_plane, float far_plane) {
    Mat4 result{};

    const float f = 1.0f / std::tan(fov * 0.5f);

    result.m[0] = f / aspect;
    result.m[5] = -f;
    result.m[10] = far_plane / (near_plane - far_plane);
    result.m[11] = -1.0f;
    result.m[14] =
        (near_plane * far_plane) / (near_plane - far_plane);

    return result;
}


Mat4 orthographicMatrix(
    float left,
    float right,
    float bottom,
    float top,
    float near_plane,
    float far_plane) {

    Mat4 result = identityMatrix();

    result.m[0] = 2.0f / (right - left);
    result.m[5] = -2.0f / (top - bottom);
    result.m[10] = 1.0f / (near_plane - far_plane);

    result.m[12] = -(right + left) / (right - left);
    result.m[13] = (top + bottom) / (top - bottom);
    result.m[14] = near_plane / (near_plane - far_plane);

    return result;
}



VkShaderModule vertex_shader = VK_NULL_HANDLE;
VkShaderModule fragment_shader = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

VkBuffer lighting_buffer = VK_NULL_HANDLE;
VmaAllocation lighting_buffer_allocation = VK_NULL_HANDLE;
void* lighting_buffer_mapped = nullptr;

VkBuffer point_lights_buffer = VK_NULL_HANDLE;
VmaAllocation point_lights_buffer_allocation = VK_NULL_HANDLE;
void* point_lights_buffer_mapped = nullptr;

VkDescriptorSetLayout lighting_descriptor_set_layout =
    VK_NULL_HANDLE;

VkDescriptorPool lighting_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet lighting_descriptor_set = VK_NULL_HANDLE;

VkDescriptorSetLayout point_lights_descriptor_set_layout =
    VK_NULL_HANDLE;

VkDescriptorPool point_lights_descriptor_pool =
    VK_NULL_HANDLE;

VkDescriptorSet point_lights_descriptor_set =
    VK_NULL_HANDLE;

VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
VmaAllocation spot_lights_buffer_allocation = VK_NULL_HANDLE;
void* spot_lights_buffer_mapped = nullptr;

VkDescriptorSetLayout spot_lights_descriptor_set_layout =
    VK_NULL_HANDLE;

VkDescriptorPool spot_lights_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet spot_lights_descriptor_set = VK_NULL_HANDLE;

struct Vertex {
    float position[3];
    float normal[3];
};


const VkVertexInputBindingDescription vertex_binding = {
    .binding = 0,
    .stride = sizeof(Vertex),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
};

const VkVertexInputAttributeDescription vertex_attributes[] = {

    // location = 0: координаты вершины
    {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, position),
    },

    // location = 1: нормаль вершины
    {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, normal),
    }

};

std::vector<char> readShaderFile(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return {};
    }

    const size_t file_size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);

    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    if (code.empty() || code.size() % 4 != 0) {
        return VK_NULL_HANDLE;
    }

    const VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            graphics::internal::context.device,
            &create_info,
            nullptr,
            &shader_module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return shader_module;
}

bool createGraphicsPipeline() {
    auto& context = graphics::internal::context;

    if (vertex_shader == VK_NULL_HANDLE ||
        fragment_shader == VK_NULL_HANDLE) {
        return false;
    }

    const VkPipelineShaderStageCreateInfo shader_stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader,
            .pName = "main",
        },
    };

const VkPipelineVertexInputStateCreateInfo vertex_input = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &vertex_binding,
    .vertexAttributeDescriptionCount = 2,
    .pVertexAttributeDescriptions = vertex_attributes,
};

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };


    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };


    const VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(float) * 32,
    };


const VkDescriptorSetLayout set_layouts[] = {
    lighting_descriptor_set_layout,       // set = 0: UBO
    point_lights_descriptor_set_layout,   // set = 1: Point Light SSBO
    spot_lights_descriptor_set_layout     // set = 2: Spot Light SSBO
};


const VkPipelineLayoutCreateInfo pipeline_layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 3,
    .pSetLayouts = set_layouts,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_constant_range,
};

    if (vkCreatePipelineLayout(
            graphics::internal::context.device,
            &pipeline_layout_info,
            nullptr,
            &pipeline_layout) != VK_SUCCESS) {
        return false;
    }


    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };


    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };


    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };



    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = graphics::internal::context.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(
            graphics::internal::context.device,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &graphics_pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(
            graphics::internal::context.device,
            pipeline_layout,
            nullptr);
        pipeline_layout = VK_NULL_HANDLE;
        return false;
    }

    return true;

}

const std::array<Vertex, 24> cube_positions = {{

    // Задняя грань
    {{-0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f, -0.5f}, {0, 0, 0}},
    {{-0.5f,  0.5f, -0.5f}, {0, 0, 0}},

    // Передняя грань
    {{-0.5f, -0.5f,  0.5f}, {0, 0, 0}},
    {{ 0.5f, -0.5f,  0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f,  0.5f}, {0, 0, 0}},
    {{-0.5f,  0.5f,  0.5f}, {0, 0, 0}},

    // Нижняя грань
    {{-0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f, -0.5f,  0.5f}, {0, 0, 0}},
    {{-0.5f, -0.5f,  0.5f}, {0, 0, 0}},

    // Верхняя грань
    {{-0.5f,  0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f,  0.5f}, {0, 0, 0}},
    {{-0.5f,  0.5f,  0.5f}, {0, 0, 0}},

    // Левая грань
    {{-0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{-0.5f, -0.5f,  0.5f}, {0, 0, 0}},
    {{-0.5f,  0.5f,  0.5f}, {0, 0, 0}},
    {{-0.5f,  0.5f, -0.5f}, {0, 0, 0}},

    // Правая грань
    {{ 0.5f, -0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f, -0.5f}, {0, 0, 0}},
    {{ 0.5f,  0.5f,  0.5f}, {0, 0, 0}},
    {{ 0.5f, -0.5f,  0.5f}, {0, 0, 0}}

}};

const std::array<uint32_t, 36> cube_indices = {{
    // Задняя грань
    0, 2, 1,
    0, 3, 2,
    // Передняя грань
    4, 5, 6,
    4, 6, 7,
    // Нижняя грань
    8, 10, 9,
    8, 11, 10,
    // Верхняя грань
    12, 13, 14,
    12, 14, 15,
    // Левая грань
    16, 17, 18,
    16, 18, 19,
    // Правая грань
    20, 21, 22,
    20, 22, 23

}};

const std::array<Vertex, 24> cube_vertices = [] {
    auto vertices = cube_positions;

    // Для каждой грани используются 2 треугольника,
    // то есть 6 индексов.
    for (size_t face = 0; face < 6; ++face) {
        const size_t base = face * 6;

        const uint32_t i0 = cube_indices[base];
        const uint32_t i1 = cube_indices[base + 1];
        const uint32_t i2 = cube_indices[base + 2];

        const Vec3 p0 = {
            vertices[i0].position[0],
            vertices[i0].position[1],
            vertices[i0].position[2]
        };

        const Vec3 p1 = {
            vertices[i1].position[0],
            vertices[i1].position[1],
            vertices[i1].position[2]
        };

        const Vec3 p2 = {
            vertices[i2].position[0],
            vertices[i2].position[1],
            vertices[i2].position[2]
        };

        const Vec3 edge1 = subtractVec3(p1, p0);
        const Vec3 edge2 = subtractVec3(p2, p0);

        const Vec3 normal = normalizeVec3(
            crossVec3(edge1, edge2)
        );

        // Одна плоская нормаль на всю грань.
        for (size_t j = 0; j < 6; ++j) {
            const uint32_t index = cube_indices[base + j];

            vertices[index].normal[0] = normal.x;
            vertices[index].normal[1] = normal.y;
            vertices[index].normal[2] = normal.z;
        }
    }

    return vertices;
}();

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;

bool createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    auto& context = graphics::internal::context;

    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocation_info = {
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
             VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO,
};

    return vmaCreateBuffer(
        context.allocator,
        &buffer_info,
        &allocation_info,
        &buffer,
        &allocation,
        nullptr
    ) == VK_SUCCESS;
}

bool createLightingResources() {
    auto& context = graphics::internal::context;

    // 1. Создаём буфер для параметров освещения.
    if (!createBuffer(
            sizeof(LightingUniform),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            lighting_buffer,
            lighting_buffer_allocation)) {
        std::cerr << "Failed to create lighting uniform buffer\n";
        return false;
    }

    VmaAllocationInfo allocation_info{};
    vmaGetAllocationInfo(
        context.allocator,
        lighting_buffer_allocation,
        &allocation_info
    );

    lighting_buffer_mapped = allocation_info.pMappedData;

    if (lighting_buffer_mapped == nullptr) {
        std::cerr << "Lighting uniform buffer is not mapped\n";
        return false;
    }

    // 2. Описываем binding 0 для fragment shader.
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    if (vkCreateDescriptorSetLayout(
            context.device,
            &layout_info,
            nullptr,
            &lighting_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create lighting descriptor set layout\n";
        return false;
    }

// Создаём SBO для массива Point Light.
if (!createBuffer(
        sizeof(PointLightsBuffer),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        point_lights_buffer,
        point_lights_buffer_allocation)) {
    std::cerr << "Failed to create point lights SSBO\n";
    return false;
}

VmaAllocationInfo point_allocation_info{};
vmaGetAllocationInfo(
    context.allocator,
    point_lights_buffer_allocation,
    &point_allocation_info
);

point_lights_buffer_mapped = point_allocation_info.pMappedData;

if (point_lights_buffer_mapped == nullptr) {
    std::cerr << "Point lights SSBO is not mapped\n";
    return false;
}

// Создаём SSBO для массива Spot Light.
if (!createBuffer(
        sizeof(SpotLightsBuffer),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        spot_lights_buffer,
        spot_lights_buffer_allocation)) {
    std::cerr << "Failed to create spot lights SSBO\n";
    return false;
}

VmaAllocationInfo spot_allocation_info{};
vmaGetAllocationInfo(
    context.allocator,
    spot_lights_buffer_allocation,
    &spot_allocation_info
);

spot_lights_buffer_mapped = spot_allocation_info.pMappedData;

if (spot_lights_buffer_mapped == nullptr) {
    std::cerr << "Spot lights SSBO is not mapped\n";
    return false;
}

    // 3. Создаём отдельный descriptor pool для освещения.
    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
    };

    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    if (vkCreateDescriptorPool(
            context.device,
            &pool_info,
            nullptr,
            &lighting_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create lighting descriptor pool\n";
        return false;
    }

    // 4. Выделяем descriptor set.
    const VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = lighting_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &lighting_descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(
            context.device,
            &set_info,
            &lighting_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate lighting descriptor set\n";
        return false;
    }

    // 5. Привязываем uniform buffer к binding 0.
    const VkDescriptorBufferInfo buffer_info = {
        .buffer = lighting_buffer,
        .offset = 0,
        .range = sizeof(LightingUniform),
    };

    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = lighting_descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };


    // Обновляем существующий descriptor set для UBO.
    vkUpdateDescriptorSets(
        context.device,
        1,
        &write,
        0,
        nullptr
    );

    // Создаём layout для SSBO с массивом Point Light.
    const VkDescriptorSetLayoutBinding point_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo point_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &point_binding,
    };

    if (vkCreateDescriptorSetLayout(
            context.device,
            &point_layout_info,
            nullptr,
            &point_lights_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create point lights descriptor layout\n";
        return false;
    }

// Создаём layout для SSBO с массивом Spot Light.
const VkDescriptorSetLayoutBinding spot_binding = {
    .binding = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
};

const VkDescriptorSetLayoutCreateInfo spot_layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 1,
    .pBindings = &spot_binding,
};

if (vkCreateDescriptorSetLayout(
        context.device,
        &spot_layout_info,
        nullptr,
        &spot_lights_descriptor_set_layout) != VK_SUCCESS) {
    std::cerr << "Failed to create spot lights descriptor layout\n";
    return false;
}

    // Создаём pool для одного SSBO descriptor set.
    const VkDescriptorPoolSize point_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };

    const VkDescriptorPoolCreateInfo point_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &point_pool_size,
    };

    if (vkCreateDescriptorPool(
            context.device,
            &point_pool_info,
            nullptr,
            &point_lights_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create point lights descriptor pool\n";
        return false;
    }

    // Выделяем descriptor set.
    const VkDescriptorSetAllocateInfo point_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = point_lights_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &point_lights_descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(
            context.device,
            &point_set_info,
            &point_lights_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate point lights descriptor set\n";
        return false;
    }

    // Привязываем SSBO к binding 0 нового descriptor set.
    const VkDescriptorBufferInfo point_buffer_info = {
        .buffer = point_lights_buffer,
        .offset = 0,
        .range = sizeof(PointLightsBuffer),
    };

    const VkWriteDescriptorSet point_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = point_lights_descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &point_buffer_info,
    };

    vkUpdateDescriptorSets(
        context.device,
        1,
        &point_write,
        0,
        nullptr
    );


    // Создаём отдельный descriptor pool для Spot Light.
    const VkDescriptorPoolSize spot_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };

    const VkDescriptorPoolCreateInfo spot_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &spot_pool_size,
    };

    if (vkCreateDescriptorPool(
            context.device,
            &spot_pool_info,
            nullptr,
            &spot_lights_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create spot lights descriptor pool\n";
        return false;
    }

    // Выделяем descriptor set для Spot Light.
    const VkDescriptorSetAllocateInfo spot_set_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = spot_lights_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &spot_lights_descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(
            context.device,
            &spot_set_allocate_info,
            &spot_lights_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate spot lights descriptor set\n";
        return false;
    }

    // Привязываем Spot Light SSBO к binding = 0.
    const VkDescriptorBufferInfo spot_buffer_info = {
        .buffer = spot_lights_buffer,
        .offset = 0,
        .range = sizeof(SpotLightsBuffer),
    };

    const VkWriteDescriptorSet spot_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = spot_lights_descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &spot_buffer_info,
    };

    vkUpdateDescriptorSets(
        context.device,
        1,
        &spot_write,
        0,
        nullptr
    );


    return true;

}

} // namespace

void setWindow(GLFWwindow* window) {
    app_window = window;
}

bool initialize() {
    const VkDeviceSize vertex_buffer_size =
        sizeof(cube_vertices);

    const VkDeviceSize index_buffer_size =
        sizeof(cube_indices);

    if (!createBuffer(
            vertex_buffer_size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_buffer,
            vertex_buffer_allocation)) {
        return false;
    }

    if (!createBuffer(
            index_buffer_size,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            index_buffer,
            index_buffer_allocation)) {
        return false;
    }

VmaAllocationInfo vertex_info{};
vmaGetAllocationInfo(
    graphics::internal::context.allocator,
    vertex_buffer_allocation,
    &vertex_info
);
std::memcpy(vertex_info.pMappedData, cube_vertices.data(), sizeof(cube_vertices));

VmaAllocationInfo index_info{};
vmaGetAllocationInfo(
    graphics::internal::context.allocator,
    index_buffer_allocation,
    &index_info
);
std::memcpy(index_info.pMappedData, cube_indices.data(), sizeof(cube_indices));

    const auto vertex_code = readShaderFile("shaders/cube.vert.spv");
    const auto fragment_code = readShaderFile("shaders/cube.frag.spv");

    vertex_shader = createShaderModule(vertex_code);
    fragment_shader = createShaderModule(fragment_code);

    if (vertex_shader == VK_NULL_HANDLE ||
        fragment_shader == VK_NULL_HANDLE) {
        return false;
    }

if (!createLightingResources()) {
    std::cerr << "Failed to create lighting resources\n";
    return false;
}

if (!createGraphicsPipeline()) {
    std::cerr << "Failed to create graphics pipeline\n";
    return false;
}

std::cout << "Graphics pipeline created successfully\n";


    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;

    vkQueueWaitIdle(context.graphics_queue);

    // Освобождаем descriptor pools.
    if (lighting_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            context.device,
            lighting_descriptor_pool,
            nullptr
        );
        lighting_descriptor_pool = VK_NULL_HANDLE;
        lighting_descriptor_set = VK_NULL_HANDLE;
    }

    if (point_lights_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            context.device,
            point_lights_descriptor_pool,
            nullptr
        );
        point_lights_descriptor_pool = VK_NULL_HANDLE;
        point_lights_descriptor_set = VK_NULL_HANDLE;
    }

    if (spot_lights_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            context.device,
            spot_lights_descriptor_pool,
            nullptr
        );
        spot_lights_descriptor_pool = VK_NULL_HANDLE;
        spot_lights_descriptor_set = VK_NULL_HANDLE;
    }

    // Сначала уничтожаем pipeline и pipeline layout.
    if (graphics_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            context.device,
            graphics_pipeline,
            nullptr
        );
        graphics_pipeline = VK_NULL_HANDLE;
    }

    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            context.device,
            pipeline_layout,
            nullptr
        );
        pipeline_layout = VK_NULL_HANDLE;
    }

    // Теперь можно уничтожить descriptor set layouts.
    if (lighting_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context.device,
            lighting_descriptor_set_layout,
            nullptr
        );
        lighting_descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (point_lights_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context.device,
            point_lights_descriptor_set_layout,
            nullptr
        );
        point_lights_descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (spot_lights_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context.device,
            spot_lights_descriptor_set_layout,
            nullptr
        );
        spot_lights_descriptor_set_layout = VK_NULL_HANDLE;
    }

    // Освобождаем буферы освещения.
    if (lighting_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            lighting_buffer,
            lighting_buffer_allocation
        );
        lighting_buffer = VK_NULL_HANDLE;
        lighting_buffer_allocation = VK_NULL_HANDLE;
        lighting_buffer_mapped = nullptr;
    }

    if (point_lights_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            point_lights_buffer,
            point_lights_buffer_allocation
        );
        point_lights_buffer = VK_NULL_HANDLE;
        point_lights_buffer_allocation = VK_NULL_HANDLE;
        point_lights_buffer_mapped = nullptr;
    }

    if (spot_lights_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            spot_lights_buffer,
            spot_lights_buffer_allocation
        );
        spot_lights_buffer = VK_NULL_HANDLE;
        spot_lights_buffer_allocation = VK_NULL_HANDLE;
        spot_lights_buffer_mapped = nullptr;
    }

    // Освобождаем шейдеры.
    if (vertex_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            context.device,
            vertex_shader,
            nullptr
        );
        vertex_shader = VK_NULL_HANDLE;
    }

    if (fragment_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            context.device,
            fragment_shader,
            nullptr
        );
        fragment_shader = VK_NULL_HANDLE;
    }

    // Освобождаем геометрию.
    if (vertex_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            vertex_buffer,
            vertex_buffer_allocation
        );
        vertex_buffer = VK_NULL_HANDLE;
        vertex_buffer_allocation = VK_NULL_HANDLE;
    }

    if (index_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            index_buffer,
            index_buffer_allocation
        );
        index_buffer = VK_NULL_HANDLE;
        index_buffer_allocation = VK_NULL_HANDLE;
    }
}


void updateCameraMovement(double delta_time) {
    GLFWwindow* window = app_window;

    if (window == nullptr) {
        return;
    }
//Расчёт скорости движения за кадр
    const float speed =
        camera_move_speed * static_cast<float>(delta_time);
//Расчет направления вперед
    const float forward_x =
        std::cos(camera_yaw) * std::cos(camera_pitch);
    const float forward_y = std::sin(camera_pitch);
    const float forward_z =
        std::sin(camera_yaw) * std::cos(camera_pitch);

    // Направление вправо относительно камеры
    const float right_x = -std::sin(camera_yaw);
    const float right_z = std::cos(camera_yaw);

    // Вперёд и назад
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera_x += forward_x * speed;
        camera_y += forward_y * speed;
        camera_z += forward_z * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera_x -= forward_x * speed;
        camera_y -= forward_y * speed;
        camera_z -= forward_z * speed;
    }

    // Влево и вправо
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera_x -= right_x * speed;
        camera_z -= right_z * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera_x += right_x * speed;
        camera_z += right_z * speed;
    }

    // Вверх и вниз
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera_y += speed;
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        camera_y -= speed;
    }
}

void update(double time) {

const double delta_time =
    previous_frame_time > 0.0
        ? time - previous_frame_time
        : 0.0;

previous_frame_time = time;

updateCameraMovement(
    std::clamp(delta_time, 0.0, 0.05)
);

rotation_angle = static_cast<float>(time) * rotation_speed;

ImGui::Begin("Cube settings");

ImGui::Separator();
ImGui::Text("Camera");

ImGui::SliderFloat("Camera X", &camera_x, -5.0f, 5.0f);
ImGui::SliderFloat("Camera Y", &camera_y, -5.0f, 5.0f);
ImGui::SliderFloat("Camera Z", &camera_z, -5.0f, 5.0f);

float yaw_degrees = camera_yaw * 57.2957795f;
float pitch_degrees = camera_pitch * 57.2957795f;

if (ImGui::SliderFloat(
        "Camera yaw", &yaw_degrees, -180.0f, 180.0f)) {
    camera_yaw = yaw_degrees * 0.0174532925f;
}

if (ImGui::SliderFloat(
        "Camera pitch", &pitch_degrees, -89.0f, 89.0f)) {
    camera_pitch = pitch_degrees * 0.0174532925f;
}

float fov_degrees = camera_fov * 57.2957795f;

if (ImGui::SliderFloat(
        "Camera FOV", &fov_degrees, 30.0f, 100.0f)) {
    camera_fov = fov_degrees * 0.0174532925f;
}

ImGui::Checkbox("Perspective projection", &use_perspective);
ImGui::SliderFloat("Rotation speed", &rotation_speed, 0.0f, 3.0f);
ImGui::SliderFloat("Position X", &cube_position_x, -2.0f, 2.0f);
ImGui::SliderFloat("Position Y", &cube_position_y, -2.0f, 2.0f);
ImGui::SliderFloat("Position Z", &cube_position_z, -1.5f, 1.5f);
ImGui::SliderFloat("Scale", &cube_scale, 0.2f, 2.0f);

ImGui::Separator();
ImGui::Text("Directional light");

ImGui::SliderFloat3(
    "Light direction",
    lighting.light_direction,
    -1.0f,
    1.0f
);

ImGui::ColorEdit3(
    "Light color",
    lighting.light_color
);

ImGui::SliderFloat(
    "Light intensity",
    &lighting.light_intensity,
    0.0f,
    5.0f
);

ImGui::Separator();
ImGui::Text("Point Light");

ImGui::Checkbox("Point light enabled", &lighting.point_enabled);

ImGui::SliderFloat3(
    "Point position",
    lighting.point_position,
    -5.0f,
    5.0f
);

ImGui::ColorEdit3(
    "Point color",
    lighting.point_color
);

ImGui::SliderFloat(
    "Point intensity",
    &lighting.point_intensity,
    0.0f,
    50.0f
);

ImGui::Separator();
ImGui::Text("Spot Light");

ImGui::Checkbox(
    "Spot light enabled",
    &lighting.spot_enabled
);

ImGui::SliderFloat3(
    "Spot position",
    lighting.spot_position,
    -5.0f,
    5.0f
);

ImGui::SliderFloat3(
    "Spot direction",
    lighting.spot_direction,
    -1.0f,
    1.0f
);

ImGui::ColorEdit3(
    "Spot color",
    lighting.spot_color
);

ImGui::SliderFloat(
    "Spot intensity",
    &lighting.spot_intensity,
    0.0f,
    50.0f
);

ImGui::SliderFloat(
    "Spot inner angle",
    &lighting.spot_inner_angle,
    1.0f,
    85.0f
);

ImGui::SliderFloat(
    "Spot outer angle",
    &lighting.spot_outer_angle,
    1.0f,
    89.0f
);

if (lighting.spot_inner_angle > lighting.spot_outer_angle) {
    lighting.spot_inner_angle = lighting.spot_outer_angle;
}

ImGui::Separator();
ImGui::Text("Ambient light");

ImGui::ColorEdit3(
    "Ambient color",
    lighting.ambient_color
);

ImGui::SliderFloat(
    "Ambient intensity",
    &lighting.ambient_intensity,
    0.0f,
    1.0f
);

ImGui::Separator();
ImGui::Text("Material");

ImGui::ColorEdit3(
    "Albedo",
    lighting.albedo
);

ImGui::ColorEdit3(
    "Specular color",
    lighting.specular_color
);

ImGui::SliderFloat(
    "Shininess",
    &lighting.shininess,
    1.0f,
    256.0f
);

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(fd.command_buffer, &begin_info);

    const VkClearValue clear_values[] = {
        {
            .color = {
                { 0.1f, 0.1f, 0.1f, 1.0f }
            }
        },
        {
            .depthStencil = {
                1.0f, 0
            }
        }
    };

    const VkRenderPassBeginInfo render_pass_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = graphics::internal::context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {
            .offset = { 0, 0 },
            .extent = graphics::internal::context.swapchain_extent,
        },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };

    
    vkCmdBeginRenderPass(
        fd.command_buffer,
        &render_pass_begin,
        VK_SUBPASS_CONTENTS_INLINE
    );

    auto& context = graphics::internal::context;

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .offset = { 0, 0 },
        .extent = context.swapchain_extent,
    };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    vkCmdBindPipeline(
        fd.command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        graphics_pipeline
    );
//Привязка буфера вершин
    const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(
        fd.command_buffer, 0, 1, &vertex_buffer, &vertex_offset
    );

    vkCmdBindIndexBuffer(
        fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32
    );

    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

const Mat4 model = createModel(rotation_angle);

const Mat4 mvp = createMvp(
    rotation_angle,
    aspect,
    use_perspective
);

struct PushConstants {
    Mat4 model;
    Mat4 mvp;
};

const PushConstants push_constants = {
    .model = model,
    .mvp = mvp
};

//Структу параметров освещения
LightingUniform uniform{};

// Направление света
uniform.light_direction[0] = lighting.light_direction[0];
uniform.light_direction[1] = lighting.light_direction[1];
uniform.light_direction[2] = lighting.light_direction[2];
uniform.light_direction[3] = 0.0f;

// Цвет и интенсивность направленного света
uniform.light_color_intensity[0] = lighting.light_color[0];
uniform.light_color_intensity[1] = lighting.light_color[1];
uniform.light_color_intensity[2] = lighting.light_color[2];
uniform.light_color_intensity[3] = lighting.light_intensity;

// Фоновое освещение
uniform.ambient_color_intensity[0] = lighting.ambient_color[0];
uniform.ambient_color_intensity[1] = lighting.ambient_color[1];
uniform.ambient_color_intensity[2] = lighting.ambient_color[2];
uniform.ambient_color_intensity[3] = lighting.ambient_intensity;

// Основной цвет материала
uniform.albedo[0] = lighting.albedo[0];
uniform.albedo[1] = lighting.albedo[1];
uniform.albedo[2] = lighting.albedo[2];
uniform.albedo[3] = 1.0f;

// Цвет блика и shininess
uniform.specular_color_shininess[0] = lighting.specular_color[0];
uniform.specular_color_shininess[1] = lighting.specular_color[1];
uniform.specular_color_shininess[2] = lighting.specular_color[2];
uniform.specular_color_shininess[3] = lighting.shininess;

// Положение камеры в мировом пространстве
uniform.camera_position[0] = camera_x;
uniform.camera_position[1] = camera_y;
uniform.camera_position[2] = camera_z;
uniform.camera_position[3] = 1.0f;

uniform.point_position[0] = lighting.point_position[0];
uniform.point_position[1] = lighting.point_position[1];
uniform.point_position[2] = lighting.point_position[2];
uniform.point_position[3] = 1.0f;

uniform.point_color_intensity[0] = lighting.point_color[0];
uniform.point_color_intensity[1] = lighting.point_color[1];
uniform.point_color_intensity[2] = lighting.point_color[2];
uniform.point_color_intensity[3] =
    lighting.point_enabled ? lighting.point_intensity : 0.0f;

//Создается стурктура для массива точечных источников
PointLightsBuffer point_data{};
point_data.count = lighting.point_enabled ? 1u : 0u;

if (point_data.count > 0) {
    auto& point = point_data.lights[0];

    point.position[0] = lighting.point_position[0];
    point.position[1] = lighting.point_position[1];
    point.position[2] = lighting.point_position[2];
    point.position[3] = 1.0f;

    point.color_intensity[0] = lighting.point_color[0];
    point.color_intensity[1] = lighting.point_color[1];
    point.color_intensity[2] = lighting.point_color[2];
    point.color_intensity[3] = lighting.point_intensity;
}

std::memcpy(
    point_lights_buffer_mapped,
    &point_data,
    sizeof(point_data)
);



SpotLightsBuffer spot_lights_data{};
spot_lights_data.count = lighting.spot_enabled ? 1u : 0u;

if (spot_lights_data.count > 0) {
    auto& spot = spot_lights_data.lights[0];

    spot.position[0] = lighting.spot_position[0];
    spot.position[1] = lighting.spot_position[1];
    spot.position[2] = lighting.spot_position[2];
    spot.position[3] = 1.0f;

    spot.direction[0] = lighting.spot_direction[0];
    spot.direction[1] = lighting.spot_direction[1];
    spot.direction[2] = lighting.spot_direction[2];
    spot.direction[3] = 0.0f;

    spot.color_intensity[0] = lighting.spot_color[0];
    spot.color_intensity[1] = lighting.spot_color[1];
    spot.color_intensity[2] = lighting.spot_color[2];
    spot.color_intensity[3] = lighting.spot_intensity;

    spot.angles[0] = lighting.spot_inner_angle;
    spot.angles[1] = lighting.spot_outer_angle;
    spot.angles[2] = 0.0f;
    spot.angles[3] = 0.0f;
}

std::memcpy(
    spot_lights_buffer_mapped,
    &spot_lights_data,
    sizeof(SpotLightsBuffer)
);

vmaFlushAllocation(
    context.allocator,
    spot_lights_buffer_allocation,
    0,
    sizeof(SpotLightsBuffer)
);


vmaFlushAllocation(
    context.allocator,
    spot_lights_buffer_allocation,
    0,
    sizeof(SpotLightsBuffer)
);


vmaFlushAllocation(
    context.allocator,
    point_lights_buffer_allocation,
    0,
    sizeof(point_data)
);

// Передаём данные из C++ в uniform buffer.
std::memcpy(
    lighting_buffer_mapped,
    &uniform,
    sizeof(uniform)
);

vmaFlushAllocation(
    context.allocator,
    lighting_buffer_allocation,
    0,
    sizeof(uniform)
);

vkCmdPushConstants(
    fd.command_buffer,
    pipeline_layout,
    VK_SHADER_STAGE_VERTEX_BIT,
    0,
    sizeof(PushConstants),
    &push_constants
);

vkCmdBindDescriptorSets(
    fd.command_buffer,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipeline_layout,
    0,
    1,
    &lighting_descriptor_set,
    0,
    nullptr
);

vkCmdBindDescriptorSets(
    fd.command_buffer,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipeline_layout,
    1,
    1,
    &point_lights_descriptor_set,
    0,
    nullptr
);


vkCmdBindDescriptorSets(
    fd.command_buffer,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    pipeline_layout,
    2,  // set = 2: Spot Light SSBO
    1,
    &spot_lights_descriptor_set,
    0,
    nullptr
);


    vkCmdDrawIndexed(
        fd.command_buffer,
        static_cast<uint32_t>(cube_indices.size()),
        1, 0, 0, 0
    );

    vkCmdEndRenderPass(fd.command_buffer);

    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application