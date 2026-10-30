#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Transform {
    mat4 model;
    mat4 mvp;
} transform;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;

void main() {
    vec4 worldPosition =
        transform.model * vec4(inPosition, 1.0);

    fragPosition = worldPosition.xyz;

mat3 normalMatrix = transpose(
    inverse(mat3(transform.model))
);

fragNormal = normalize(normalMatrix * inNormal);

    gl_Position =
        transform.mvp * vec4(inPosition, 1.0);
}