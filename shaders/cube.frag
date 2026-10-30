
#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform Lighting {
    vec4 lightDirection;
    vec4 lightColorIntensity;
    vec4 ambientColorIntensity;
    vec4 albedo;
    vec4 specularColorShininess;
    vec4 cameraPosition;

    vec4 pointPosition;
    vec4 pointColorIntensity;
} lighting;


struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

layout(set = 1, binding = 0, std430) readonly buffer PointLights {
    uint count;
    uint padding0;
    uint padding1;
    uint padding2;
    PointLight lights[];
} pointLights;


struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 angles;
};

layout(set = 2, binding = 0, std430) readonly buffer SpotLights {
    uint count;
    uint padding0;
    uint padding1;
    uint padding2;
    SpotLight lights[];
} spotLights;



void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(
        lighting.cameraPosition.xyz - fragPosition
    );

    float shininess = max(
        lighting.specularColorShininess.a, 1.0
    );

    // Ambient
    vec3 ambient =
        lighting.ambientColorIntensity.rgb *
        lighting.ambientColorIntensity.a *
        lighting.albedo.rgb;

    // Directional light
    vec3 dirLight = normalize(lighting.lightDirection.xyz);
    float dirDiff = max(dot(normal, dirLight), 0.0);

    vec3 dirHalfway = dirLight + viewDir;
    float dirSpec = 0.0;

    if (dirDiff > 0.0 && dot(dirHalfway, dirHalfway) > 0.000001) {
        dirSpec = pow(
            max(dot(normal, normalize(dirHalfway)), 0.0),
            shininess
        );
    }

    vec3 directional =
        lighting.lightColorIntensity.rgb *
        lighting.lightColorIntensity.a *
        (
            dirDiff * lighting.albedo.rgb +
            dirSpec * lighting.specularColorShininess.rgb
        );


    // Point lights from SSBO
    vec3 pointLighting = vec3(0.0);

    for (uint i = 0; i < min(pointLights.count, 8u); ++i) {
        vec3 toPoint =
            pointLights.lights[i].position.xyz - fragPosition;

        float pointDistance2 = max(dot(toPoint, toPoint), 0.01);
        vec3 pointDir = toPoint * inversesqrt(pointDistance2);

        float pointDiff = max(dot(normal, pointDir), 0.0);
        vec3 pointHalfway = pointDir + viewDir;
        float pointSpec = 0.0;

        if (pointDiff > 0.0 &&
            dot(pointHalfway, pointHalfway) > 0.000001) {
            pointSpec = pow(
                max(dot(normal, normalize(pointHalfway)), 0.0),
                shininess
            );
        }

        pointLighting +=
            pointLights.lights[i].colorIntensity.rgb *
            pointLights.lights[i].colorIntensity.a /
            pointDistance2 *
            (
                pointDiff * lighting.albedo.rgb +
                pointSpec * lighting.specularColorShininess.rgb
            );
    }



    // Spot lights from SSBO
    vec3 spotLighting = vec3(0.0);

    for (uint i = 0; i < min(spotLights.count, 8u); ++i) {
        vec3 fromSpot =
            fragPosition - spotLights.lights[i].position.xyz;

        float spotDistance2 = max(
            dot(fromSpot, fromSpot),
            0.01
        );

        vec3 spotRay = fromSpot * inversesqrt(spotDistance2);

        vec3 spotDir = normalize(
            spotLights.lights[i].direction.xyz
        );

        // Углы в градусах преобразуются в cosine-пороги.
        float innerCos = cos(
            radians(spotLights.lights[i].angles.x)
        );

        float outerCos = cos(
            radians(spotLights.lights[i].angles.y)
        );

        float theta = dot(spotRay, spotDir);
        float cone = smoothstep(outerCos, innerCos, theta);

        vec3 spotToSurface = -spotRay;
        float spotDiff = max(
            dot(normal, spotToSurface),
            0.0
        );

        vec3 spotHalfway = spotToSurface + viewDir;
        float spotSpec = 0.0;

        if (spotDiff > 0.0 &&
            dot(spotHalfway, spotHalfway) > 0.000001) {
            spotSpec = pow(
                max(dot(normal, normalize(spotHalfway)), 0.0),
                shininess
            );
        }

        spotLighting +=
            cone *
            spotLights.lights[i].colorIntensity.rgb *
            spotLights.lights[i].colorIntensity.a /
            spotDistance2 *
            (
                spotDiff * lighting.albedo.rgb +
                spotSpec * lighting.specularColorShininess.rgb
            );
    }


    vec3 result =
        ambient +
        directional +
        pointLighting +
        spotLighting;

    outColor = vec4(result, 1.0);
}
