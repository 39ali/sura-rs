#version 460
#extension GL_GOOGLE_include_directive : require
#include "mesh.vert"

layout(set = 0, binding = 2) uniform sampler2D albedo;

layout(location = 0) in vec4 o_color;
layout(location = 1) in vec2 o_uv;
layout(location = 2) in vec3 o_normal;
layout(location = 3) in vec4 o_tangent;
layout(location = 4) in flat uint32_t o_material_id;

layout(location = 0) out vec4 uFragColor;
void main() { uFragColor = texture(albedo, o_uv) * o_color; }
