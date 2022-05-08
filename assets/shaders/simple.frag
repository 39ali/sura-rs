#version 460

layout(set = 0, binding = 2) uniform sampler2D albedo;

layout(location = 0) in vec4 i_color;
layout(location = 1) in vec2 i_uv;
layout(location = 0) out vec4 uFragColor;

void main() { uFragColor = texture(albedo, i_uv) + vec4(0.0); }
// void main() { uFragColor = i_color; }
