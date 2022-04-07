#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 1) uniform sampler2D tex;

layout(location = 0) in vec4 i_color;
layout(location = 1) in vec2 i_uv;
layout(location = 0) out vec4 uFragColor;

void main() { uFragColor = texture(tex, i_uv); }
