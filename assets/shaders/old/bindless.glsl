
#extension GL_GOOGLE_include_directive : require
#include "mesh.glsl"

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0, scalar) readonly buffer MeshBuffer { GpuMesh v[]; }
meshes;

layout(set = 0, binding = 1) uniform sampler2D maps[];

layout(push_constant, scalar) uniform Constants { uint64_t vertices_ptr; }
constants;
