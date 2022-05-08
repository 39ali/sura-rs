#version 460

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_scalar_block_layout : require

#define sizeof(Type) (uint64_t(Type(uint64_t(0)) + 1))

// layout(location = 0) in vec3 pos;
// layout(location = 1) in vec2 uv;

layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer Vec3P {
  vec3 v;
};
layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer Vec2P {
  vec2 v;
};

struct GpuMesh {
  uint32_t pos_offset;
  uint32_t uv_offset;
};

layout(set = 0, binding = 0, scalar) readonly buffer MeshBuffer { GpuMesh v[]; }
meshes;

layout(set = 0, binding = 1, scalar) uniform MVP {
  mat4 model;
  mat4 view;
  mat4 proj;
}
mvp;

// layout(set = 0, binding = 3, scalar) readonly buffer UVBuffer { vec2 v[]; }
// uvs;

layout(push_constant, scalar) uniform Constants { uint64_t vertices_ptr; }
constants;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec2 o_uv;
void main() {

  GpuMesh mesh = meshes.v[0];

  vec3 p = Vec3P(constants.vertices_ptr + mesh.pos_offset +
                 gl_VertexIndex * sizeof(Vec3P))
               .v;

  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(p, 1.0);

  o_color = vec4(1.0);
  o_uv = Vec2P(constants.vertices_ptr + mesh.uv_offset +
               gl_VertexIndex * sizeof(Vec2P))
             .v;

  // o_uv = vec2(uvs.v[gl_VertexIndex]);
}
