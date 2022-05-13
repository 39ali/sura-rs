#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_scalar_block_layout : require

#define sizeof(Type) (uint64_t(Type(uint64_t(0)) + 1))

layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer Vec4P {
  vec4 v;
};
layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer Vec3P {
  vec3 v;
};
layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer Vec2P {
  vec2 v;
};

layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer U32P {
  uint32_t v;
};

struct GpuMesh {
  uint32_t pos_offset;
  uint32_t uv_offset;
  uint32_t normal_offset;
  uint32_t colors_offset;
  uint32_t tangents_offset;
  uint32_t materials_data_offset;
  uint32_t material_ids_offset;
};

layout(set = 0, binding = 0, scalar) readonly buffer MeshBuffer { GpuMesh v[]; }
meshes;

layout(set = 0, binding = 1, scalar) uniform MVP {
  mat4 model;
  mat4 view;
  mat4 proj;
}
mvp;

layout(push_constant, scalar) uniform Constants { uint64_t vertices_ptr; }
constants;