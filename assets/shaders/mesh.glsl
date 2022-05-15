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

// mesh material
const uint32_t MAPS_COUNT = 5;
const uint32_t NORMAL_MAP_INDEX = 0;
const uint32_t SPECULAR_MAP_INDEX = 1;
const uint32_t ALBEDO_MAP_INDEX = 2;
const uint32_t EMISSIVE_MAP_INDEX = 3;
const uint32_t OCCLUSION_MAP_INDEX = 4;

struct MeshMaterial {
  vec4 base_color_factor;
  uint32_t maps_index[MAPS_COUNT];
  float roughness_factor;
  float metalness_factor;
  vec3 emissive_factors;
};
layout(buffer_reference, scalar,
       buffer_reference_align = 4) readonly buffer MeshMaterialP {
  MeshMaterial v;
};

// end