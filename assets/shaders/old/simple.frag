#version 460
#extension GL_GOOGLE_include_directive : require
#include "bindless.glsl"

layout(location = 0) in vec4 o_color;
layout(location = 1) in vec2 o_uv;
layout(location = 2) in vec3 o_normal;
layout(location = 3) in vec4 o_tangent;
layout(location = 4) in flat uint32_t o_material_id;
layout(location = 5) in flat uint32_t o_mesh_index;

layout(location = 0) out vec4 uFragColor;
void main() {

  GpuMesh mesh = meshes.v[o_mesh_index];
  MeshMaterial material =
      MeshMaterialP(constants.vertices_ptr + mesh.materials_data_offset +
                    o_material_id * sizeof(MeshMaterialP))
          .v;

  const uint32_t albedo_index = material.maps_index[ALBEDO_MAP_INDEX];
  uFragColor = texture(maps[nonuniformEXT(albedo_index)], o_uv) * o_color;
}
