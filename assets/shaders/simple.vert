#version 460

#extension GL_GOOGLE_include_directive : require
#include "mesh.vert"

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec2 o_uv;
layout(location = 2) out vec3 o_normal;
layout(location = 3) out vec4 o_tangent;
layout(location = 4) out flat uint32_t o_material_id;

void main() {

  GpuMesh mesh = meshes.v[0];

  vec3 pos = Vec3P(constants.vertices_ptr + mesh.pos_offset +
                   gl_VertexIndex * sizeof(Vec3P))
                 .v;

  vec4 color = Vec4P(constants.vertices_ptr + mesh.colors_offset +
                     gl_VertexIndex * sizeof(Vec4P))
                   .v;

  vec2 uv = Vec2P(constants.vertices_ptr + mesh.uv_offset +
                  gl_VertexIndex * sizeof(Vec2P))
                .v;

  // TODO:pack this
  vec3 normal = Vec3P(constants.vertices_ptr + mesh.normal_offset +
                      gl_VertexIndex * sizeof(Vec3P))
                    .v;

  vec4 tangent = Vec4P(constants.vertices_ptr + mesh.tangents_offset +
                       gl_VertexIndex * sizeof(Vec4P))
                     .v;

  uint32_t material_id =
      U32P(constants.vertices_ptr + mesh.material_ids_offset +
           gl_VertexIndex * sizeof(U32P))
          .v;

  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(pos, 1.0);
  o_color = color;
  o_uv = uv;
  o_normal = normal;
  o_tangent = tangent;
  o_material_id = material_id;
}
