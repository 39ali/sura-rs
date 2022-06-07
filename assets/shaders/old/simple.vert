#version 460

#extension GL_GOOGLE_include_directive : require
#include "bindless.glsl"

layout(set = 1, binding = 0) uniform Camera {
  mat4 view;
  mat4 proj;
}
camera;

layout(set = 1, binding = 1, scalar) buffer Transform { mat4 v[]; }
transforms;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec2 o_uv;
layout(location = 2) out vec3 o_normal;
layout(location = 3) out vec4 o_tangent;
layout(location = 4) out flat uint32_t o_material_id;
layout(location = 5) out flat uint32_t o_mesh_index;

void main() {

  GpuMesh mesh = meshes.v[gl_InstanceIndex];
  mat4 model_transform = transforms.v[gl_InstanceIndex];

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

  gl_Position = camera.proj * camera.view * model_transform * vec4(pos, 1.0);
  o_color = color;
  o_uv = uv;
  o_normal = normal;
  o_tangent = tangent;
  o_material_id = material_id;
  o_mesh_index = gl_InstanceIndex;
}
