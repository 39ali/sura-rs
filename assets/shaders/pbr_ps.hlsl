#include "bindless.hlsl"
#include "samplers.hlsl"

struct PsIn {
  [[vk::location(0)]] float4 color : TEXCOORD0;
  [[vk::location(1)]] float2 uv : TEXCOORD1;
  [[vk::location(2)]] float3 normal : TEXCOORD2;
  [[vk::location(3)]] float3 tangent : TEXCOORD3;
  [[vk::location(4)]] float3 bitangent : TEXCOORD4;
  [[vk::location(5)]] nointerpolation uint material_id : TEXCOORD5;
  [[vk::location(6)]] nointerpolation uint mesh_id : TEXCOORD6;
  [[vk::location(7)]] float3 model_position : TEXCOORD7;
  [[vk::location(8)]] float4x4 model_to_world : TEXCOORD8;
};

struct PsOut {
  float4 color : SV_Target0;
};

[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> frame_constants;

#define M_PI 3.14159265358979323846264338327950288

#define GGX_CORRELATED 1

struct Brdf {
  static float3 f_fresnel_schlick(float3 f0, float cosTheta) {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cosTheta), 5);
  }

  static float ndf_GGX(float ndoth, float roughness) {
    float alpha = roughness * roughness;
    float alphaSq = alpha * alpha;

    float denom = (ndoth * ndoth) * (alphaSq - 1.0) + 1.0;
    return alphaSq / (M_PI * denom * denom);
  }

  static float g_schlickG1(float cosTheta, float k) {
    return cosTheta / (cosTheta * (1.0 - k) + k);
  }

  static float g_schlickGGX(float ndotl, float ndotv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return g_schlickG1(ndotl, k) * g_schlickG1(ndotv, k);
  }

  static float v_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
  }

  static float lambert() { return 1.0 / M_PI; }

  static float3 evaluate(float3 n, float3 v, float3 l, float roughness,
                         float3 f0, float3 albedo, float3 radiance) {
    float3 h = normalize(v + l);
    float ndoth = max(dot(n, h), 0.0);
    float hdotv = max(dot(h, v), 0.0);

    float ndotv = max(dot(n, v), 0.0); //+ 1e-5;
    float ndotl = max(dot(n, l), 0.0);

    float D = ndf_GGX(ndoth, roughness);
    float3 F = f_fresnel_schlick(f0, hdotv);

    // spec brdf
    float3 Fr;

#if GGX_CORRELATED
    float V = v_SmithGGXCorrelated(ndotv + 1e-5, ndotl, roughness);
    Fr = D * F * V;
#else
    float G = g_schlickGGX(ndotl, ndotv, roughness);
    Fr = D * F * G / max(4.0 * ndotv * ndotl, 1e-5);
#endif

    // diffuse brdf
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light);
    float3 Fd = (1.0 - F) * albedo * lambert();

    return (Fd + Fr) * radiance * ndotl;
  }
};

PsOut main(PsIn ps) {
  PsOut ps_out;

  Mesh mesh = meshes[ps.mesh_id];

  MeshMaterial mat = vertices.Load<MeshMaterial>(
      mesh.materials_data_offset + ps.material_id * sizeof(MeshMaterial));

  Texture2D albedo_tex =
      maps[NonUniformResourceIndex(mat.maps_index[ALBEDO_MAP_INDEX])];
  float3 albedo = albedo_tex.Sample(sampler_n, ps.uv).xyz *
                  mat.base_color_factor.xyz * ps.color.xyz;

  // TODO: implement emission
  // Texture2D emissive_tex=
  // maps[NonUniformResourceIndex(mat.maps_index[EMISSIVE_MAP_INDEX])];

  Texture2D spec_tex =
      maps[NonUniformResourceIndex(mat.maps_index[SPECULAR_MAP_INDEX])];
  const float4 metalness_roughness = spec_tex.Sample(sampler_n, ps.uv);
  const float roughness = mat.roughness_factor * metalness_roughness.y;
  const float metalness = mat.metalness_factor * metalness_roughness.z;

  // ao
  Texture2D occlusion_tex =
      maps[NonUniformResourceIndex(mat.maps_index[OCCLUSION_MAP_INDEX])];
  // TODO : add the strength factor
  const float occlusion = occlusion_tex.Sample(sampler_n, ps.uv).r;

  // normals
  Texture2D normal_tex =
      maps[NonUniformResourceIndex(mat.maps_index[NORMAL_MAP_INDEX])];
  const float3 ts_normal =
      normalize(normal_tex.Sample(sampler_n, ps.uv).xyz * 2.0 -
                1.0); //[0.0,1.0]->[-1.0,1.0]
  const float3x3 TBN = float3x3(ps.tangent, ps.bitangent, ps.normal);
  float3 N = (mul(ts_normal, TBN));
  N = normalize(mul((float3x3)ps.model_to_world, N));
  //

  const float3 world_pos =
      mul(ps.model_to_world, float4(ps.model_position, 1.0)).xyz;
  const float3 V = normalize(frame_constants.cam_pos - world_pos);

  const float3 F0 = lerp(0.04, albedo, metalness);
  // only non-metals have full diffuse lighting
  albedo = lerp(albedo, 0.0, metalness);

  float3 Lo = 0.0;
  for (int i = 0; i < LIGHTS_COUNT; i++) {

    float3 L = normalize(frame_constants.light_positions[i] - world_pos);

    // point light radiance
    float distance = length(frame_constants.light_positions[i] - world_pos);
    float attenuation = 1.0 / (distance * distance);
    float3 radiance = frame_constants.light_colors[i] * attenuation;

    Lo += Brdf::evaluate(N, V, L, roughness, F0, albedo, radiance);
  }

  float3 ambient = 0.3f * albedo * occlusion;

  float3 color = ambient + Lo;

  // // HDR tonemapping  (Reinhard)
  // color = color / (color + (1.0));
  // gamma correctiton
  color = pow(color, (1.0 / 2.2));

  ps_out.color = float4(color, 1.0);

  return ps_out;
}