#ifndef BINDLESS_HLSL
#define BINDLESS_HLSL
#include "mesh.hlsl"
#include "samplers.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 0)]] Texture2D<float4> maps[];
[[vk::binding(2, 0)]] ByteAddressBuffer vertices;

#define LIGHTS_COUNT 4

struct FrameConstants {
  float4x4 view;
  float4x4 proj; 
  float3 cam_pos;
  float3 light_positions[LIGHTS_COUNT];
  float3 light_colors[LIGHTS_COUNT];
};

#endif 