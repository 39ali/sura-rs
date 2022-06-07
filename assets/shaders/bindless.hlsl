#ifndef BINDLESS_HLSL
#define BINDLESS_HLSL
#include "mesh.hlsl"
#include "samplers.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 0)]] Texture2D<float4> maps[];
[[vk::binding(2, 0)]] ByteAddressBuffer vertices;

#endif 