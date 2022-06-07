#include "bindless.hlsl"
#include "samplers.hlsl"

struct PsIn{
   [[vk::location(0)]]  float4 color:TEXCOORD0 ; 
   [[vk::location(1)]]  float2 uv:TEXCOORD1; 
   [[vk::location(2)]]  float3 normal:TEXCOORD2;
   [[vk::location(3)]]  float4 tangent:TEXCOORD3; 
   [[vk::location(4)]]  nointerpolation uint material_id:TEXCOORD4; 
   [[vk::location(5)]]  nointerpolation uint mesh_id:TEXCOORD5; 
}; 

struct PsOut {
    float4 color : SV_Target0;
};

PsOut main(PsIn ps){
    PsOut ps_out; 

    Mesh mesh = meshes[ps.mesh_id];

    MeshMaterial mat = vertices.Load<MeshMaterial>(mesh.materials_data_offset + ps.material_id*sizeof(MeshMaterial));

    Texture2D albedo_tex = maps[NonUniformResourceIndex(mat.maps_index[ALBEDO_MAP_INDEX])];

    ps_out.color =  albedo_tex.Sample(sampler_n,ps.uv);

    return ps_out;
}