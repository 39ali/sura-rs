#pragma enable_d3d11_debug_symbols
#include "bindless.hlsl"

struct FrameConstants {
  float4x4 view;
  float4x4 proj; 
};


[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> frame_constants;
[[vk::binding(1, 1)]] StructuredBuffer< float4x4> transforms;




struct VsOut{
 
    float4 positon : SV_Position;
   [[vk::location(0)]]  float4 color:TEXCOORD0 ; 
   [[vk::location(1)]]  float2 uv:TEXCOORD1; 
   [[vk::location(2)]]  float3 normal:TEXCOORD2 ;
   [[vk::location(3)]]  float4 tangent:TEXCOORD3; 
   [[vk::location(4)]]  nointerpolation  uint material_id:TEXCOORD4; 
   [[vk::location(5)]]  nointerpolation  uint mesh_id:TEXCOORD5; 
};

VsOut main(uint vertex_index:SV_VertexID , uint instance_index:SV_InstanceID){
    VsOut vs_out; 

    const Mesh mesh=meshes[instance_index];

    float4x4 model_transform = transforms[instance_index];
    
    float3 pos = asfloat(vertices.Load3(mesh.pos_offset + vertex_index*sizeof(float3)));

    float4 color =asfloat(vertices.Load4(mesh.colors_offset + vertex_index*sizeof(float4)));
    
    float2 uv = asfloat(vertices.Load2(mesh.uv_offset + vertex_index*sizeof(float2))); 
    
    float3 normal =asfloat(vertices.Load3(mesh.uv_offset + vertex_index*sizeof(float3)));
    
    float4 tangent =asfloat(vertices.Load4(mesh.tangents_offset + vertex_index*sizeof(float4)));

    uint material_id = vertices.Load(mesh.material_ids_offset + vertex_index*sizeof(uint));

    vs_out.color = color; 
    vs_out.uv = uv; 
    vs_out.normal = normal; 
    vs_out.tangent = tangent; 
    vs_out.material_id = material_id; 
    vs_out.mesh_id = instance_index; 

    float4x4 world_to_view = mul(mul(frame_constants.proj,frame_constants.view),model_transform);
    vs_out.positon = mul(world_to_view,float4(pos,1.0) ); 

    return vs_out;
}