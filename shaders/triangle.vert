#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 pos;


layout(set=0,binding=0) uniform MVP {
mat4 model; 
mat4 view ; 
mat4 proj ; 
} mvp;


layout( push_constant ) uniform MVP_PUSH
{
mat4 model; 
mat4 view ; 
mat4 proj ; 
} mvp_push;


layout (location = 0) out vec4 o_color;
void main() {
    o_color = mvp_push.proj* mvp_push.view * mvp_push.model*vec4(pos,1.0);
    o_color = vec4(pos,1.0);
    gl_Position = mvp.proj* mvp.view * mvp.model*vec4 (pos,1.0);
}
