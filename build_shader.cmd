
 dxc.exe /Od  -Zi assets\shaders\simple_vs.hlsl -T vs_6_6 -spirv -fvk-use-scalar-layout -fspv-target-env=vulkan1.2 -WX -Fo assets\shaders\out\simple_vs.spv 
@REM  dxc.exe /Od  -Zi assets\shaders\simple_ps.hlsl -T ps_6_6 -spirv -fvk-use-scalar-layout -fspv-target-env=vulkan1.2 -WX -Fo assets\shaders\out\simple_ps.spv
 dxc.exe /Od  -Zi assets\shaders\pbr_ps.hlsl -T ps_6_6 -spirv -fvk-use-scalar-layout -fspv-target-env=vulkan1.2 -WX -Fo assets\shaders\out\pbr_ps.spv
