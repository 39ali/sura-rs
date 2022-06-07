
 dxc.exe /Od  -Zi assets\shaders\simple_vs.hlsl -T vs_6_6 -spirv -fvk-use-scalar-layout -fspv-target-env=vulkan1.2 -WX -Fo assets\shaders\simple_vs.spv
 dxc.exe /Od  -Zi assets\shaders\simple_ps.hlsl -T ps_6_6 -spirv -fvk-use-scalar-layout -fspv-target-env=vulkan1.2 -WX -Fo assets\shaders\simple_ps.spv
