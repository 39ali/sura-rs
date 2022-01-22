extern crate ash;
extern crate winit;

extern crate base64;
extern crate gltf;

extern crate custom_error;
extern crate glam;

extern crate image;
extern crate indexmap;

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

use core::slice::{self};
use std::{ffi::CString, mem, rc::Rc};

use ash::vk::{self};
use device::{GFXDevice, GPUBuffer};
use std::time::Instant;
mod device;

mod gpu_structs;
use gpu_structs::*;
mod renderable;
use renderable::*;

/// Make sure to create the fence in a signaled state on the first use.
#[allow(clippy::too_many_arguments)]
pub fn record_submit_commandbuffer<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(
                submit_queue,
                &[submit_info.build()],
                command_buffer_reuse_fence,
            )
            .expect("queue submit failed.");
    }
}

#[derive(Default, Debug)]
struct MVP {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

// fn create_graphics_pipeline<'a>(
//     device: &'a Rc<GFXDevice>,
//     renderpass: vk::RenderPass,
//     renderable: &Renderable,
// ) -> (
//     device::GPUBuffer<'a>,
//     device::GPUBuffer<'a>,
//     device::GPUBuffer<'a>,
// ) {
//     unsafe {
//         assert!(
//             renderable.meshes.len() == 1,
//             "multiple meshes aren't supported:({})meshes",
//             renderable.meshes.len()
//         );
//         let mesh = &renderable.meshes[0];

//         let mut desc = device::GPUBufferDesc {
//             size: 0u64,
//             memory_location: device::MemLoc::CpuToGpu,
//             usage: device::GPUBufferUsage::INDEX_BUFFER,
//             ..Default::default()
//         };

//         let index_buffer = match mesh.index_buffer {
//             Indices::None => device.create_buffer::<u32>(&desc, None),
//             Indices::U32(ref b) => {
//                 desc.index_buffer_type = Some(GPUIndexedBufferType::U32);
//                 desc.size = std::mem::size_of_val(b.as_slice()) as u64;
//                 device.create_buffer(&desc, Some(b))
//             }
//             Indices::U16(ref b) => {
//                 desc.index_buffer_type = Some(GPUIndexedBufferType::U16);
//                 desc.size = std::mem::size_of_val(b.as_slice()) as u64;
//                 device.create_buffer(&desc, Some(b))
//             }
//             Indices::U8(ref b) => {
//                 desc.index_buffer_type = Some(GPUIndexedBufferType::U8);
//                 desc.size = std::mem::size_of_val(b.as_slice()) as u64;
//                 device.create_buffer(&desc, Some(b))
//             }
//         };

//         let mesh_buffer = mesh.get_buffer();

//         //NOTE(ALI): we're using cpuTogpu because we don't support GpuOnly yet
//         let desc = device::GPUBufferDesc {
//             size: mesh_buffer.len() as u64,
//             memory_location: device::MemLoc::CpuToGpu,
//             usage: device::GPUBufferUsage::VERTEX_BUFFER,
//             ..Default::default()
//         };

//         let vertex_buffer = device.create_buffer(&desc, Some(&mesh_buffer));

//         // create uniform
//         let desc = device::GPUBufferDesc {
//             size: std::mem::size_of::<MVP>() as u64,
//             memory_location: device::MemLoc::CpuToGpu,
//             usage: device::GPUBufferUsage::UNIFORM_BUFFER,
//             ..Default::default()
//         };

//         let vertex_shader = device.create_shader(&include_bytes!("../shaders/vert.spv")[..]);

//         let frag_shader = device.create_shader(&include_bytes!("../shaders/frag.spv")[..]);

//         let pso_desc = PipelineStateDesc {
//             fragment: Some(frag_shader),
//             vertex: Some(vertex_shader),
//             vertex_input_binding_descriptions: vec![vk::VertexInputBindingDescription {
//                 binding: 0,
//                 stride: mesh.stride() as u32,
//                 input_rate: vk::VertexInputRate::VERTEX,
//             }],
//             vertex_input_attribute_descriptions: vec![
//                 vk::VertexInputAttributeDescription {
//                     location: 0,
//                     binding: 0,
//                     format: vk::Format::R32G32B32_SFLOAT,
//                     offset: 0u32, //offset_of!(Vertex, pos) as u32,
//                 },
//                 // vk::VertexInputAttributeDescription {
//                 //     location: 1,
//                 //     binding: 0,
//                 //     format: vk::Format::R32G32B32A32_SFLOAT,
//                 //     offset: 0u32, //(mem::size_of::<f32>() * 3) as u32, //offset_of!(Vertex, color) as u32,
//                 // },
//             ],
//         };
//         let pso = device.create_pipeline_state(&pso_desc);

//         let mvp_buffer = device.create_buffer::<u8>(&desc, None);

//         // let mvp_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
//         //     .binding(0)
//         //     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
//         //     .descriptor_count(1)
//         //     .stage_flags(vk::ShaderStageFlags::VERTEX)
//         //     .build();

//         // let bindings = &[mvp_ubo_binding];

//         // let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

//         // let desc_set_layout = device
//         //     .device
//         //     .create_descriptor_set_layout(&create_info, None)
//         //     .expect("failed to create descriptor set layout");

//         // let set_layouts = &[desc_set_layout];

//         // let push_constant_ranges = &[vk::PushConstantRange::builder()
//         //     .stage_flags(vk::ShaderStageFlags::VERTEX)
//         //     .offset(0)
//         //     .size(std::mem::size_of::<MVP>() as u32)
//         //     .build()];

//         // let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
//         //     .push_constant_ranges(push_constant_ranges)
//         //     .set_layouts(set_layouts);

//         // let pipeline_layout = device
//         //     .device
//         //     .create_pipeline_layout(&layout_create_info, None)
//         //     .unwrap();

//         // let shader_entry_name = CString::new("main").unwrap();
//         // let shader_stage_create_infos = [
//         //     vk::PipelineShaderStageCreateInfo {
//         //         module: pso_desc.vertex.unwrap().module,
//         //         p_name: shader_entry_name.as_ptr(),
//         //         stage: vk::ShaderStageFlags::VERTEX,
//         //         ..Default::default()
//         //     },
//         //     vk::PipelineShaderStageCreateInfo {
//         //         s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
//         //         module: pso_desc.fragment.unwrap().module,
//         //         p_name: shader_entry_name.as_ptr(),
//         //         stage: vk::ShaderStageFlags::FRAGMENT,
//         //         ..Default::default()
//         //     },
//         // ];

//         // let desc_set = build_descriptors(device, &desc_set_layout, &mvp_buffer);

//         // let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
//         //     binding: 0,
//         //     stride: mesh.stride() as u32,
//         //     input_rate: vk::VertexInputRate::VERTEX,
//         // }];

//         // let vertex_input_attribute_descriptions = [
//         //     vk::VertexInputAttributeDescription {
//         //         location: 0,
//         //         binding: 0,
//         //         format: vk::Format::R32G32B32_SFLOAT,
//         //         offset: 0u32, //offset_of!(Vertex, pos) as u32,
//         //     },
//         //     // vk::VertexInputAttributeDescription {
//         //     //     location: 1,
//         //     //     binding: 0,
//         //     //     format: vk::Format::R32G32B32A32_SFLOAT,
//         //     //     offset: 0u32, //(mem::size_of::<f32>() * 3) as u32, //offset_of!(Vertex, color) as u32,
//         //     // },
//         // ];

//         // let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
//         //     vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
//         //     p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
//         //     vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
//         //     p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
//         //     ..Default::default()
//         // };

//         // let graphics_pipelines = device
//         //     .device
//         //     .create_graphics_pipelines(
//         //         vk::PipelineCache::null(),
//         //         &[graphic_pipeline_info.build()],
//         //         None,
//         //     )
//         //     .expect("Unable to create graphics pipeline");

//         (pso, index_buffer, vertex_buffer, mvp_buffer)
//     }
// }

// fn build_descriptors(
//     device: &Rc<GFXDevice>,
//     desc_set_layout: &vk::DescriptorSetLayout,
//     uniform_buffer: &GPUBuffer,
// ) -> vk::DescriptorSet {
//     unsafe {
//         //create uniform pool
//         let uniform_pool_size = vk::DescriptorPoolSize::builder()
//             .descriptor_count(1)
//             .ty(vk::DescriptorType::UNIFORM_BUFFER)
//             .build();

//         let pool_sizes = &[uniform_pool_size];

//         let ci = vk::DescriptorPoolCreateInfo::builder()
//             .pool_sizes(pool_sizes)
//             .max_sets(3);

//         let desc_pool = device
//             .device
//             .create_descriptor_pool(&ci, None)
//             .expect("couldn't create descrriptor pool");

//         //allocate desc sets

//         let desc_set_layouts = &[*desc_set_layout];
//         let ci = vk::DescriptorSetAllocateInfo::builder()
//             .descriptor_pool(desc_pool)
//             .set_layouts(desc_set_layouts)
//             .build();

//         let desc_sets = device
//             .device
//             .allocate_descriptor_sets(&ci)
//             .expect("failed to allocate descriptor sets");

//         // update/define desc
//         let desc_buffer = vk::DescriptorBufferInfo::builder()
//             .range(vk::WHOLE_SIZE)
//             .buffer(uniform_buffer.buffer)
//             .offset(0)
//             .build();
//         //update desc set
//         let wds = vk::WriteDescriptorSet::builder()
//             .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
//             .dst_set(desc_sets[0])
//             .dst_binding(0)
//             .dst_array_element(0)
//             .buffer_info(&[desc_buffer])
//             .build();
//         let desc_writes = &[wds];
//         device.device.update_descriptor_sets(desc_writes, &[]);

//         desc_sets[0]
//     }
// }

// fn update_uniform_buffer(width: i32, height: i32, start: &Instant) -> MVP {
//     let elapsed = { start.elapsed() };

//     let view = glam::Mat4::look_at_lh(
//         glam::vec3(0.0f32, 2.0, -7.0),
//         glam::vec3(0.0f32, 0.0, 0.0),
//         glam::vec3(0.0f32, 1.0f32, 0.0f32),
//     );
//     let proj = glam::Mat4::perspective_lh(
//         f32::to_radians(45.0f32),
//         width as f32 / height as f32,
//         0.01f32,
//         100.0f32,
//     );

//     //https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
//     let proj = proj.mul_mat4(&glam::mat4(
//         glam::vec4(1.0f32, 0.0, 0.0, 0.0),
//         glam::vec4(0.0f32, -1.0, 0.0, 0.0),
//         glam::vec4(0.0f32, 0.0, 1.0f32, 0.0),
//         glam::vec4(0.0f32, 0.0, 0.0f32, 1.0),
//     ));

//     let model = glam::Quat::from_axis_angle(
//         glam::vec3(0.0f32, 1.0, 0.0),
//         f32::to_radians(elapsed.as_millis() as f32) * 0.05f32,
//     );
//     let model = glam::Mat4::from_quat(model);

//     MVP { proj, view, model }
// }

fn main() {
    let renderable = load_gltf("./models/gltf_logo/scene.gltf");

    let start: Instant = Instant::now();

    unsafe {
        let window_width = 1024;
        let window_height = 768;
        let mut events_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("Sura")
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(window_width),
                f64::from(window_height),
            ))
            .build(&events_loop)
            .unwrap();

        let device = Rc::new(GFXDevice::new(&window));

        assert!(
            renderable.meshes.len() == 1,
            "multiple meshes aren't supported:({})meshes",
            renderable.meshes.len()
        );
        let mesh = &renderable.meshes[0];

        let mut desc = device::GPUBufferDesc {
            size: 0u64,
            memory_location: device::MemLoc::CpuToGpu,
            usage: device::GPUBufferUsage::INDEX_BUFFER,
            ..Default::default()
        };

        let index_buffer = match mesh.index_buffer {
            Indices::None => device.create_buffer::<u32>(&desc, None),
            Indices::U32(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U32);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                device.create_buffer(&desc, Some(b))
            }
            Indices::U16(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U16);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                device.create_buffer(&desc, Some(b))
            }
            Indices::U8(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U8);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                device.create_buffer(&desc, Some(b))
            }
        };

        let mesh_buffer = mesh.get_buffer();

        //NOTE(ALI): we're using cpuTogpu because we don't support GpuOnly yet
        let desc = device::GPUBufferDesc {
            size: mesh_buffer.len() as u64,
            memory_location: device::MemLoc::CpuToGpu,
            usage: device::GPUBufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        let vertex_buffer = device.create_buffer(&desc, Some(&mesh_buffer));

        // create uniform
        let desc = device::GPUBufferDesc {
            size: std::mem::size_of::<MVP>() as u64,
            memory_location: device::MemLoc::CpuToGpu,
            usage: device::GPUBufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        };

        let vertex_shader = device.create_shader(&include_bytes!("../shaders/vert.spv")[..]);

        let frag_shader = device.create_shader(&include_bytes!("../shaders/frag.spv")[..]);

        let pso_desc = PipelineStateDesc {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            vertex_input_binding_descriptions: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: mesh.stride() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vertex_input_attribute_descriptions: vec![
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0u32, //offset_of!(Vertex, pos) as u32,
                },
                // vk::VertexInputAttributeDescription {
                //     location: 1,
                //     binding: 0,
                //     format: vk::Format::R32G32B32A32_SFLOAT,
                //     offset: 0u32, //(mem::size_of::<f32>() * 3) as u32, //offset_of!(Vertex, color) as u32,
                // },
            ],
        };
        let pso = device.create_pipeline_state(&pso_desc);

        let mvp_buffer = device.create_buffer::<u8>(&desc, None);

        // let ci = vk::ImageCreateInfo::builder()
        //     .array_layers(1)
        //     .mip_levels(1)
        //     .extent(vk::Extent3D {
        //         height: window_height,
        //         width: window_width,
        //         depth: 1,
        //     })
        //     .format(vk::Format::D32_SFLOAT_S8_UINT)
        //     .image_type(vk::ImageType::TYPE_2D)
        //     .samples(vk::SampleCountFlags::TYPE_1)
        //     .tiling(vk::ImageTiling::OPTIMAL)
        //     .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        //     .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // let vi = vk::ImageViewCreateInfo::builder()
        //     .subresource_range(
        //         vk::ImageSubresourceRange::builder()
        //             .aspect_mask(vk::ImageAspectFlags::DEPTH)
        //             .level_count(1)
        //             .layer_count(1)
        //             .build(),
        //     )
        //     .format(ci.format)
        //     .view_type(vk::ImageViewType::TYPE_2D);

        // let img = device.create_image();

        // let depth_view = img.create_view(vk::ImageAspectFlags::DEPTH, 1, 1);

        let attachments = [
            vk::AttachmentDescription {
                format: device.swapchain.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            // vk::AttachmentDescription {
            //     samples: vk::SampleCountFlags::TYPE_1,
            //     load_op: vk::AttachmentLoadOp::CLEAR,
            //     store_op: vk::AttachmentStoreOp::DONT_CARE,
            //     final_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            //     ..Default::default()
            // },
        ];

        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        // let depth_ref = vk::AttachmentReference {
        //     attachment: 1,
        //     layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        // };

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_ref)
            // .depth_stencil_attachment(&depth_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let renderpass_ci = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let renderpass = device
            .device
            .create_render_pass(&renderpass_ci, None)
            .expect("failed to create a renderpass");

        let framebuffers: Vec<vk::Framebuffer> = device
            .swapchain
            .present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderpass)
                    .attachments(&framebuffer_attachments)
                    .width(device.swapchain.width)
                    .height(device.swapchain.height)
                    .layers(1);

                device
                    .device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            })
            .collect();

        //     let g = device.clone();

        //     let pipelines = { create_graphics_pipeline(&g, renderpass, &renderable) };
        //     let graphic_pipeline = pipelines.0[0];
        //     let graphic_pipeline_layout = pipelines.1;
        //     let index_buffer = pipelines.2;
        //     let vertex_buffer = pipelines.3.buffer;
        //     let mut uniform_buffer = pipelines.4;
        //     let desc_set = pipelines.5;
        //     let viewports = [vk::Viewport {
        //         x: 0.0,
        //         y: 0.0,
        //         width: device.swapchain.width as f32,
        //         height: device.swapchain.height as f32,
        //         min_depth: 0.0,
        //         max_depth: 1.0,
        //     }];

        //     let scissors = [vk::Rect2D {
        //         offset: vk::Offset2D { x: 0, y: 0 },
        //         extent: vk::Extent2D {
        //             width: device.swapchain.width,
        //             height: device.swapchain.height,
        //         },
        //     }];

        let info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let fence = device
            .device
            .create_fence(&info, None)
            .expect("failed to create fence");

        let gfx = device.clone();

        events_loop.run_return(|event, _c, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput { input, .. } => {
                        if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                            *control_flow = ControlFlow::Exit;
                        }
                    }

                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                Event::MainEventsCleared => {
                    // Application update code.

                    // Queue a RedrawRequested event.
                    //
                    // You only need to call this if you've determined that you need to redraw, in
                    // applications which do not always need to. Applications that redraw continuously
                    // can just render here instead.
                    // window.request_redraw();

                    let (present_index, _) = device
                        .swapchain_loader
                        .acquire_next_image(
                            device.swapchain.swapchain,
                            u64::MAX,
                            device.present_complete_semaphore,
                            vk::Fence::null(),
                        )
                        .unwrap();

                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [1.0, 0.0, 1.0, 0.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(renderpass)
                        .framebuffer(framebuffers[present_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: device.swapchain.width,
                                height: device.swapchain.height,
                            },
                        })
                        .clear_values(&clear_values);

                    record_submit_commandbuffer(
                        &device.device,
                        device.command_buffers[present_index as usize],
                        fence,
                        device.graphics_queue,
                        &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                        &[device.present_complete_semaphore],
                        &[device.rendering_complete_semaphore],
                        |device, draw_command_buffer| {
                            device.cmd_begin_render_pass(
                                draw_command_buffer,
                                &render_pass_begin_info,
                                vk::SubpassContents::INLINE,
                            );
                            // device.cmd_bind_pipeline(
                            //     draw_command_buffer,
                            //     vk::PipelineBindPoint::GRAPHICS,
                            //     graphic_pipeline,
                            // );
                            gfx.bind_pipeline(&pso, &draw_command_buffer);
                            device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                            device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                            device.cmd_bind_vertex_buffers(
                                draw_command_buffer,
                                0,
                                &[vertex_buffer],
                                &[0],
                            );

                            // push_constants
                            let mvp = update_uniform_buffer(window_width, window_height, &start);

                            uniform_buffer.allocation.mapped_slice_mut().unwrap()
                                [0..mem::size_of::<MVP>()]
                                .copy_from_slice(slice::from_raw_parts(
                                    (&mvp as *const MVP) as *const u8,
                                    mem::size_of::<MVP>(),
                                ));

                            let constants = slice::from_raw_parts(
                                (&mvp as *const MVP) as *const u8,
                                mem::size_of::<MVP>(),
                            );
                            device.cmd_push_constants(
                                draw_command_buffer,
                                graphic_pipeline_layout,
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                constants,
                            );

                            device.cmd_bind_descriptor_sets(
                                draw_command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                graphic_pipeline_layout,
                                0,
                                &[desc_set],
                                &[],
                            );

                            let index_type = match index_buffer.desc.index_buffer_type {
                                Some(ref t) => match t {
                                    GPUIndexedBufferType::U32 => vk::IndexType::UINT32,
                                    GPUIndexedBufferType::U16 => vk::IndexType::UINT16,
                                    GPUIndexedBufferType::U8 => vk::IndexType::UINT8_EXT,
                                },
                                _ => {
                                    panic!("index buffer type is not defined");
                                }
                            };

                            device.cmd_bind_index_buffer(
                                draw_command_buffer,
                                index_buffer.buffer,
                                0,
                                index_type,
                            );

                            let index_count = match renderable.meshes[0].index_buffer {
                                Indices::None => 0,
                                Indices::U32(ref i) => i.len(),
                                Indices::U16(ref i) => i.len(),
                                Indices::U8(ref i) => i.len(),
                            } as u32;

                            device.cmd_draw_indexed(draw_command_buffer, index_count, 1, 0, 0, 1);
                            // Or draw without the index buffer
                            // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
                            device.cmd_end_render_pass(draw_command_buffer);
                        },
                    );

                    let wait_semaphors = [device.rendering_complete_semaphore];
                    let swapchains = [device.swapchain.swapchain];
                    let image_indices = [present_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices);

                    let _r = device
                        .swapchain_loader
                        .queue_present(device.graphics_queue, &present_info)
                        .unwrap();

                    //   k  device.device.device_wait_idle().unwrap();
                }

                // Event::RedrawRequested(_) => {
                //     // Redraw the application.
                //     //
                //     // It's preferable for applications that do not render continuously to render in
                //     // this event rather than in MainEventsCleared, since rendering in here allows
                //     // the program to gracefully handle redraws requested by the OS.
                // }
                _ => (),
            }
        });
    }
}
