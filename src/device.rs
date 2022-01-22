extern crate ash;
extern crate bitflags;
extern crate spirv_reflect;

use core::slice;
use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{CStr, CString},
};

use ash::{
    vk::{
        self, PhysicalDevice, PhysicalDevicePortabilitySubsetFeaturesKHR,
        PhysicalDevicePortabilitySubsetFeaturesKHRBuilder,
    },
    Entry, Instance,
};

use gpu_allocator::vulkan::*;

pub use crate::gpu_structs::*;

#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            (&b.$field as *const _ as isize) - (&b as *const _ as isize)
        }
    }};
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        return vk::FALSE;
    }

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

#[derive(Clone)]
pub struct Shader {
    pub module: vk::ShaderModule,
    pub inputs: Vec<spirv_reflect::types::ReflectInterfaceVariable>,
    // pub sets: Vec<spirv_reflect::types::ReflectDescriptorSet>,
    // pub bindings: Vec<spirv_reflect::types::ReflectDescriptorBinding>,
    pub entry_point_name: String,
    pub shader_stage: spirv_reflect::types::ReflectShaderStageFlags,
    pub desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}
#[derive(Clone)]
pub struct GPUBuffer<'a> {
    pub allocation: Allocation,
    allocator: &'a RefCell<Allocator>,
    device: &'a ash::Device,
    pub buffer: ash::vk::Buffer,
    pub desc: GPUBufferDesc,
}

impl<'a> Drop for GPUBuffer<'a> {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            self.allocator
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct GPUImage<'a> {
    allocation: Allocation,
    allocator: &'a RefCell<Allocator>,
    device: &'a ash::Device,
    img: vk::Image,
    format: vk::Format,
    view: vk::ImageView,
}

impl GPUImage<'_> {
    pub fn create_view(
        &self,
        aspect: vk::ImageAspectFlags,
        layer_count: u32,
        level_count: u32,
    ) -> vk::ImageView {
        let depth_image_view_info = vk::ImageViewCreateInfo::builder()
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect)
                    .level_count(layer_count)
                    .layer_count(level_count)
                    .build(),
            )
            .image(self.img)
            .format(self.format)
            .view_type(vk::ImageViewType::TYPE_2D);

        unsafe {
            self.device
                .create_image_view(&depth_image_view_info, None)
                .expect("image view creation failed")
        }
    }
}

impl<'a> Drop for GPUImage<'a> {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            self.allocator
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.img, None);
        }
    }
}

pub struct GFXDevice<'a> {
    _entry: Entry,
    instance: ash::Instance,
    pub surface_loader: ash::extensions::khr::Surface,
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,

    pub device: ash::Device,
    pub surface: vk::SurfaceKHR,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub allocator: RefCell<Allocator>,
    pub graphics_queue: vk::Queue,

    descriptor_binder: DescriptorBinder<'a>,
    pdevice: PhysicalDevice,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
}

impl<'a> GFXDevice<'a> {
    // fn bind_vertex_buffer(&self, cmd: &CommandBuffer, buffer: &GPUBuffer) {
    //     unsafe {
    //         self.device.cmd_push_constants(
    //             cmd.cmd,
    //             graphic_pipeline_layout,
    //             vk::ShaderStageFlags::VERTEX,
    //             0,
    //             constants,
    //         );
    //     }
    // }

    fn bind_vertex_buffer(&self, cmd: &CommandBuffer, buffer: &GPUBuffer) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(cmd.cmd, 0, &[buffer.buffer], &[0]);
        }
    }
    fn bind_index_buffer(
        &self,
        cmd: &CommandBuffer,
        index_buffer: &GPUBuffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(cmd.cmd, index_buffer.buffer, 0, index_type);
        }
    }
    // build pipeline if needed
    fn build_pipeline(&self, cmd: &mut CommandBuffer) {
        if cmd.pipeline_is_dirty {
            let pipeline_state = &mut cmd.pipeline_state;

            let vertex_input_attribute_descriptions = &pipeline_state
                .pipeline_desc
                .vertex_input_attribute_descriptions;

            let vertex_input_binding_descriptions = &pipeline_state
                .pipeline_desc
                .vertex_input_binding_descriptions;

            let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
                vertex_attribute_description_count: pipeline_state
                    .pipeline_desc
                    .vertex_input_attribute_descriptions
                    .len() as u32,
                p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
                vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
                p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
                ..Default::default()
            };

            pipeline_state.pipeline_info.p_vertex_input_state = &vertex_input_state_info;

            let graphics_pipelines = unsafe {
                self.device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[pipeline_state.pipeline_info],
                        None,
                    )
                    .expect("Unable to create graphics pipeline")
            };

            cmd.pipeline = graphics_pipelines[0];
            cmd.pipeline_is_dirty = false;
        }
    }

    pub fn flush(&self, cmd: &mut CommandBuffer) {
        unsafe {
            self.build_pipeline(cmd);

            self.device.cmd_bind_pipeline(
                cmd.cmd,
                cmd.pipeline_state.pipeline_desc.bind_point,
                cmd.pipeline,
            );
        }
    }

    pub fn draw_indexed(
        &self,
        cmd: &mut CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.flush(cmd);

            self.device.cmd_draw_indexed(
                cmd.cmd,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn bind_viewport(&self, cmd: &CommandBuffer, viewport: &vk::Viewport) {
        unsafe {
            self.device.cmd_set_viewport(cmd.cmd, 0, &[*viewport]);
            // device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
        }
    }
    pub fn bind_pipeline(&self, pipeline_state: &PipelineState, cmd: &mut CommandBuffer) {
        cmd.pipeline_state = (*pipeline_state).clone();
        cmd.pipeline_is_dirty = true;
    }
    // pub fn bind_pipeline(&self, pipeline_state: &GPUPipelineState) {}
    pub fn create_pipeline_state(&self, desc: &PipelineStateDesc) -> PipelineState {
        unsafe {
            let mut shader_stage_create_infos = vec![];
            let mut set_layouts = vec![];
            let mut push_constant_ranges = vec![];

            if let Some(ref vertex_shader) = desc.vertex {
                let shader_entry_name =
                    CString::new(vertex_shader.entry_point_name.clone()).unwrap();
                shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                    module: vertex_shader.module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                });

                set_layouts.extend(vertex_shader.desc_set_layouts.clone());
                push_constant_ranges.extend(vertex_shader.push_constant_ranges.clone());
            };

            if let Some(ref fragment_shader) = desc.fragment {
                let shader_entry_name =
                    CString::new(fragment_shader.entry_point_name.clone()).unwrap();
                shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                    module: fragment_shader.module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                });
                set_layouts.extend(fragment_shader.desc_set_layouts.clone());
                push_constant_ranges.extend(fragment_shader.push_constant_ranges.clone());
            };

            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: f32::MAX,
                height: f32::MAX,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: u32::MAX,
                    height: u32::MAX,
                },
            }];
            let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
                .scissors(&scissors)
                .viewports(&viewports);

            let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::BACK,
                ..Default::default()
            };
            let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };
            let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            };
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            }];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states);

            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

            // build the layout from shaders , we can cache them in the future
            let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&set_layouts);

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap();

            // renderpass

            let attachments = [
                vk::AttachmentDescription {
                    format: surface_format.format,
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

            let renderpass = self
                .device
                .create_render_pass(&renderpass_ci, None)
                .expect("failed to create a renderpass");

            let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stage_create_infos)
                .input_assembly_state(&vertex_input_assembly_state_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_state_info)
                .depth_stencil_state(&depth_state_info)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(renderpass)
                .build();

            // we attach these in build_pipeline
            // .vertex_input_state(&vertex_input_state_info)
            PipelineState {
                pipeline_info: graphic_pipeline_info,
                pipeline_desc: (*desc).clone(),
            }
        }
    }

    pub fn bind_resource(&mut self, set: u32, binding: u32, buf: &GPUBuffer<'a>) {}

    pub fn create_shader(&self, byte_code: &[u8]) -> Shader {
        unsafe {
            match spirv_reflect::create_shader_module(byte_code) {
                Ok(module) => {
                    let entry_point_name = module.get_entry_point_name();
                    // let _generator = module.get_generator();
                    let mut shader_stage = module.get_shader_stage();
                    // let _source_lang = module.get_source_language();
                    // let _source_lang_ver = module.get_source_language_version();
                    // let _source_file = module.get_source_file();
                    // let _source_text = module.get_source_text();
                    // let _spv_execution_model = module.get_spirv_execution_model();
                    // let _output_vars = module.enumerate_output_variables(None).unwrap();

                    // let bindings = module.enumerate_descriptor_bindings(None).unwrap();
                    let sets = module.enumerate_descriptor_sets(None).unwrap();

                    let input_vars = module.enumerate_input_variables(None).unwrap();

                    let push_constants = module.enumerate_push_constant_blocks(None).unwrap();

                    let code = module.get_code();
                    let shader_info = vk::ShaderModuleCreateInfo::builder().code(&code);

                    let module = self
                        .device
                        .create_shader_module(&shader_info, None)
                        .expect("Shader module error");

                    // create pipeline layout for shader
                    let mut push_constant_ranges = vec![];

                    let shader_stages = {
                        let mut flag = vk::ShaderStageFlags::default();
                        if shader_stage
                            .contains(spirv_reflect::types::ReflectShaderStageFlags::VERTEX)
                        {
                            shader_stage
                                .remove(spirv_reflect::types::ReflectShaderStageFlags::VERTEX);
                            flag |= vk::ShaderStageFlags::VERTEX
                        }
                        if shader_stage
                            .contains(spirv_reflect::types::ReflectShaderStageFlags::FRAGMENT)
                        {
                            shader_stage
                                .remove(spirv_reflect::types::ReflectShaderStageFlags::FRAGMENT);
                            flag |= vk::ShaderStageFlags::FRAGMENT
                        }

                        if !shader_stage.is_empty() {
                            panic!("shader is not supported!");
                        }
                        flag
                    };

                    //push_constants
                    for p in &push_constants {
                        push_constant_ranges.push(
                            vk::PushConstantRange::builder()
                                .stage_flags(shader_stages)
                                .offset(p.offset)
                                .size(p.size)
                                .build(),
                        );
                    }

                    // uniforms
                    let mut desc_set_layouts = vec![];
                    for set in &sets {
                        let mut set_layout_bindings = Vec::with_capacity(set.bindings.len());
                        for bind in &set.bindings {
                            let desc_type = match bind.descriptor_type{
                            spirv_reflect::types::ReflectDescriptorType::Undefined => panic!("undefiend descriptor type "),
                            spirv_reflect::types::ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
                            spirv_reflect::types::ReflectDescriptorType::CombinedImageSampler =>  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            spirv_reflect::types::ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                            spirv_reflect::types::ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                            spirv_reflect::types::ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                            spirv_reflect::types::ReflectDescriptorType::StorageTexelBuffer => todo!(),
                            spirv_reflect::types::ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                            spirv_reflect::types::ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                            spirv_reflect::types::ReflectDescriptorType::UniformBufferDynamic => todo!(),
                            spirv_reflect::types::ReflectDescriptorType::StorageBufferDynamic => todo!(),
                            spirv_reflect::types::ReflectDescriptorType::InputAttachment => todo!(),
                            spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV => todo!(),
                        };

                            let binding_layout = vk::DescriptorSetLayoutBinding::builder()
                                .binding(bind.binding)
                                .descriptor_type(desc_type)
                                .descriptor_count(bind.count)
                                .stage_flags(shader_stages)
                                .build();

                            set_layout_bindings.push(binding_layout);
                        }
                        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                            .bindings(&set_layout_bindings);

                        let desc_set_layout = self
                            .device
                            .create_descriptor_set_layout(&create_info, None)
                            .expect("failed to create descriptor set layout");

                        desc_set_layouts.push(desc_set_layout);
                    }

                    // allocate desc sets used by shaders
                    // let ci = vk::DescriptorSetAllocateInfo::builder()
                    //     .descriptor_pool(self.descriptor_pool)
                    //     .set_layouts(&desc_set_layouts)
                    //     .build();

                    // let desc_sets = self
                    //     .device
                    //     .allocate_descriptor_sets(&ci)
                    //     .expect("failed to allocate descriptor sets");

                    // //TODO(ALI): we use one buffer per binding (this can be optimized)
                    // // update desc
                    // let buffers = vec![];
                    // for set in &sets {
                    //     for binding in &set.bindings {
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
                    //         self.device.update_descriptor_sets(desc_writes, &[]);
                    //     }
                    // }

                    Shader {
                        module,
                        inputs: input_vars,
                        shader_stage,
                        entry_point_name,
                        desc_set_layouts,
                        // desc_sets,
                        push_constant_ranges,
                    }
                }
                Err(err) => {
                    panic!("Error occurred while creating shader - {:?}", err);
                }
            }
        }
    }
    pub fn create_image(&self, desc: &GPUImageDesc, data: Option<&[u8]>) -> GPUImage {
        unsafe {
            let img_info = vk::ImageCreateInfo::builder()
                .array_layers(1)
                .mip_levels(1)
                .extent(vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: desc.depth,
                })
                .format(vk::Format::R8G8B8A8_SRGB)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED);
            // .queue_family_indices(&[]);

            let img = self
                .device
                .create_image(&img_info, None)
                .expect("failed to create image");
            let requirements = self.device.get_image_memory_requirements(img);

            let mut allocation = self
                .allocator
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Image allocation",
                    requirements,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true, // Buffers are always linear
                })
                .expect("failed to allocate image");

            self.device
                .bind_image_memory(img, allocation.memory(), allocation.offset())
                .unwrap();

            match data {
                Some(content) => {
                    allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(content);
                }
                None => {}
            }

            //create view
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .base_mip_level(0)
                .layer_count(1)
                .level_count(1)
                .build();

            let view_info = vk::ImageViewCreateInfo::builder()
                .format(vk::Format::R8G8B8A8_SNORM)
                .image(img)
                .subresource_range(subresource_range)
                .view_type(vk::ImageViewType::TYPE_2D);

            let view = self
                .device
                .create_image_view(&view_info, None)
                .expect("failed to create image view");

            GPUImage {
                allocation,
                allocator: &self.allocator,
                device: &self.device,
                img,
                format: img_info.format,
                view,
            }
        }
    }

    pub fn create_buffer<T>(&self, desc: &GPUBufferDesc, data: Option<&[T]>) -> GPUBuffer {
        unsafe {
            let mut info = vk::BufferCreateInfo::builder()
                .size(desc.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let location = match desc.memory_location {
                MemLoc::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
                MemLoc::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
                MemLoc::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
                MemLoc::Unknown => gpu_allocator::MemoryLocation::Unknown,
            };

            let mut usage = vk::BufferUsageFlags::default();
            if desc.usage.contains(GPUBufferUsage::TRANSFER_SRC) {
                usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(GPUBufferUsage::TRANSFER_DST) {
                usage |= vk::BufferUsageFlags::TRANSFER_DST;
            }
            if desc.usage.contains(GPUBufferUsage::VERTEX_BUFFER) {
                usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
            }
            if desc.usage.contains(GPUBufferUsage::INDEX_BUFFER) {
                usage |= vk::BufferUsageFlags::INDEX_BUFFER;
            }
            if desc.usage.contains(GPUBufferUsage::TRANSFER_SRC) {
                usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(GPUBufferUsage::INDIRECT_BUFFER) {
                usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::UNIFORM_BUFFER) {
                usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::STORAGE_BUFFER) {
                usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::STORAGE_TEXEL_BUFFER) {
                usage |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::UNIFORM_TEXEL_BUFFER) {
                usage |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
            }
            info.usage = usage;

            let buffer = self.device.create_buffer(&info, None).unwrap();
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let mut allocation = self
                .allocator
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Buffer allocation",
                    requirements,
                    location,
                    linear: true, // Buffers are always linear
                })
                .unwrap();

            // Bind memory to the buffer
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();

            match data {
                Some(content) => {
                    let slice = slice::from_raw_parts(
                        content.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(content),
                    );
                    allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(slice);
                }
                None => {}
            }

            GPUBuffer {
                allocation,
                allocator: &self.allocator,
                buffer,
                device: &self.device,
                desc: (*desc).clone(),
            }
        }
    }

    fn init_descriptors(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(1024)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        // let uniform_pool_size = vk::DescriptorPoolSize::builder()
        // .descriptor_count(1024)
        // .ty(vk::DescriptorType::UNIFORM_BUFFER)
        // .build();

        // let uniform_pool_size = vk::DescriptorPoolSize::builder()
        // .descriptor_count(1024)
        // .ty(vk::DescriptorType::UNIFORM_BUFFER)
        // .build();

        let pool_sizes = &[uniform_pool_size];

        let ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(3);

        let desc_pool = unsafe {
            device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descrriptor pool")
        };

        desc_pool
    }

    fn create_swapchain(&self, desc: &SwapchainDesc) -> Swapchain {
        unsafe {
            let surface_format = self
                .surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface)
                .unwrap()[0];

            println!("surface format :{:?}", surface_format);

            let mut desired_image_count = self.surface_capabilities.min_image_count + 1;
            if self.surface_capabilities.max_image_count > 0
                && desired_image_count > self.surface_capabilities.max_image_count
            {
                desired_image_count = self.surface_capabilities.max_image_count;
            }

            let surface_resolution = vk::Extent2D::builder()
                .width(desc.width)
                .height(desc.height);
            let pre_transform = if self
                .surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                self.surface_capabilities.current_transform
            };
            let present_modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(self.pdevice, self.surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(*surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap();

            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    self.device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                })
                .collect();

            let renderpass = {
                let attachments = [
                    vk::AttachmentDescription {
                        format: surface_format.format,
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
                self.device
                    .create_render_pass(&renderpass_ci, None)
                    .expect("failed to create a renderpass")
            };

            let framebuffers: Vec<vk::Framebuffer> = present_image_views
                .iter()
                .map(|&present_image_view| {
                    let framebuffer_attachments = [present_image_view];
                    let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(renderpass)
                        .attachments(&framebuffer_attachments)
                        .width(desc.width)
                        .height(desc.height)
                        .layers(1);

                    self.device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                })
                .collect();

            Swapchain {
                format: surface_format,
                swapchain,
                present_images,
                present_image_views,
                desc: (*desc).clone(),
                swapchain_loader: self.swapchain_loader,
                device: self.device,
            }
        }
    }

    fn create_device(
        instance: &Instance,
        pdevice: PhysicalDevice,
        queue_family_index: u32,
    ) -> ash::Device {
        unsafe {
            let is_vk_khr_portability_subset = instance
                .enumerate_device_extension_properties(pdevice)
                .unwrap()
                .iter()
                .any(|ext| -> bool {
                    let e = CStr::from_ptr(ext.extension_name.as_ptr());
                    // println!("line : {:?} ", e);
                    if e.eq(vk::KhrPortabilitySubsetFn::name()) {
                        return true;
                    }

                    false
                });

            let mut device_extension_names_raw =
                vec![ash::extensions::khr::Swapchain::name().as_ptr()];

            if is_vk_khr_portability_subset {
                device_extension_names_raw.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
            }

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut ci = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features);

            let mut g: PhysicalDevicePortabilitySubsetFeaturesKHRBuilder;
            if is_vk_khr_portability_subset {
                g = PhysicalDevicePortabilitySubsetFeaturesKHR::builder()
                    .image_view_format_swizzle(true);
                ci = ci.push_next(&mut g);
            }

            //            let mut next = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let features2 = &mut vk::PhysicalDeviceFeatures2::builder()
                //              .push_next(&mut next)
                .build();

            instance.get_physical_device_features2(pdevice, features2);

            // println!("next {:?}", next);

            let buffer_address_feature =
                &mut ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true);
            ci = ci.push_next(buffer_address_feature);

            instance
                .create_device(pdevice, &ci, None)
                .expect("device creation failed")
        }
    }

    fn pick_physical_device(
        instance: &Instance,
        surface_loader: &ash::extensions::khr::Surface,
        surface: &vk::SurfaceKHR,
    ) -> (PhysicalDevice, usize) {
        unsafe {
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("physical device error");

            let mut possible_devices = pdevices.iter().filter_map(|pdevice| {
                let props = instance.get_physical_device_queue_family_properties(*pdevice);

                let mut device_match =
                    props
                        .iter()
                        .enumerate()
                        .filter_map(|(queue_family_index, info)| {
                            let mut choose = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);

                            choose = choose
                                && surface_loader
                                    .get_physical_device_surface_support(
                                        *pdevice,
                                        queue_family_index as u32,
                                        *surface,
                                    )
                                    .unwrap();

                            if choose {
                                Some((*pdevice, queue_family_index))
                            } else {
                                None
                            }
                        });

                device_match.next()
            });

            for x in possible_devices.clone() {
                let props = instance.get_physical_device_properties(x.0);

                println!(
                    "device available {:?} , {:?}",
                    CStr::from_ptr(props.device_name.as_ptr()),
                    props.device_type
                );
            }

            let pdevice = possible_devices.next().unwrap();

            let props = instance.get_physical_device_properties(pdevice.0);

            // println!("limits:{:?}", props.limits);

            println!(
                "Picked :{:?} , type:{:?}",
                CStr::from_ptr(props.device_name.as_ptr()),
                props.device_type
            );

            pdevice
        }
    }

    pub fn new(window: &winit::window::Window) -> Self {
        unsafe {
            let entry = Entry::new().unwrap();

            let app_name = CString::new("Sura Engine").unwrap();

            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();

            let mut extension_names_raw = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            extension_names_raw.push(ash::extensions::ext::DebugUtils::name().as_ptr());
            extension_names_raw.push(vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 0, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);

            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();

            let pdevice = GFXDevice::pick_physical_device(&instance, &surface_loader, &surface);

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice.0, surface)
                .unwrap();

            let device = GFXDevice::create_device(&instance, pdevice.0, pdevice.1 as u32);
            let graphics_queue = device.get_device_queue(pdevice.1 as u32, 0);

            // swapchain
            let size = window.inner_size();
            let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
            // let swapchain = GFXDevice::create_swapchain(
            //     &pdevice.0,
            //     &device,
            //     &surface_loader,
            //     &surface,
            //     &swapchain_loader,
            //     size.width as u32,
            //     size.height as u32,
            // );

            let ci = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(pdevice.1 as u32);

            let command_pool = device
                .create_command_pool(&ci, None)
                .expect("pool creation failed");

            let ci = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(swapchain.image_count)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device.allocate_command_buffers(&ci).unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice.0,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .expect("allocator creation failed");

            // let descriptor_pool = GFXDevice::init_descriptors(&device);
            let descriptor_binder = DescriptorBinder::new(&device);
            Self {
                _entry: entry,
                instance,
                debug_call_back,
                debug_utils_loader,
                surface_loader,
                surface,
                surface_capabilities,
                pdevice: pdevice.0,
                device,
                swapchain_loader,
                command_pool,
                command_buffers,
                present_complete_semaphore,
                rendering_complete_semaphore,
                allocator: RefCell::new(allocator),
                graphics_queue,
                descriptor_binder,
            }
        }
    }
}
// create_pipeline();
// create_buffer();
// create_sampler();
// create_texture();

impl Drop for GFXDevice<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_command_pool(self.command_pool, None);

            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);

            drop(self.allocator.get_mut());

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct DescriptorInfo {
    pub set: ash::vk::DescriptorSet,
    pub set_layout: ash::vk::DescriptorSetLayout,
}

struct DescriptorBinder<'a> {
    pub buffers: Vec<GPUBuffer<'a>>,
    // pub bindings: Vec<u32>,
    pub sets: Vec<DescriptorInfo>,
    pub descriptor_pool: vk::DescriptorPool,
}

impl<'a> DescriptorBinder<'a> {
    pub fn new(device: &ash::Device) -> Self {
        let descriptor_pool = Self::init_descriptors(device);

        Self {
            buffers: Vec::with_capacity(16),
            sets: Vec::with_capacity(16),
            // bindings: Vec::with_capacity(16),
            descriptor_pool,
        }
    }

    fn init_descriptors(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(1024)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        let pool_sizes = &[uniform_pool_size];

        let ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(3);

        let desc_pool = unsafe {
            device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descrriptor pool")
        };

        desc_pool
    }

    pub fn get_descriptor_set(
        &mut self,
        device: &ash::Device,
        set_layout: &ash::vk::DescriptorSetLayout,
    ) -> ash::vk::DescriptorSet {
        let mut found_set = None;
        for set in &self.sets {
            if set.set_layout == *set_layout {
                found_set = Some(set)
            }
        }

        if let Some(set) = found_set {
            return set.set;
        }
        let desc_set_layouts = &[*set_layout];
        let ci = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(desc_set_layouts)
            .build();

        let desc_sets = unsafe {
            device
                .allocate_descriptor_sets(&ci)
                .expect("failed to allocate descriptor sets")
        };

        let set = desc_sets[0];
        let set_info = DescriptorInfo {
            set,
            set_layout: set_layout.clone(),
        };

        self.sets.push(set_info);

        set
    }
}
