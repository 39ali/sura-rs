extern crate ash;
extern crate bitflags;
extern crate spirv_reflect;

use core::slice::{self};
use std::{
    borrow::{Borrow, BorrowMut, Cow},
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{hash_map::DefaultHasher, BTreeMap, HashMap},
    ffi::{c_void, CStr, CString},
    hash::{Hash, Hasher},
    mem::{self, ManuallyDrop},
    ops::Deref,
    rc::Rc,
};

use ash::{
    vk::{
        self, AccessFlags, CompareOp, PhysicalDevice, PhysicalDevicePortabilitySubsetFeaturesKHR,
        PhysicalDevicePortabilitySubsetFeaturesKHRBuilder, PhysicalDeviceProperties2Builder,
        SwapchainKHR,
    },
    Entry, Instance,
};

use gpu_allocator::vulkan::*;
use spirv_reflect::types::ReflectDescriptorSet;

pub use crate::gpu_structs::*;

pub type Alloc = Rc<RefCell<ManuallyDrop<Allocator>>>;

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

pub struct VulkanShader {
    pub(crate) device: ash::Device,
    pub module: vk::ShaderModule,
    pub inputs: Vec<spirv_reflect::types::ReflectInterfaceVariable>,
    pub sets: Vec<spirv_reflect::types::ReflectDescriptorSet>,
    pub push_constants: Vec<spirv_reflect::types::ReflectBlockVariable>,
    pub entry_point_name: String,
    pub shader_stage: spirv_reflect::types::ReflectShaderStageFlags,
}
impl Drop for VulkanShader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

impl std::hash::Hash for VulkanShader {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.module.hash(state);
    }
}

// #[derive(Clone)]
pub struct VulkanBuffer {
    pub allocation: ManuallyDrop<gpu_allocator::vulkan::Allocation>,
    pub allocator: Alloc,
    pub device: ash::Device,
    pub buffer: ash::vk::Buffer,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            let allocation = ManuallyDrop::take(&mut self.allocation);
            self.allocator
                .deref()
                .borrow_mut()
                .free(allocation)
                .unwrap();
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct VulkanImage {
    pub allocation: ManuallyDrop<gpu_allocator::vulkan::Allocation>,
    pub allocator: Alloc,
    pub device: ash::Device,
    pub img: vk::Image,
    pub format: vk::Format,
    pub views: RefCell<Vec<vk::ImageView>>,
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        unsafe {
            let allocation = ManuallyDrop::take(&mut self.allocation);
            self.allocator
                .deref()
                .borrow_mut()
                .free(allocation)
                .unwrap();

            for view in self.views.take() {
                self.device.destroy_image_view(view, None);
            }

            self.device.destroy_image(self.img, None);
        }
    }
}

pub struct VkSwapchain {
    pub device: ash::Device,
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    pub desc: SwapchainDesc,

    pub format: vk::SurfaceFormatKHR,
    pub swapchain: SwapchainKHR,
    pub present_images: Vec<vk::Image>, // owned by the OS
    pub present_image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub renderpass: vk::RenderPass,

    pub aquire_semaphore: vk::Semaphore,
    pub release_semaphore: vk::Semaphore,
    pub image_index: u32,
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        unsafe {
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            self.device.destroy_render_pass(self.renderpass, None);

            self.device.destroy_semaphore(self.aquire_semaphore, None);
            self.device.destroy_semaphore(self.release_semaphore, None);

            self.present_image_views.iter().for_each(|v| {
                self.device.destroy_image_view(*v, None);
            });

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct VKPipelineState {
    pub device: ash::Device,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
    pub vertex_input_assembly_state_info: vk::PipelineInputAssemblyStateCreateInfo,
    pub(crate) rasterization_info: vk::PipelineRasterizationStateCreateInfo,
    pub(crate) multisample_state_info: vk::PipelineMultisampleStateCreateInfo,
    pub(crate) depth_state_info: vk::PipelineDepthStencilStateCreateInfo,
    pub dynamic_state: [vk::DynamicState; 2],
    pub(crate) pipeline_layout: vk::PipelineLayout,

    pub(crate) viewports: Vec<vk::Viewport>,
    pub(crate) scissors: Vec<vk::Rect2D>,
    // color blend state
    pub color_blend_logic_op: vk::LogicOp,
    pub color_blend_attachment_states: [vk::PipelineColorBlendAttachmentState; 1],

    pub(crate) renderpass: vk::RenderPass,
}

impl Drop for VKPipelineState {
    fn drop(&mut self) {
        unsafe {
            for layout in &self.set_layouts {
                self.device.destroy_descriptor_set_layout(*layout, None);
            }

            self.device.destroy_render_pass(self.renderpass, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub struct VulkanSampler {
    pub sampler: vk::Sampler,
    pub device: ash::Device,
}

impl Drop for VulkanSampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

pub struct GFXDevice {
    _entry: Entry,
    instance: ash::Instance,
    pub surface_loader: ash::extensions::khr::Surface,
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    pdevice: PhysicalDevice,
    pub device: ash::Device,
    pub surface: vk::SurfaceKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,

    pub allocator: Alloc,
    pub graphics_queue: vk::Queue,
    graphics_queue_index: u32,

    //caches
    pipeline_cache: HashMap<u64, vk::Pipeline>,

    // Frame data
    command_buffers: RefCell<Vec<Vec<CommandBuffer>>>,
    descriptor_binders: RefCell<Vec<DescriptorBinder>>,

    release_fences: Vec<vk::Fence>, //once it's signaled cmds can be reused again
    frame_count: Cell<usize>,
    current_command: Cell<usize>,
    current_swapchain: RefCell<Option<Swapchain>>,
    //
    copy_manager: ManuallyDrop<RefCell<CopyManager>>,
    device_properties: vk::PhysicalDeviceProperties,
}

impl GFXDevice {
    const FRAME_MAX_COUNT: usize = 2;
    const COMMAND_BUFFER_MAX_COUNT: usize = 8;

    pub fn bind_push_constants(&self, cmd: Cmd, pso: &PipelineState, data: &[u8]) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_push_constants(
                cmd.cmd,
                pso.internal.deref().borrow().pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                data,
            );
        }
    }

    pub fn bind_vertex_buffer(&self, cmd: Cmd, buffer: &GPUBuffer, offset: u64) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_bind_vertex_buffers(
                cmd.cmd,
                0,
                &[buffer.internal.deref().borrow().buffer],
                &[offset],
            );
        }
    }
    fn bind_vertex_buffers(&self, _cmd: &CommandBuffer, _buffer: &GPUBuffer, _offset: u32) {
        todo!();
        // unsafe {
        //     self.device
        //         .cmd_bind_vertex_buffers(cmd.cmd, 0, &[buffer.buffer], &[offset]);
        // }
    }
    pub fn bind_index_buffer(
        &self,
        cmd: Cmd,
        index_buffer: &GPUBuffer,
        _offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_bind_index_buffer(
                cmd.cmd,
                index_buffer.internal.deref().borrow().buffer,
                0,
                index_type,
            );
        }
    }
    // build pipeline if needed
    fn build_pipeline(&self, cmd: &mut RefMut<CommandBuffer>) {
        // if cmd.pipeline_is_dirty {
        //     // self.pipeline_cache.get(cmd.prev_pipeline_hash)
        // }

        // self.pipeline_cache.insert(hash, pipeline_state.hash);

        if cmd.pipeline_is_dirty {
            let graphics_pipelines = {
                let pipeline_state = cmd.pipeline_state.as_ref().unwrap();
                let pipeline_desc = &pipeline_state.pipeline_desc;
                let pipeline_state = &pipeline_state.internal.deref().borrow();

                let vertex_input_attribute_descriptions =
                    &pipeline_desc.vertex_input_attribute_descriptions;

                let vertex_input_binding_descriptions =
                    &pipeline_desc.vertex_input_binding_descriptions;

                let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
                    vertex_attribute_description_count: pipeline_desc
                        .vertex_input_attribute_descriptions
                        .len() as u32,
                    p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
                    vertex_binding_description_count: vertex_input_binding_descriptions.len()
                        as u32,
                    p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
                    ..Default::default()
                };

                let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
                    .scissors(&pipeline_state.scissors)
                    .viewports(&pipeline_state.viewports);

                let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                    .logic_op(vk::LogicOp::CLEAR)
                    .attachments(&pipeline_state.color_blend_attachment_states);

                let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
                    .dynamic_states(&pipeline_state.dynamic_state);

                let mut shader_stage_create_infos = vec![];

                let mut shader_entry_names = vec![];
                if let Some(ref vertex_shader) = pipeline_desc.vertex {
                    let vertex_shader = &*vertex_shader.internal;
                    shader_entry_names
                        .push(CString::new(vertex_shader.entry_point_name.clone()).unwrap());
                    let shader_entry_name = shader_entry_names.last().unwrap();
                    shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                        module: vertex_shader.module,
                        p_name: shader_entry_name.as_ptr(),
                        stage: vk::ShaderStageFlags::VERTEX,
                        ..Default::default()
                    });
                };

                if let Some(ref fragment_shader) = pipeline_desc.fragment {
                    let fragment_shader = &*fragment_shader.internal;
                    shader_entry_names
                        .push(CString::new(fragment_shader.entry_point_name.clone()).unwrap());
                    let shader_entry_name = shader_entry_names.last().unwrap();

                    shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                        module: fragment_shader.module,
                        p_name: shader_entry_name.as_ptr(),
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    });
                };

                let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
                    .stages(&shader_stage_create_infos)
                    .input_assembly_state(&pipeline_state.vertex_input_assembly_state_info)
                    .viewport_state(&viewport_state_info)
                    .rasterization_state(&pipeline_state.rasterization_info)
                    .multisample_state(&pipeline_state.multisample_state_info)
                    .depth_stencil_state(&pipeline_state.depth_state_info)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&dynamic_state_info)
                    .layout(pipeline_state.pipeline_layout)
                    .render_pass(pipeline_state.renderpass)
                    .vertex_input_state(&vertex_input_state_info)
                    .build();

                unsafe {
                    self.device
                        .create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            &[graphic_pipeline_info],
                            None,
                        )
                        .expect("Unable to create graphics pipeline")
                }
            };

            cmd.pipeline = Some(graphics_pipelines[0]);
            cmd.pipeline_is_dirty = false;
        }
    }

    fn flush(&self, cmd: &mut RefMut<CommandBuffer>) {
        self.build_pipeline(cmd);

        let pso = cmd.pipeline_state.as_ref().unwrap();
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd.cmd,
                pso.pipeline_desc.bind_point,
                cmd.pipeline.unwrap(),
            );
        }

        let mut sets = vec![];
        let desc_binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

        let pso = pso.internal.deref().borrow();

        for set in &pso.set_layouts {
            let desc_set = desc_binder.get_descriptor_set(&self.device, set);

            sets.push(desc_set);
        }

        unsafe {
            // update desc
            for bind in &desc_binder.binder_buff {
                let dst_set = sets[bind.0 .0 as usize];
                let dst_binding = bind.0 .1;
                let buffer = bind.1.internal.deref().borrow().buffer;

                let desc_buffer = vk::DescriptorBufferInfo::builder()
                    .range(vk::WHOLE_SIZE)
                    .buffer(buffer)
                    .offset(0)
                    .build();
                //update desc set
                let wds = vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(dst_set)
                    .dst_binding(dst_binding)
                    .dst_array_element(0)
                    .buffer_info(&[desc_buffer])
                    .build();
                let desc_writes = &[wds];
                self.device.update_descriptor_sets(desc_writes, &[]);
            }

            for bind in &desc_binder.binder_img {
                let dst_set = sets[bind.0 .0 as usize];
                let dst_binding = bind.0 .1;
                let img_view_index = bind.1 .0;
                let sampler = bind.1 .1.internal.deref().borrow().sampler;
                let gpu_img = &bind.1 .2;
                let vk_img = gpu_img.internal.deref().borrow();
                let img_view = vk_img.views.borrow()[img_view_index as usize];

                let image_info = vk::DescriptorImageInfo::builder()
                    .image_view(img_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler)
                    .build();
                //update desc set
                let wds = vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(dst_set)
                    .dst_binding(dst_binding)
                    .dst_array_element(0)
                    .image_info(&[image_info])
                    .build();
                let desc_writes = &[wds];
                self.device.update_descriptor_sets(desc_writes, &[]);
            }

            self.device.cmd_bind_descriptor_sets(
                cmd.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pso.pipeline_layout,
                0,
                &sets,
                &[],
            );
        }
    }

    pub fn draw_indexed(
        &self,
        cmd: Cmd,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            let mut cmd = self.get_cmd_mut(cmd);
            self.flush(&mut cmd);

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

    pub fn bind_viewports(&self, cmd: Cmd, viewports: &[vk::Viewport]) {
        let cmd = self.get_cmd(cmd);
        unsafe {
            self.device.cmd_set_viewport(cmd.cmd, 0, viewports);
        }
    }

    pub fn bind_scissors(&self, cmd: Cmd, scissors: &[vk::Rect2D]) {
        let cmd = self.get_cmd(cmd);
        unsafe {
            self.device.cmd_set_scissor(cmd.cmd, 0, &scissors);
        }
    }

    pub fn bind_pipeline(&self, cmd: Cmd, pipeline_state: &PipelineState) {
        let mut cmd = self.get_cmd_mut(cmd);

        if cmd.pipeline_state.is_none() {
            cmd.pipeline_state = Some(pipeline_state.clone());
            cmd.pipeline_is_dirty = true;
            return;
        }

        if cmd.pipeline_state.as_ref().unwrap().hash == pipeline_state.hash {
            return;
        }

        //FIXME : we leak pipeline if the pipeline_state changes
        cmd.prev_pipeline_hash = cmd.pipeline_state.as_ref().unwrap().hash;
        cmd.pipeline_state = Some(pipeline_state.clone());
        cmd.pipeline_is_dirty = true;
    }

    fn create_pipeline_layout(
        &self,
        desc: &PipelineStateDesc,
    ) -> (
        vk::PipelineLayout,
        Vec<vk::DescriptorSetLayout>,
        Vec<vk::PushConstantRange>,
    ) {
        //merge descriptor sets from shaders
        let mut sets = BTreeMap::<u32, ReflectDescriptorSet>::new();
        let mut push_constants_ranges = vec![];
        if desc.vertex.is_some() {
            for set in &desc.vertex.as_ref().unwrap().internal.sets {
                if sets.contains_key(&set.set) {
                    sets.get_mut(&set.set)
                        .unwrap()
                        .bindings
                        .extend_from_slice(&set.bindings[..]);
                } else {
                    sets.insert(set.set, set.clone());
                }
            }

            //push_constants
            for p in &desc.vertex.as_ref().unwrap().internal.push_constants {
                push_constants_ranges.push(
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(p.offset)
                        .size(p.size)
                        .build(),
                );
            }
        }
        if desc.fragment.is_some() {
            for set in &desc.fragment.as_ref().unwrap().internal.sets {
                if sets.contains_key(&set.set) {
                    sets.get_mut(&set.set)
                        .unwrap()
                        .bindings
                        .extend_from_slice(&set.bindings[..]);
                } else {
                    sets.insert(set.set, set.clone());
                }
            }

            //push_constants
            for p in &desc.fragment.as_ref().unwrap().internal.push_constants {
                push_constants_ranges.push(
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .offset(p.offset)
                        .size(p.size)
                        .build(),
                );
            }
        }

        let mut desc_set_layouts = Vec::with_capacity(sets.len());
        for set in &sets {
            let set_index = set.0;
            let set = set.1;
            let mut set_layout_bindings = Vec::with_capacity(set.bindings.len());
            for bind in &set.bindings {
                let desc_type = match bind.descriptor_type {
                    spirv_reflect::types::ReflectDescriptorType::Undefined => {
                        panic!("undefiend descriptor type ")
                    }
                    spirv_reflect::types::ReflectDescriptorType::Sampler => {
                        vk::DescriptorType::SAMPLER
                    }
                    spirv_reflect::types::ReflectDescriptorType::CombinedImageSampler => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    spirv_reflect::types::ReflectDescriptorType::SampledImage => {
                        vk::DescriptorType::SAMPLED_IMAGE
                    }
                    spirv_reflect::types::ReflectDescriptorType::StorageImage => {
                        vk::DescriptorType::STORAGE_IMAGE
                    }
                    spirv_reflect::types::ReflectDescriptorType::UniformTexelBuffer => {
                        vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    }
                    spirv_reflect::types::ReflectDescriptorType::StorageTexelBuffer => todo!(),
                    spirv_reflect::types::ReflectDescriptorType::UniformBuffer => {
                        vk::DescriptorType::UNIFORM_BUFFER
                    }
                    spirv_reflect::types::ReflectDescriptorType::StorageBuffer => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    spirv_reflect::types::ReflectDescriptorType::UniformBufferDynamic => todo!(),
                    spirv_reflect::types::ReflectDescriptorType::StorageBufferDynamic => todo!(),
                    spirv_reflect::types::ReflectDescriptorType::InputAttachment => todo!(),
                    spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV => todo!(),
                };

                //NOTE : does using 'ALL' effect preformance ?
                let shader_stages = vk::ShaderStageFlags::ALL;
                let binding_layout = vk::DescriptorSetLayoutBinding::builder()
                    .binding(bind.binding)
                    .descriptor_type(desc_type)
                    .descriptor_count(bind.count)
                    .stage_flags(shader_stages)
                    .build();

                set_layout_bindings.push(binding_layout);
            }

            let create_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_layout_bindings);

            unsafe {
                let desc_set_layout = self
                    .device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("failed to create descriptor set layout");
                desc_set_layouts.push(desc_set_layout);
            }
        }

        // build the layout from shaders , we can cache them in the future
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constants_ranges)
            .set_layouts(&desc_set_layouts);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_create_info, None)
                .expect("failed to create pipelinelayout")
        };

        (pipeline_layout, desc_set_layouts, push_constants_ranges)
    }

    pub fn create_pipeline_state(&self, desc: &PipelineStateDesc) -> PipelineState {
        let mut s = DefaultHasher::new();
        desc.hash(&mut s);
        let hash = s.finish();

        // TODO:cache this
        let (pipeline_layout, set_layouts, push_constant_ranges) =
            self.create_pipeline_layout(desc);

        unsafe {
            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };
            let viewports = vec![vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: f32::MAX,
                height: f32::MAX,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = vec![vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: u32::MAX,
                    height: u32::MAX,
                },
            }];
            // let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            //     .scissors(&scissors)
            //     .viewports(&viewports);

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
            // let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            //     .logic_op(vk::LogicOp::CLEAR)
            //     .attachments(&color_blend_attachment_states);

            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            // let dynamic_state_info =
            //     vk::PipelineDynamicStateCreateInfo::builder().dynamic_stats(&dynamic_state);

            // renderpass

            let surface_format = self
                .surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface)
                .unwrap()[0];

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

            // let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            //     .stages(&shader_stage_create_infos)
            //     .input_assembly_state(&vertex_input_assembly_state_info)
            //     .viewport_state(&viewport_state_info)
            //     .rasterization_state(&rasterization_info)
            //     .multisample_state(&multisample_state_info)
            //     .depth_stencil_state(&depth_state_info)
            //     .color_blend_state(&color_blend_state)
            //     .dynamic_state(&dynamic_state_info)
            //     .layout(pipeline_layout)
            //     .render_pass(renderpass)
            //     .build();

            // we attach these in build_pipeline
            // .vertex_input_state(&vertex_input_state_info)
            PipelineState {
                pipeline_desc: (*desc).clone(),
                hash,
                internal: Rc::new(RefCell::new(VKPipelineState {
                    device: self.device.clone(),
                    set_layouts,
                    push_constant_ranges,
                    vertex_input_assembly_state_info,
                    viewports,
                    scissors,
                    rasterization_info,
                    multisample_state_info,
                    depth_state_info,
                    color_blend_logic_op: vk::LogicOp::CLEAR,
                    dynamic_state,
                    pipeline_layout,
                    renderpass,
                    color_blend_attachment_states,
                })),
            }
        }
    }

    pub fn bind_resource_buffer(&self, set: u32, binding: u32, buf: &GPUBuffer) {
        let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

        binder.binder_buff.insert((set, binding), (*buf).clone());
    }

    pub fn bind_resource_img(
        &self,
        set: u32,
        binding: u32,
        img: &GPUImage,
        view_index: u32,
        sampler: &Sampler,
    ) {
        let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

        binder.binder_img.insert(
            (set, binding),
            (view_index, sampler.clone(), (*img).clone()),
        );
    }

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

                    let vshader = VulkanShader {
                        device: self.device.clone(),
                        module,
                        inputs: input_vars,
                        shader_stage,
                        entry_point_name,
                        sets,
                        push_constants,
                    };
                    Shader {
                        internal: Rc::new(vshader),
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
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
            // .queue_family_indices(&[]);

            let img = self
                .device
                .create_image(&img_info, None)
                .expect("failed to create image");
            let requirements = self.device.get_image_memory_requirements(img);

            let mut allocation = self
                .allocator
                .deref()
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Image allocation",
                    requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                })
                .expect("failed to allocate image");

            self.device
                .bind_image_memory(img, allocation.memory(), allocation.offset())
                .expect("image memory  binding failed");

            //create view
            // let subresource_range = vk::ImageSubresourceRange::builder()
            //     .aspect_mask(vk::ImageAspectFlags::COLOR)
            //     .base_array_layer(0)
            //     .base_mip_level(0)
            //     .layer_count(1)
            //     .level_count(1)
            //     .build();

            // let view_info = vk::ImageViewCreateInfo::builder()
            //     .format(vk::Format::R8G8B8A8_SNORM)
            //     .image(img)
            //     .subresource_range(subresource_range)
            //     .view_type(vk::ImageViewType::TYPE_2D);

            // let view = self
            //     .device
            //     .create_image_view(&view_info, None)
            //     .expect("failed to create image view");

            let gpu_image = GPUImage {
                desc: *desc,
                internal: Rc::new(RefCell::new(VulkanImage {
                    allocation: ManuallyDrop::new(allocation),
                    allocator: self.allocator.clone(),
                    device: self.device.clone(),
                    img,
                    format: img_info.format,
                    views: RefCell::new(Vec::new()),
                })),
            };

            match data {
                Some(content) => {
                    self.copy_manager
                        .deref()
                        .borrow_mut()
                        .copy_image(self, &gpu_image, content);

                    // allocation
                    //     .mapped_slice_mut()
                    //     .unwrap()
                    //     .copy_from_slice(content);
                }
                _ => {}
            }

            gpu_image
        }
    }
    pub fn create_image_view(
        &self,
        img: &GPUImage,
        aspect: vk::ImageAspectFlags,
        layer_count: u32,
        level_count: u32,
    ) -> u32 {
        let internal = img.internal.deref().borrow_mut();
        let depth_image_view_info = vk::ImageViewCreateInfo::builder()
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect)
                    .level_count(layer_count)
                    .layer_count(level_count)
                    .build(),
            )
            .image(internal.img)
            .format(internal.format)
            .view_type(vk::ImageViewType::TYPE_2D);

        let view = unsafe {
            self.device
                .create_image_view(&depth_image_view_info, None)
                .expect("image view creation failed")
        };

        let index = (internal.views.borrow().len()) as u32;
        internal.views.borrow_mut().push(view);
        index
    }

    pub fn create_sampler(&self) -> Sampler {
        let create_info = vk::SamplerCreateInfo::builder()
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(self.device_properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .unnormalized_coordinates(false)
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0f32)
            .min_lod(0f32)
            .max_lod(0f32);

        let sampler = unsafe {
            self.device
                .create_sampler(&create_info, None)
                .expect("creating sampler failed")
        };

        Sampler {
            internal: Rc::new(RefCell::new(VulkanSampler {
                sampler,
                device: self.device.clone(),
            })),
        }
    }
    pub fn create_buffer(&self, desc: &GPUBufferDesc, data: Option<&[u8]>) -> GPUBuffer {
        unsafe {
            let mut info = vk::BufferCreateInfo::builder()
                .size(desc.size as u64)
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

            let mut allocation = (*self.allocator)
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Buffer allocation",
                    requirements,
                    location,
                    linear: true, // Buffers are always linear
                })
                .unwrap();

            let gpu_buffer = GPUBuffer {
                internal: Rc::new(RefCell::new(VulkanBuffer {
                    allocation: ManuallyDrop::new(allocation),
                    allocator: self.allocator.clone(),
                    buffer,
                    device: self.device.clone(),
                })),
                desc: *desc,
            };

            // Bind memory to the buffer
            self.device
                .bind_buffer_memory(
                    buffer,
                    gpu_buffer.internal.deref().borrow().allocation.memory(),
                    gpu_buffer.internal.deref().borrow().allocation.offset(),
                )
                .unwrap();

            if data.is_some() {
                let content = data.unwrap();
                if location.eq(&gpu_allocator::MemoryLocation::GpuOnly) {
                    self.copy_manager
                        .deref()
                        .borrow_mut()
                        .copy_buffer(self, &gpu_buffer, content)
                } else {
                    gpu_buffer
                        .internal
                        .deref()
                        .borrow_mut()
                        .allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(content);
                }
            }

            gpu_buffer
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

        unsafe {
            device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descrriptor pool")
        }
    }

    pub fn begin_command_buffer(&self) -> Cmd {
        assert!(self.current_command.get() + 1 < GFXDevice::COMMAND_BUFFER_MAX_COUNT);

        let cmd_index = self.current_command.get();

        let cmd = &self.command_buffers.borrow_mut()[self.get_current_frame_index()][cmd_index];

        self.current_command.set(self.current_command.get() + 1);

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd.cmd, &command_buffer_begin_info)
                .expect("Begin commandbuffer");
        }

        Cmd(cmd_index)
    }

    pub fn end_command_buffers(&self) {
        let cmds = &self.command_buffers.borrow()[self.get_current_frame_index()];
        unsafe {
            let cmd_count = self.current_command.get();
            // end used cmds
            for i in 0..cmd_count {
                let cmd = &cmds[i];
                self.device
                    .end_command_buffer(cmd.cmd)
                    .expect("End commandbuffer");
            }

            let command_buffers: Vec<vk::CommandBuffer> =
                cmds[..cmd_count].iter().map(|b| b.cmd).collect();

            let mut wait_semaphores = Vec::with_capacity(2);
            let mut signal_semaphores = Vec::with_capacity(1);

            if self.current_swapchain.borrow().is_some() {
                wait_semaphores.push(
                    self.current_swapchain
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .internal
                        .deref()
                        .borrow()
                        .aquire_semaphore,
                );
                signal_semaphores.push(
                    self.current_swapchain
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .internal
                        .deref()
                        .borrow()
                        .release_semaphore,
                );
            };

            let mut submits = vec![];

            // flush copy manager
            // if swapchain is enabled wait for the aquire semaphore
            let copy_manager_wait_semaphore = if wait_semaphores.is_empty() {
                vk::Semaphore::null()
            } else {
                wait_semaphores[0]
            };
            if self.copy_manager.deref().borrow_mut().flush(
                self,
                copy_manager_wait_semaphore,
                &mut submits,
                &mut wait_semaphores,
            ) {
                //if flush was true then we're waiting for the aquire semaphore
                // meaning that the rendering submit doesn't need to wait for it
                wait_semaphores.swap_remove(0);
            }
            //

            // submit render submit info to queue
            let render_submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]) // TODO: is this ok for compute ?
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();

            submits.push(render_submit_info);

            let release_fence = self.release_fences[self.get_current_frame_index()];

            self.device
                .queue_submit(self.graphics_queue, submits.as_slice(), release_fence)
                .expect("queue submit failed.");

            self.device
                .wait_for_fences(&[release_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            self.device
                .reset_fences(&[release_fence])
                .expect("Reset fences failed.");

            //reset pools
            for cmd in &cmds[..cmd_count] {
                self.device
                    .reset_command_pool(cmd.command_pool, vk::CommandPoolResetFlags::empty())
                    .expect("reset cmd pool failed");
            }
            //reset counter
            self.current_command.set(0);
            self.frame_count.set(self.frame_count.get() + 1);

            //rest copy manager
            self.copy_manager.deref().borrow_mut().reset(self);

            // present the queue
            let current_swapchain = self.current_swapchain.borrow();
            if current_swapchain.is_some() {
                let internal = &current_swapchain.as_ref().unwrap().internal;
                let swapchains = &[internal.deref().borrow().swapchain];
                let image_indices = &[self
                    .current_swapchain
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .internal
                    .deref()
                    .borrow()
                    .image_index];
                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&signal_semaphores)
                    .swapchains(swapchains)
                    .image_indices(image_indices);
                self.swapchain_loader
                    .queue_present(self.graphics_queue, &present_info)
                    .expect("Queue present error");
            }
        }
    }

    pub fn begin_renderpass(&self, cmd: Cmd, swapchain: &Swapchain) {
        let cmd = self.get_cmd(cmd);

        let mut internal = (*swapchain.internal).borrow_mut();

        unsafe {
            let (present_index, _) = self
                .swapchain_loader
                .acquire_next_image(
                    internal.swapchain,
                    u64::MAX,
                    internal.aquire_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();

            (internal).image_index = present_index;
            let sp = RefCell::new(Some(swapchain.clone()));
            self.current_swapchain.swap(&sp);

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: internal.desc.clearcolor,
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
                .render_pass(internal.renderpass)
                .framebuffer(internal.framebuffers[internal.image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: internal.desc.width,
                        height: internal.desc.height,
                    },
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(
                cmd.cmd,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_renderpass(&self, cmd: Cmd) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_end_render_pass(cmd.cmd);
        }
    }

    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Swapchain {
        unsafe {
            let surface_format = self
                .surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface)
                .unwrap()[0];

            // println!("surface format :{:?}", surface_format);

            let desired_image_count = {
                if desc.framebuffer_count <= self.surface_capabilities.max_image_count {
                    desc.framebuffer_count
                } else {
                    println!("warning framebuffer_count is bigger than  surface_capabilities.max_image_count , using {} instead " , self.surface_capabilities.max_image_count);
                    self.surface_capabilities.max_image_count
                }
            };

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

            let present_mode = {
                if desc.vsync {
                    vk::PresentModeKHR::FIFO
                } else {
                    let present_modes = self
                        .surface_loader
                        .get_physical_device_surface_present_modes(self.pdevice, self.surface)
                        .unwrap();

                    *present_modes
                        .iter()
                        .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
                        .unwrap_or(&vk::PresentModeKHR::FIFO)
                }
            };

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

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let aquire_semaphore = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let release_semaphore = self
                .device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            Swapchain {
                internal: Rc::new(RefCell::new(VkSwapchain {
                    format: surface_format,
                    swapchain,
                    present_images,
                    present_image_views,
                    desc: (*desc).clone(),
                    swapchain_loader: self.swapchain_loader.clone(),
                    device: self.device.clone(),
                    framebuffers,
                    renderpass,
                    aquire_semaphore,
                    release_semaphore,
                    image_index: 0,
                })),
            }
        }
    }

    fn get_cmd(&self, cmd: Cmd) -> Ref<CommandBuffer> {
        let cmd = Ref::map(self.command_buffers.borrow(), |f| {
            &f[self.get_current_frame_index()][cmd.0]
        });
        cmd
    }

    fn get_cmd_mut(&self, cmd: Cmd) -> RefMut<CommandBuffer> {
        let cmd = RefMut::map(self.command_buffers.borrow_mut(), |f| {
            &mut f[self.get_current_frame_index()][cmd.0]
        });
        cmd
    }

    fn get_current_frame_index(&self) -> usize {
        self.frame_count.get() % GFXDevice::FRAME_MAX_COUNT
    }

    fn init_frames(
        device: &ash::Device,
        graphics_queue_index: u32,
    ) -> (
        RefCell<Vec<Vec<CommandBuffer>>>,
        RefCell<Vec<DescriptorBinder>>,
        Vec<vk::Fence>,
    ) {
        unsafe {
            let mut release_fence = Vec::with_capacity(GFXDevice::FRAME_MAX_COUNT);
            let mut command_buffers: Vec<Vec<CommandBuffer>> =
                Vec::with_capacity(GFXDevice::FRAME_MAX_COUNT);
            let mut descriptor_binders: Vec<DescriptorBinder> =
                Vec::with_capacity(GFXDevice::FRAME_MAX_COUNT);

            for _i in 0..GFXDevice::FRAME_MAX_COUNT {
                let descriptor_binder = DescriptorBinder::new(device);
                let mut cmds = Vec::with_capacity(GFXDevice::COMMAND_BUFFER_MAX_COUNT);

                for j in 0..GFXDevice::COMMAND_BUFFER_MAX_COUNT {
                    let ci = vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                        .queue_family_index(graphics_queue_index);
                    let command_pool = device
                        .create_command_pool(&ci, None)
                        .expect("pool creation failed");

                    let ci = vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY);

                    let command = device
                        .allocate_command_buffers(&ci)
                        .expect("Failed to allocated cmd buffer")[0];

                    cmds.push(CommandBuffer {
                        command_pool,
                        cmd: command,
                        pipeline_state: None,
                        pipeline_is_dirty: true,
                        pipeline: None,
                        prev_pipeline_hash: 0,
                    });
                }

                let info = vk::FenceCreateInfo {
                    ..Default::default()
                };

                let fence = device
                    .create_fence(&info, None)
                    .expect("failed to create fence");

                release_fence.push(fence);

                descriptor_binders.push(descriptor_binder);

                command_buffers.push(cmds);
            }

            (
                RefCell::new(command_buffers),
                RefCell::new(descriptor_binders),
                release_fence,
            )
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

            let features2 = &mut vk::PhysicalDeviceFeatures2::default();
            instance.get_physical_device_features2(pdevice, features2);

            let buffer_address_feature =
                &mut ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true);

            info!("device features :{:?}", features2);

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut ci = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(features2)
                .push_next(buffer_address_feature);

            let mut g: PhysicalDevicePortabilitySubsetFeaturesKHRBuilder;
            if is_vk_khr_portability_subset {
                g = PhysicalDevicePortabilitySubsetFeaturesKHR::builder()
                    .image_view_format_swizzle(true);
                ci = ci.push_next(&mut g);
            };

            instance
                .create_device(pdevice, &ci, None)
                .expect("device creation failed")
        }
    }

    fn pick_physical_device(
        instance: &Instance,
        surface_loader: &ash::extensions::khr::Surface,
        surface: &vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, usize, vk::PhysicalDeviceProperties) {
        unsafe {
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("physical device error");

            let possible_devices: Vec<(vk::PhysicalDevice, usize, vk::PhysicalDeviceProperties)> =
                pdevices
                    .iter()
                    .filter_map(|pdevice| {
                        let props = instance.get_physical_device_queue_family_properties(*pdevice);

                        let mut device_match =
                            props
                                .iter()
                                .enumerate()
                                .filter_map(|(queue_family_index, info)| {
                                    let mut choose =
                                        info.queue_flags.contains(vk::QueueFlags::GRAPHICS);

                                    choose = choose
                                        && surface_loader
                                            .get_physical_device_surface_support(
                                                *pdevice,
                                                queue_family_index as u32,
                                                *surface,
                                            )
                                            .unwrap();

                                    let props = instance.get_physical_device_properties(*pdevice);

                                    if choose {
                                        Some((*pdevice, queue_family_index, props))
                                    } else {
                                        None
                                    }
                                });

                        device_match.next()
                    })
                    .collect();

            for x in &possible_devices {
                debug!(
                    "device available {:?} , {:?}",
                    CStr::from_ptr(x.2.device_name.as_ptr()),
                    x.2.device_type
                );
            }

            let pdevice = *possible_devices
                .iter()
                .find(|d| {
                    let props = &d.2;
                    if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        return true;
                    };
                    false
                })
                .or_else(|| possible_devices.first())
                .unwrap();

            info!(
                "Picked :{:?} , type:{:?}",
                CStr::from_ptr(pdevice.2.device_name.as_ptr()),
                pdevice.2.device_type
            );

            pdevice
        }
    }

    pub fn wait_for_gpu(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle error ");
        }
    }

    pub fn new(window: &winit::window::Window) -> Self {
        unsafe {
            let entry = unsafe { Entry::load().expect("failed to load vulkan dll") };

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
            let device_properties = pdevice.2;
            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice.0, surface)
                .unwrap();

            let device = GFXDevice::create_device(&instance, pdevice.0, pdevice.1 as u32);

            let graphics_queue_index = pdevice.1 as u32;
            let graphics_queue = device.get_device_queue(graphics_queue_index, 0);

            // swapchain
            let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice.0,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .expect("allocator creation failed");

            let (command_buffers, descriptor_binders, release_fences) =
                GFXDevice::init_frames(&device, graphics_queue_index);

            let copy_manager = ManuallyDrop::new(RefCell::new(CopyManager::new(
                graphics_queue_index,
                &device,
            )));

            let gfx = Self {
                _entry: entry,
                instance,
                debug_call_back,
                debug_utils_loader,
                surface_loader,
                surface,
                surface_capabilities,
                pdevice: pdevice.0,
                device_properties,
                device,
                swapchain_loader,
                allocator: Rc::new(RefCell::new(ManuallyDrop::new(allocator))),
                graphics_queue,
                graphics_queue_index,
                pipeline_cache: HashMap::new(),
                // Frame data
                command_buffers,
                descriptor_binders,
                release_fences,
                frame_count: Cell::new(0),
                current_command: Cell::new(0),
                current_swapchain: RefCell::new(None),
                //
                copy_manager,
            };

            gfx
        }
    }
}

impl Drop for GFXDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            // drop commad buffers
            {
                let mut command_buffers = self.command_buffers.borrow_mut();
                for frame_cmds in command_buffers.iter() {
                    for cmd in frame_cmds {
                        if cmd.pipeline.is_some() {
                            self.device.destroy_pipeline(cmd.pipeline.unwrap(), None);
                        }
                        self.device.destroy_command_pool(cmd.command_pool, None);
                    }
                }
                command_buffers.clear();
            }

            //drop  descriptor sets
            self.descriptor_binders.borrow_mut().clear();

            for fence in &self.release_fences {
                self.device.destroy_fence(*fence, None);
            }

            let current_swapchain = self.current_swapchain.replace(None);
            drop(current_swapchain);

            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);

            //free copy manager
            ManuallyDrop::drop(&mut self.copy_manager);
            let allocator = &mut *(*self.allocator).borrow_mut();
            ManuallyDrop::drop(allocator);

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct DescriptorInfo {
    pub set: ash::vk::DescriptorSet,
    pub set_layout: ash::vk::DescriptorSetLayout,
}

struct DescriptorBinder {
    pub device: ash::Device,
    pub sets: Vec<DescriptorInfo>,
    pub descriptor_pool: vk::DescriptorPool,
    //TODO : create a struct for this tuple
    // set , bind
    pub binder_buff: HashMap<(u32, u32), GPUBuffer>,
    // set , bind , view_index , sampler ,gpuimage
    pub binder_img: HashMap<(u32, u32), (u32, Sampler, GPUImage)>,
}

impl DescriptorBinder {
    pub fn new(device: &ash::Device) -> Self {
        let descriptor_pool = Self::init_descriptors(device);

        Self {
            device: device.clone(),
            binder_buff: HashMap::new(),
            binder_img: HashMap::new(),
            sets: Vec::with_capacity(16),
            descriptor_pool,
        }
    }

    fn init_descriptors(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(1024)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        let combined_image_sampler_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(1024)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build();

        let pool_sizes = &[uniform_pool_size, combined_image_sampler_pool_size];

        let ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(3);

        unsafe {
            device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descrriptor pool")
        }
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
            set_layout: *set_layout,
        };

        self.sets.push(set_info);

        set
    }
}

impl Drop for DescriptorBinder {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.binder_buff.clear();
            self.binder_img.clear();
        }
    }
}

#[derive(Default)]
struct CopyManager {
    free_buffers: Vec<GPUBuffer>,
    used_buffers: Vec<GPUBuffer>,
    cmd: CommandBuffer,

    // needed to be alive for vk::submitInfo
    copy_wait_semaphores: Vec<vk::Semaphore>,
    copy_cmds: Vec<vk::CommandBuffer>,
    copy_signal_semaphores: Vec<vk::Semaphore>,
}

impl CopyManager {
    const BUFFERS_COUNT: usize = 10;
    pub fn new(transfer_queue_index: u32, device: &ash::Device) -> Self {
        let cmd = unsafe {
            let ci = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(transfer_queue_index);
            let command_pool = device
                .create_command_pool(&ci, None)
                .expect("pool creation failed");

            let ci = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let cmd = device
                .allocate_command_buffers(&ci)
                .expect("Failed to allocated cmd buffer")[0];

            CommandBuffer {
                command_pool,
                cmd,
                pipeline_state: None,
                pipeline_is_dirty: false,
                prev_pipeline_hash: 0,
                pipeline: None,
            }
        };

        let free_buffers = Vec::with_capacity(Self::BUFFERS_COUNT);
        let used_buffers = Vec::with_capacity(Self::BUFFERS_COUNT);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let copy_semaphore = unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        let copy_signal_semaphores = vec![copy_semaphore];

        let copy_wait_semaphores = Vec::<vk::Semaphore>::new();
        let copy_cmds = vec![cmd.cmd];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd.cmd, &begin_info).unwrap() }

        CopyManager {
            free_buffers,
            used_buffers,
            cmd,
            copy_signal_semaphores,
            copy_wait_semaphores,
            copy_cmds,
        }
    }

    fn pick_stagging_buffer(&mut self, size: usize, gfx: &GFXDevice) -> GPUBuffer {
        let used_buffer: Option<&GPUBuffer> = None;
        let used_buffer_i = 0;

        let used_buffer_i = self.free_buffers.iter().position(|buffer| {
            if buffer.desc.size <= size {
                return true;
            }
            false
        });

        let used_buffer = match used_buffer_i {
            Some(i) => {
                let buff = self.free_buffers.swap_remove(i);
                self.used_buffers.push(buff.clone());
                buff
            }
            None => {
                let desc = GPUBufferDesc {
                    index_buffer_type: None,
                    size: size,
                    memory_location: MemLoc::CpuToGpu,
                    usage: GPUBufferUsage::TRANSFER_SRC,
                };
                let new_buffer = gfx.create_buffer(&desc, None);
                self.used_buffers.push(new_buffer.clone());
                new_buffer
            }
        };

        used_buffer
    }

    fn transition_image_layout(
        &self,
        gfx: &GFXDevice,
        img: &GPUImage,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let vk_cmd = self.cmd.cmd;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .base_mip_level(0)
            .layer_count(1)
            .level_count(1)
            .build();
        let mut image_memory_barrier = vk::ImageMemoryBarrier::builder()
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.internal.deref().borrow().img)
            .subresource_range(subresource_range)
            .old_layout(old_layout)
            .new_layout(new_layout);

        let (src_stage_mask, dst_stage_mask) = if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            image_memory_barrier = image_memory_barrier
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            image_memory_barrier = image_memory_barrier
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        } else {
            panic!("wut");
        };

        let image_memory_barriers = &[image_memory_barrier.build()];
        unsafe {
            gfx.device.cmd_pipeline_barrier(
                vk_cmd,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                image_memory_barriers,
            );
        }
    }
    pub fn copy_image(&mut self, gfx: &GFXDevice, img: &GPUImage, data: &[u8]) {
        let used_buffer = self.pick_stagging_buffer(img.desc.size, gfx);
        let vk_buff = used_buffer.internal;
        let vk_cmd = self.cmd.cmd;

        // copy data to staging buffer
        unsafe {
            let src = data.as_ptr().cast::<c_void>();
            let dst = vk_buff
                .deref()
                .borrow_mut()
                .allocation
                .mapped_ptr()
                .unwrap();
            dst.as_ptr()
                .copy_from_nonoverlapping(src, std::mem::size_of_val(data));
        }

        // change image layout
        let dst_image_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        self.transition_image_layout(gfx, img, vk::ImageLayout::UNDEFINED, dst_image_layout);

        //copy buffer to image memory
        let image_subresource = vk::ImageSubresourceLayers {
            layer_count: 1,
            base_array_layer: 0,
            mip_level: 0,
            aspect_mask: vk::ImageAspectFlags::COLOR,
        };
        let region = vk::BufferImageCopy::builder()
            .buffer_image_height(0)
            .buffer_row_length(0)
            .buffer_offset(0)
            .image_subresource(image_subresource)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                depth: 1,
                height: img.desc.height,
                width: img.desc.width,
            })
            .build();

        let regions = &[region];

        unsafe {
            gfx.device.cmd_copy_buffer_to_image(
                vk_cmd,
                vk_buff.deref().borrow().buffer,
                img.internal.deref().borrow().img,
                dst_image_layout,
                regions,
            );
        }

        // change image layout
        self.transition_image_layout(
            gfx,
            img,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }

    pub fn copy_buffer(&mut self, gfx: &GFXDevice, buffer: &GPUBuffer, data: &[u8]) {
        let used_buffer = self.pick_stagging_buffer(buffer.desc.size, gfx);
        let src_buff = used_buffer.internal;
        let dst_buff = &buffer.internal;
        // copy data to staging buffer
        unsafe {
            let src = data.as_ptr().cast::<c_void>();
            let dst = src_buff
                .deref()
                .borrow_mut()
                .allocation
                .mapped_ptr()
                .unwrap();
            dst.as_ptr()
                .copy_from_nonoverlapping(src, std::mem::size_of_val(data));
        }

        let region = vk::BufferCopy {
            dst_offset: 0,
            size: buffer.desc.size as u64,
            src_offset: 0,
        };
        let regions = &[region];
        let src_buffer = src_buff.deref().borrow().buffer;
        let dst_buffer = dst_buff.deref().borrow().buffer;
        // copy from stagging buffer to GPU buffer
        unsafe {
            gfx.device
                .cmd_copy_buffer(self.cmd.cmd, src_buffer, dst_buffer, regions)
        }
    }

    pub fn flush(
        &mut self,
        gfx: &GFXDevice,
        wait_semaphore: vk::Semaphore,
        submits: &mut Vec<vk::SubmitInfo>,
        render_wait_semaphores: &mut Vec<vk::Semaphore>,
    ) -> bool {
        if self.used_buffers.is_empty() {
            return false;
        }

        self.copy_wait_semaphores.clear();
        self.copy_wait_semaphores.push(wait_semaphore);
        render_wait_semaphores.push(self.copy_signal_semaphores[0]);

        let copy_submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(self.copy_wait_semaphores.as_slice())
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::BOTTOM_OF_PIPE])
            .command_buffers(self.copy_cmds.as_slice())
            .signal_semaphores(self.copy_signal_semaphores.as_slice())
            .build();

        submits.push(copy_submit_info);

        unsafe {
            gfx.device.end_command_buffer(self.cmd.cmd).unwrap();
        }

        true
    }

    pub fn reset(&mut self, gfx: &GFXDevice) {
        //reset pools
        unsafe {
            gfx.device
                .reset_command_pool(self.cmd.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("reset cmd pool failed");
        }

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            gfx.device
                .begin_command_buffer(self.cmd.cmd, &begin_info)
                .unwrap()
        }

        self.free_buffers.extend(self.used_buffers.drain(..));
    }
}
