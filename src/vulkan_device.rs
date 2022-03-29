extern crate ash;
extern crate bitflags;
extern crate spirv_reflect;

use core::slice;
use std::{
    borrow::Cow,
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{hash_map::DefaultHasher, HashMap},
    ffi::{CStr, CString},
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    ops::Deref,
    rc::Rc,
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

pub use crate::vulkan_types::*;
use ash::vk::SwapchainKHR;

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

pub struct VKShader {
    pub(crate) device: ash::Device,
    pub module: vk::ShaderModule,
    pub inputs: Vec<spirv_reflect::types::ReflectInterfaceVariable>,
    // pub sets: Vec<spirv_reflect::types::ReflectDescriptorSet>,
    // pub bindings: Vec<spirv_reflect::types::ReflectDescriptorBinding>,
    pub entry_point_name: String,
    pub shader_stage: spirv_reflect::types::ReflectShaderStageFlags,
    pub desc_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}
impl Drop for VKShader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

impl std::hash::Hash for VKShader {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.module.hash(state);
    }
}

// #[derive(Clone)]
pub struct VKBuffer {
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub(crate) allocator: Alloc,
    pub(crate) device: ash::Device,
    pub buffer: ash::vk::Buffer,
    pub desc: GPUBufferDesc,
}

impl Drop for VKBuffer {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            (*self.allocator)
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct VKImage {
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub allocator: Alloc,
    pub device: ash::Device,
    pub img: vk::Image,
    pub format: vk::Format,
    pub view: vk::ImageView,
}

impl VKImage {
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

impl Drop for VKImage {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            (*self.allocator)
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();

            self.device.destroy_image_view(self.view, None);
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
            for bind in &desc_binder.binder {
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

            self.device.cmd_bind_descriptor_sets(
                cmd.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pso.pipeline_layout,
                0,
                sets.as_ref(),
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
    // pub fn bind_pipeline(&self, pipeline_state: &GPUPipelineState) {}
    pub fn create_pipeline_state(&self, desc: &PipelineStateDesc) -> PipelineState {
        let mut s = DefaultHasher::new();
        desc.hash(&mut s);
        let hash = s.finish();

        unsafe {
            let mut set_layouts = vec![];
            let mut push_constant_ranges = vec![];

            if let Some(ref vertex_shader) = desc.vertex {
                set_layouts.extend(vertex_shader.internal.desc_set_layouts.clone());
                push_constant_ranges.extend(vertex_shader.internal.push_constant_ranges.clone());
            };

            if let Some(ref fragment_shader) = desc.fragment {
                set_layouts.extend(fragment_shader.internal.desc_set_layouts.clone());
                push_constant_ranges.extend(fragment_shader.internal.push_constant_ranges.clone());
            };

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

            // build the layout from shaders , we can cache them in the future
            let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&set_layouts);

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap();

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

    pub fn bind_resource(&self, set: u32, binding: u32, buf: &GPUBuffer) {
        let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

        binder.binder.insert((set, binding), (*buf).clone());
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

                    let vshader = VKShader {
                        device: self.device.clone(),
                        module,
                        inputs: input_vars,
                        shader_stage,
                        entry_point_name,
                        desc_set_layouts,
                        // desc_sets,
                        push_constant_ranges,
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
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED);
            // .queue_family_indices(&[]);

            let img = self
                .device
                .create_image(&img_info, None)
                .expect("failed to create image");
            let requirements = self.device.get_image_memory_requirements(img);

            let mut allocation = (*self.allocator)
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
                desc: desc.clone(),

                internal: Rc::new(RefCell::new(VKImage {
                    allocation,
                    allocator: self.allocator.clone(),
                    device: self.device.clone(),
                    img,
                    format: img_info.format,
                    view,
                })),
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

            let mut allocation = (*self.allocator)
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
                internal: Rc::new(RefCell::new(VKBuffer {
                    allocation,
                    allocator: self.allocator.clone(),
                    buffer,
                    device: self.device.clone(),
                    desc: (*desc).clone(),
                })),
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

            let mut wait_semaphores = Vec::with_capacity(1);
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

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]) // TODO: is this ok for compute ?
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            let release_fence = self.release_fences[self.get_current_frame_index()];

            self.device
                .queue_submit(self.graphics_queue, &[submit_info.build()], release_fence)
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

            let possible_devices: Vec<(PhysicalDevice, usize)> = pdevices
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

                                if choose {
                                    Some((*pdevice, queue_family_index))
                                } else {
                                    None
                                }
                            });

                    device_match.next()
                })
                .collect();

            for x in &possible_devices {
                let props = instance.get_physical_device_properties(x.0);

                println!(
                    "device available {:?} , {:?}",
                    CStr::from_ptr(props.device_name.as_ptr()),
                    props.device_type
                );
            }

            let pdevice = *possible_devices
                .iter()
                .find(|d| {
                    let props = instance.get_physical_device_properties(d.0);
                    if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        return true;
                    };
                    false
                })
                .or_else(|| possible_devices.first())
                .unwrap();

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

    pub fn wait_for_gpu(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle error ");
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
            let graphics_queue_index = pdevice.1 as u32;
            let graphics_queue = device.get_device_queue(graphics_queue_index, 0);

            // swapchain
            let _size = window.inner_size();
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

            // let ci = vk::CommandPoolCreateInfo::builder()
            //     .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            //     .queue_family_index(pdevice.1 as u32);

            // let command_pool = device
            //     .create_command_pool(&ci, None)
            //     .expect("pool creation failed");

            // let ci = vk::CommandBufferAllocateInfo::builder()
            //     .command_buffer_count(swapchain.image_count)
            //     .command_pool(command_pool)
            //     .level(vk::CommandBufferLevel::PRIMARY);

            // let command_buffers = device.allocate_command_buffers(&ci).unwrap();

            // let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            // let present_complete_semaphore = device
            //     .create_semaphore(&semaphore_create_info, None)
            //     .unwrap();
            // let rendering_complete_semaphore = device
            //     .create_semaphore(&semaphore_create_info, None)
            //     .unwrap();

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice.0,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .expect("allocator creation failed");

            // let descriptor_pool = GFXDevice::init_descriptors(&device);

            let (command_buffers, descriptor_binders, release_fences) =
                GFXDevice::init_frames(&device, graphics_queue_index);

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
            }
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
    // set , bind
    pub binder: HashMap<(u32, u32), GPUBuffer>,
}

impl DescriptorBinder {
    pub fn new(device: &ash::Device) -> Self {
        let descriptor_pool = Self::init_descriptors(device);

        Self {
            device: device.clone(),
            binder: HashMap::new(),
            sets: Vec::with_capacity(16),
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

            self.binder.clear();
        }
    }
}
