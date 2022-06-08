use core::slice;
use std::{
    borrow::{Borrow, Cow},
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{hash_map::DefaultHasher, BTreeMap, HashMap},
    ffi::{c_void, CStr, CString},
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    ops::Deref,
    rc::Rc,
};

use ash::{
    vk::{
        self, Handle, PhysicalDevice, PhysicalDevicePortabilitySubsetFeaturesKHR,
        PhysicalDevicePortabilitySubsetFeaturesKHRBuilder, QueryPoolCreateInfo, QueryType,
        SwapchainKHR,
    },
    Entry, Instance,
};

use gpu_allocator::vulkan::*;
use log::{debug, error, info, trace, warn};
use spirv_cross::spirv::Resource;

use crate::math_utils;

pub use super::gpu_structs::*;

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
    ast: spirv_cross::spirv::Ast<spirv_cross::glsl::Target>,
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
    _depth_images: Vec<GPUImage>,
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

pub struct VkRenderPass {
    device: ash::Device,
    pub render_pass: vk::RenderPass,
}

impl Drop for VkRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None);
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

            // self.device.destroy_render_pass(self.renderpass, None);
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

pub struct VkQueryPool {
    pub device: ash::Device,
    pub query_pool: vk::QueryPool,
    pub current_query_indx: u32,
    query_count: u32, // per swapchain
}

impl Drop for VkQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}

pub struct GFXDevice {
    _entry: Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::extensions::khr::Surface,
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    pub pdevice: PhysicalDevice,
    pub device: ash::Device,
    device_properties: vk::PhysicalDeviceProperties,
    pub surface: vk::SurfaceKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,

    pub allocator: Alloc,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_index: u32,

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
    graphics_queue_properties: vk::QueueFamilyProperties,
}

impl GFXDevice {
    const VK_API_VERSIION: u32 = vk::make_api_version(0, 1, 2, 0);
    const FRAME_MAX_COUNT: usize = 2;
    const COMMAND_BUFFER_MAX_COUNT: usize = 8;

    // get times in us
    pub fn get_query_result(&self, query_pool: &mut VkQueryPool) -> Option<Vec<f64>> {
        let mut times: Vec<u64> = std::iter::repeat(0)
            .take(query_pool.current_query_indx as usize)
            .collect();

        let curent_swapchain_indx = self
            .current_swapchain
            .borrow()
            .as_ref()
            .unwrap()
            .internal
            .deref()
            .borrow()
            .image_index;

        let first_query = curent_swapchain_indx * query_pool.query_count;

        match unsafe {
            self.device.get_query_pool_results(
                query_pool.query_pool,
                first_query,
                query_pool.current_query_indx,
                times.as_mut_slice(),
                vk::QueryResultFlags::TYPE_64,
            )
        } {
            Ok(_) => {
                let mut times_f = Vec::<f64>::with_capacity(times.capacity());

                for t in &times {
                    let mut f = math_utils::bitfield_extract(
                        *t,
                        0,
                        self.graphics_queue_properties.timestamp_valid_bits,
                    ) as f64;

                    f *= self.device_properties.limits.timestamp_period as f64;

                    times_f.push(f);
                }

                Some(times_f)
            }
            Err(e) => match e {
                vk::Result::NOT_READY => None,
                _ => {
                    error!("failed to get_query_pool_results :{:?}", e);
                    None
                }
            },
        }
    }

    pub fn reset_query(&self, cmd: Cmd, query_pool: &mut VkQueryPool) {
        let cmd = self.get_cmd(cmd);

        let curent_swapchain_indx = match self.current_swapchain.borrow().as_ref() {
            Some(swapchain) => swapchain.internal.deref().borrow().image_index,
            None => 0,
        };

        let first_query = curent_swapchain_indx * query_pool.query_count;

        unsafe {
            self.device.cmd_reset_query_pool(
                cmd.cmd,
                query_pool.query_pool,
                first_query,
                query_pool.query_count,
            );
        }
        query_pool.current_query_indx = 0;
    }

    pub fn write_time_stamp(
        &self,
        cmd: Cmd,
        query_pool: &mut VkQueryPool,
        stages: vk::PipelineStageFlags,
    ) {
        let cmd = self.get_cmd(cmd);

        let curent_swapchain_indx = self
            .current_swapchain
            .borrow()
            .as_ref()
            .unwrap()
            .internal
            .deref()
            .borrow()
            .image_index;

        let query_index =
            curent_swapchain_indx * query_pool.query_count + query_pool.current_query_indx;

        unsafe {
            self.device
                .cmd_write_timestamp(cmd.cmd, stages, query_pool.query_pool, query_index);
        }
        query_pool.current_query_indx += 1;
    }
    pub fn create_query(&self, count: u32) -> VkQueryPool {
        let query_count = Self::COMMAND_BUFFER_MAX_COUNT as u32 * count;

        let info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(Self::FRAME_MAX_COUNT as u32 * query_count)
            .build();
        let query_pool = unsafe {
            self.device
                .create_query_pool(&info, None)
                .expect("failed to create query pool")
        };

        VkQueryPool {
            device: self.device.clone(),
            query_pool,
            current_query_indx: 0,
            query_count,
        }
    }

    pub fn get_buffer_address(&self, buffer: &GPUBuffer) -> u64 {
        let info =
            vk::BufferDeviceAddressInfo::builder().buffer(buffer.internal.deref().borrow().buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    pub fn get_buffer_data<T>(&self, buffer: &GPUBuffer, count: usize) -> &[T] {
        assert!(
            buffer.desc.memory_location == MemLoc::CpuToGpu,
            "only CpuToGpu is supported for now"
        );
        let ptr = (*buffer.internal)
            .borrow_mut()
            .allocation
            .mapped_slice_mut()
            .unwrap()
            .as_ptr();
        unsafe { slice::from_raw_parts(ptr as *const T, count) }
    }

    pub fn copy_to_buffer(&self, buffer: &GPUBuffer, dst_offset: u64, data: &[u8]) {
        match buffer.desc.memory_location {
            MemLoc::Unknown => todo!(),
            MemLoc::GpuOnly => {
                let region = vk::BufferCopy {
                    dst_offset,
                    size: data.len() as u64,
                    src_offset: 0,
                };

                self.copy_manager
                    .borrow_mut()
                    .copy_buffer(self, buffer, data, region);
            }
            MemLoc::CpuToGpu => {
                let start = dst_offset as usize;
                let end = start + data.len();
                (buffer.internal)
                    .borrow_mut()
                    .allocation
                    .mapped_slice_mut()
                    .unwrap()[start..end]
                    .copy_from_slice(data);
            }
            MemLoc::GpuToCpu => todo!(),
        }
    }

    pub fn bind_push_constants(&self, cmd: Cmd, pso: &PipelineState, data: &[u8]) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_push_constants(
                cmd.cmd,
                pso.internal.deref().borrow().pipeline_layout,
                vk::ShaderStageFlags::ALL,
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

    pub fn bind_index_buffer(
        &self,
        cmd: Cmd,
        index_buffer: &GPUBuffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        let cmd = self.get_cmd(cmd);

        unsafe {
            self.device.cmd_bind_index_buffer(
                cmd.cmd,
                index_buffer.internal.deref().borrow().buffer,
                offset,
                index_type,
            );
        }
    }

    fn build_graphics_pipeline(&self, cmd: &mut RefMut<CommandBuffer>) -> Option<vk::Pipeline> {
        let pipeline_state = cmd.pipeline_state.as_ref().unwrap();
        let pipeline_desc = &pipeline_state.pipeline_desc;
        let pipeline_state = &pipeline_state.internal.deref().borrow();

        let mut vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default();
        if let Some(vertex_input_attribute_descriptions) =
            &pipeline_desc.vertex_input_attribute_descriptions
        {
            let vertex_input_binding_descriptions = &pipeline_desc
                .vertex_input_binding_descriptions
                .as_ref()
                .unwrap();

            vertex_input_state_info.vertex_attribute_description_count =
                vertex_input_attribute_descriptions.len() as u32;

            vertex_input_state_info.p_vertex_attribute_descriptions =
                vertex_input_attribute_descriptions.as_ptr();
            vertex_input_state_info.vertex_binding_description_count =
                vertex_input_binding_descriptions.len() as u32;
            vertex_input_state_info.p_vertex_binding_descriptions =
                vertex_input_binding_descriptions.as_ptr();
        }

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(&pipeline_state.scissors)
            .viewports(&pipeline_state.viewports);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&pipeline_state.color_blend_attachment_states);

        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&pipeline_state.dynamic_state);

        let mut shader_stage_create_infos = vec![];
        //needs to live until `create_graphics_pipelines`
        let mut entry_points = vec![];

        if let Some(ref vertex_shader) = pipeline_desc.vertex {
            let vertex_shader = &*vertex_shader.internal;

            let shader_entry_name = vertex_shader.ast.get_entry_points().unwrap()[0]
                .name
                .clone();

            entry_points.push(CString::new(shader_entry_name).unwrap());

            shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                module: vertex_shader.module,
                p_name: entry_points.last().unwrap().as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            });
        };

        if let Some(ref fragment_shader) = pipeline_desc.fragment {
            let fragment_shader = &*fragment_shader.internal;

            let shader_entry_name = fragment_shader.ast.get_entry_points().unwrap()[0]
                .name
                .clone();

            entry_points.push(CString::new(shader_entry_name).unwrap());

            shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                module: fragment_shader.module,
                p_name: entry_points.last().unwrap().as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            });
        };

        // TODO: use pipeline cache

        if pipeline_desc.vertex.is_some() || pipeline_desc.fragment.is_some() {
            let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
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

            let pipeline = unsafe {
                self.device
                    .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                    .expect("Unable to create graphics pipeline")
            }[0];

            return Some(pipeline);
        }
        return None;
    }

    fn build_compute_pipeline(&self, cmd: &mut RefMut<CommandBuffer>) -> Option<vk::Pipeline> {
        let pipeline_state = cmd.pipeline_state.as_ref().unwrap();
        let pipeline_desc = &pipeline_state.pipeline_desc;
        let pipeline_state = &pipeline_state.internal.deref().borrow();

        if let Some(ref compute) = pipeline_desc.compute {
            let compute_shader = &*compute.internal;

            let shader_entry_name = compute_shader.ast.get_entry_points().unwrap()[0]
                .name
                .clone();

            //needs to live until `create_compute_pipelines`
            let entry_point = CString::new(shader_entry_name).unwrap();

            let shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
                module: compute_shader.module,
                p_name: entry_point.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };

            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(shader_stage_create_info)
                .layout(pipeline_state.pipeline_layout)
                .build();

            let pipeline = unsafe {
                self.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                    .expect("Unable to create graphics pipeline")
            }[0];

            return Some(pipeline);
        };

        None
    }

    // build pipeline if needed
    fn build_pipelines(&self, cmd: &mut RefMut<CommandBuffer>) {
        if !cmd.pipeline_is_dirty {
            return;
        }

        cmd.graphics_pipeline = self.build_graphics_pipeline(cmd);
        cmd.compute_pipeline = self.build_compute_pipeline(cmd);

        cmd.pipeline_is_dirty = false;
    }

    fn flush(&self, cmd: &mut RefMut<CommandBuffer>) {
        self.build_pipelines(cmd);
        let pso = cmd.pipeline_state.as_ref().unwrap();
        unsafe {
            if let Some(graphics_pipeline) = cmd.graphics_pipeline {
                self.device.cmd_bind_pipeline(
                    cmd.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );
            }

            if let Some(compute_pipeline) = cmd.compute_pipeline {
                self.device.cmd_bind_pipeline(
                    cmd.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    compute_pipeline,
                );
            }
        }

        self.bind_sets_internal(cmd);

        // let mut sets = vec![];
        // let desc_binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

        // let pso = pso.internal.deref().borrow();

        // for set_layout in &pso.set_layouts {
        //     let desc_set = desc_binder.get_descriptor_set(&self.device, set_layout);

        //     sets.push(desc_set);
        // }

        // {
        //     let mut desc_writes = Vec::with_capacity(
        //         desc_binder.binder_buff_update.len() + desc_binder.binder_img_update.len(),
        //     );

        //     // update desc
        //     // needs to live until we update desc set
        //     let mut desc_buffer_infos = Vec::with_capacity(desc_binder.binder_buff_update.len());
        //     let mut desc_img_infos = Vec::with_capacity(desc_binder.binder_img_update.len());

        //     for bind in &desc_binder.binder_buff_update {
        //         let dst_set = sets[bind.0 as usize];
        //         let dst_binding = bind.1;
        //         let dst_array_index = bind.2;
        //         let buffer = desc_binder.binder_buff.get(bind).unwrap();
        //         let buffer_vk = buffer.internal.deref().borrow().buffer;
        //         let mut desc_type = vk::DescriptorType::default();

        //         if buffer.desc.usage.contains(GPUBufferUsage::UNIFORM_BUFFER) {
        //             desc_type = vk::DescriptorType::UNIFORM_BUFFER;
        //         }
        //         if buffer.desc.usage.contains(GPUBufferUsage::STORAGE_BUFFER) {
        //             desc_type = vk::DescriptorType::STORAGE_BUFFER;
        //         }

        //         desc_buffer_infos.push(
        //             vk::DescriptorBufferInfo::builder()
        //                 .range(vk::WHOLE_SIZE)
        //                 .buffer(buffer_vk)
        //                 .offset(0)
        //                 .build(),
        //         );
        //         let buffer_info = &desc_buffer_infos.as_slice()
        //             [desc_buffer_infos.len() - 1..desc_buffer_infos.len()];
        //         //update desc set

        //         let wds = vk::WriteDescriptorSet::builder()
        //             .descriptor_type(desc_type)
        //             .dst_set(dst_set)
        //             .dst_binding(dst_binding)
        //             .dst_array_element(dst_array_index)
        //             .buffer_info(buffer_info)
        //             .build();

        //         desc_writes.push(wds);
        //     }

        //     for bind in &desc_binder.binder_img_update {
        //         let dst_set = sets[bind.0 as usize];
        //         let dst_binding = bind.1;
        //         let dst_array_index = bind.2;
        //         let binded_img = desc_binder.binder_img.get(bind).unwrap();
        //         let img_view_index = binded_img.0;
        //         let sampler = binded_img.1.internal.deref().borrow().sampler;
        //         let gpu_img = &binded_img.2;
        //         let vk_img = gpu_img.internal.deref().borrow();
        //         let img_view = vk_img.views.borrow()[img_view_index as usize];

        //         desc_img_infos.push(
        //             vk::DescriptorImageInfo::builder()
        //                 .image_view(img_view)
        //                 .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        //                 .sampler(sampler)
        //                 .build(),
        //         );
        //         let desc_img_info =
        //             &desc_img_infos.as_slice()[desc_img_infos.len() - 1..desc_img_infos.len()];

        //         //update desc set
        //         let wds = vk::WriteDescriptorSet::builder()
        //             .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        //             .dst_set(dst_set)
        //             .dst_binding(dst_binding)
        //             .dst_array_element(dst_array_index)
        //             .image_info(desc_img_info)
        //             .build();

        //         desc_writes.push(wds);
        //     }

        //     if !desc_writes.is_empty() {
        //         unsafe {
        //             self.device.update_descriptor_sets(&desc_writes, &[]);
        //         }
        //     }

        //     //

        //     if cmd.graphics_pipeline.is_some() {
        //         unsafe {
        //             self.device.cmd_bind_descriptor_sets(
        //                 cmd.cmd,
        //                 vk::PipelineBindPoint::GRAPHICS,
        //                 pso.pipeline_layout,
        //                 0,
        //                 &sets,
        //                 &[],
        //             );
        //         }
        //     }

        //     if cmd.compute_pipeline.is_some() {
        //         unsafe {
        //             self.device.cmd_bind_descriptor_sets(
        //                 cmd.cmd,
        //                 vk::PipelineBindPoint::COMPUTE,
        //                 pso.pipeline_layout,
        //                 0,
        //                 &sets,
        //                 &[],
        //             );
        //         }
        //     }
        // }

        // desc_binder.binder_buff_update.clear();
        // desc_binder.binder_img_update.clear();
    }

    pub fn draw(
        &self,
        cmd: Cmd,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            let mut cmd = self.get_cmd_mut(cmd);
            self.flush(&mut cmd);

            self.device.cmd_draw(
                cmd.cmd,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
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

    pub fn draw_indexed_indirect(
        &self,
        cmd: Cmd,
        buffer: &GPUBuffer,
        offset: u32,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            let mut cmd = self.get_cmd_mut(cmd);
            self.flush(&mut cmd);

            let buffer = buffer.internal.deref().borrow().buffer;
            self.device.cmd_draw_indexed_indirect(
                cmd.cmd,
                buffer,
                offset.into(),
                draw_count,
                stride,
            );
        }
    }

    pub fn disptach_compute(&self, cmd: Cmd, x: u32, y: u32, z: u32) {
        unsafe {
            let mut cmd = self.get_cmd_mut(cmd);
            self.flush(&mut cmd);

            self.device.cmd_dispatch(cmd.cmd, x, y, z);
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
        #[derive(Debug)]
        struct DescBind {
            set_binding: u32,
            desc_type: vk::DescriptorType,
            desc_count: u32,
        }
        #[derive(Debug)]
        struct DescSet {
            bindless: bool,
            bindings: Vec<DescBind>,
        }
        //NOTE : sets and push_constants are shared between stages
        let mut sets = BTreeMap::<u32, DescSet>::new();
        let mut push_constants_ranges = vec![];

        let mut push_binding = |shader: &Rc<VulkanShader>,
                                u: &Resource,
                                desc_type: vk::DescriptorType| {
            let set_id = shader
                .ast
                .get_decoration(u.id, spirv_cross::spirv::Decoration::DescriptorSet)
                .unwrap();

            let set_binding = shader
                .ast
                .get_decoration(u.id, spirv_cross::spirv::Decoration::Binding)
                .unwrap();

            let t = shader.ast.get_type(u.type_id).unwrap();
            let mut desc_count = match t {
                spirv_cross::spirv::Type::Boolean {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Char { array } => array[0],
                spirv_cross::spirv::Type::Int {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::UInt {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Int64 { vecsize, array } => array[0],
                spirv_cross::spirv::Type::UInt64 { vecsize, array } => array[0],
                spirv_cross::spirv::Type::AtomicCounter { array } => array[0],
                spirv_cross::spirv::Type::Half {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Float {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Double {
                    vecsize,
                    columns,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Struct {
                    member_types,
                    array,
                } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Image { array } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::SampledImage { array } => *array.first().unwrap_or(&1),
                spirv_cross::spirv::Type::Sampler { array } => *array.first().unwrap_or(&1),

                _ => todo!(),
            };

            // shader is using bindless ! ,if one of the bindings is bindless then we assume that the whole set is bindless
            let mut is_bindless = true;
            if desc_count == 0 {
                // TODO: maybe they're too big ?
                desc_count = DescriptorBinder::BINDLESS_DESCRIPTOR_MAX_COUNT;
                is_bindless = true;
            }

            let desc_bind = DescBind {
                set_binding,
                desc_type,
                desc_count,
            };
            if sets.contains_key(&set_id) {
                //check if bindings are the same if not panic
                let bindings = &mut sets.get_mut(&set_id).unwrap().bindings;
                if let Some(binding) = bindings
                    .iter_mut()
                    .find(|bind| bind.set_binding == desc_bind.set_binding)
                {
                    binding.desc_count = u32::max(binding.desc_count, desc_bind.desc_count);
                    assert!(
                        binding.desc_type == desc_bind.desc_type,
                        "set binding type needs to be the same accross all shader stages"
                    );
                } else {
                    bindings.push(desc_bind);
                }
            } else {
                sets.insert(
                    set_id,
                    DescSet {
                        bindless: is_bindless,
                        bindings: vec![desc_bind],
                    },
                );
            }
        };

        let mut merge_desc_set = |shader: &Rc<VulkanShader>| {
            let res = shader.ast.get_shader_resources().unwrap();
            // let stage = match shader.ast.get_entry_points().unwrap()[0].execution_model {
            //     spirv_cross::spirv::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            //     spirv_cross::spirv::ExecutionModel::TessellationControl => todo!(),
            //     spirv_cross::spirv::ExecutionModel::TessellationEvaluation => todo!(),
            //     spirv_cross::spirv::ExecutionModel::Geometry => todo!(),
            //     spirv_cross::spirv::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            //     spirv_cross::spirv::ExecutionModel::GlCompute => todo!(),
            //     spirv_cross::spirv::ExecutionModel::Kernel => todo!(),
            // };
            res.push_constant_buffers.iter().for_each(|u| {
                let offset = shader
                    .ast
                    .get_decoration(u.id, spirv_cross::spirv::Decoration::Offset)
                    .unwrap();

                let size = shader.ast.get_declared_struct_size(u.base_type_id).unwrap();

                if push_constants_ranges.len() < 1 {
                    push_constants_ranges.push(
                        vk::PushConstantRange::builder()
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .offset(offset)
                            .size(size)
                            .build(),
                    );
                }
            });

            res.uniform_buffers.iter().for_each(|u| {
                push_binding(&shader, u, vk::DescriptorType::UNIFORM_BUFFER);
            });

            res.storage_buffers.iter().for_each(|u| {
                push_binding(&shader, u, vk::DescriptorType::STORAGE_BUFFER);
            });

            res.sampled_images.iter().for_each(|u| {
                push_binding(&shader, u, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
            });
            res.separate_images.iter().for_each(|u| {
                push_binding(&shader, u, vk::DescriptorType::SAMPLED_IMAGE);
            });
            res.separate_samplers.iter().for_each(|u| {
                push_binding(&shader, u, vk::DescriptorType::SAMPLER);
            });
        };

        // vertex shader
        if desc.vertex.is_some() {
            merge_desc_set(&desc.vertex.as_ref().unwrap().internal);
        }

        // fragment shader
        if desc.fragment.is_some() {
            merge_desc_set(&desc.fragment.as_ref().unwrap().internal);
        }

        // compute shader
        if desc.compute.is_some() {
            // trace!("desc.compute");
            merge_desc_set(&desc.compute.as_ref().unwrap().internal);
        }

        let mut desc_set_layouts = Vec::with_capacity(sets.len());

        // trace!("sets");
        // sets.iter().for_each(|s| {
        //     trace!("set:{:?}", s);
        // });
        // panic!("");
        for set in &sets {
            let _set_index = set.0;
            let set = set.1;
            let mut set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding> =
                Vec::with_capacity(set.bindings.len());
            for binding in &set.bindings {
                if set_layout_bindings
                    .iter()
                    .find(|ss| {
                        ss.binding == binding.set_binding && ss.descriptor_type == binding.desc_type
                    })
                    .is_some()
                {
                    continue;
                }

                let binding_layout = vk::DescriptorSetLayoutBinding::builder()
                    .binding(binding.set_binding)
                    .descriptor_type(binding.desc_type)
                    .descriptor_count(binding.desc_count)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .build();

                set_layout_bindings.push(binding_layout);
            }

            unsafe {
                let mut set_layout_create_info =
                    vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_layout_bindings);

                let desc_set_layout = if set.bindless {
                    let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
                    let bindings_flags = vec![binding_flags; set_layout_bindings.len()];

                    let mut extra_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                        .binding_flags(&bindings_flags);
                    set_layout_create_info = set_layout_create_info
                        .push_next(&mut extra_info)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);

                    self.device
                        .create_descriptor_set_layout(&set_layout_create_info, None)
                        .expect("failed to create descriptor set layout")
                } else {
                    self.device
                        .create_descriptor_set_layout(&set_layout_create_info, None)
                        .expect("failed to create descriptor set layout")
                };

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

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::FRONT,
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

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

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
                renderpass: desc.renderpass,
                color_blend_attachment_states,
            })),
        }
    }

    pub fn create_set_bindless(&self, bindings: &[vk::DescriptorSetLayoutBinding]) -> DescSet {
        self.create_set_internal(bindings, true)
    }

    pub fn create_set(&self, bindings: &[vk::DescriptorSetLayoutBinding]) -> DescSet {
        self.create_set_internal(bindings, false)
    }
    fn create_set_internal(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
        bindless: bool,
    ) -> DescSet {
        const BINDLESS_DESCRIPTOR_MAX_COUNT: u32 = 512 * 1024;
        const BINDLESS_SAMPLERS_MAX_COUNT: u32 = 4;

        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        let storage_buffer_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .build();

        let sampled_image_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .build();

        let image_sampler_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(BINDLESS_SAMPLERS_MAX_COUNT)
            .ty(vk::DescriptorType::SAMPLER)
            .build();

        let pool_sizes = &[
            uniform_pool_size,
            image_sampler_pool_size,
            sampled_image_pool_size,
            storage_buffer_size,
        ];

        let mut ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(1);

        if bindless {
            ci = ci.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        }

        let pool = unsafe {
            self.device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descriptor pool")
        };

        let bindings_flags = vec![
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
            bindings.len()
        ];

        let mut extra_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&bindings_flags);
        let mut set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        if bindless {
            set_layout_create_info = set_layout_create_info
                .push_next(&mut extra_info)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        }

        let set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&set_layout_create_info, None)
                .expect("failed to create descriptor set layout")
        };

        let desc_set = unsafe {
            self.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(pool)
                        .set_layouts(&[set_layout])
                        .build(),
                )
                .expect("failed to allocate descriptor sets")[0]
        };

        DescSet {
            internal: Rc::new(RefCell::new(VkDescSet {
                writes: Vec::new(),
                buffer_infos: Vec::new(),
                img_infos: Vec::new(),
                device: self.device.clone(),
                set: desc_set,
                pool,
                set_layout,
            })),
        }
    }

    pub fn bind_set(&self, cmd: Cmd, set: &DescSet, set_index: u32) {
        let mut cmd = self.get_cmd_mut(cmd);
        cmd.sets.push((set_index, set.clone()));
    }
    fn bind_sets_internal(&self, cmd: &mut RefMut<CommandBuffer>) {
        let pipeline_layout = cmd
            .pipeline_state
            .as_ref()
            .unwrap()
            .internal
            .deref()
            .borrow()
            .pipeline_layout;

        // trace!("ypppppppppp {:?}", cmd.graphics_pipeline);

        for set in &cmd.sets {
            let set_index = set.0;
            let set = (set.1.internal).deref().borrow().set;

            if cmd.graphics_pipeline.is_some() {
                // trace!("ypppppppppp {:?}", set.set);
                unsafe {
                    self.device.cmd_bind_descriptor_sets(
                        cmd.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        set_index,
                        &[set],
                        &[],
                    );
                }
            }

            if cmd.compute_pipeline.is_some() {
                unimplemented!();
            }
        }

        cmd.sets.clear();
    }

    // pub fn bind_resource_buffer(&self, set: u32, binding: u32, array_index: u32, buf: &GPUBuffer) {
    //     let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

    //     let key = (set, binding, array_index);
    //     if let Some(binded_buff) = binder.binder_buff.get(&key) {
    //         if binded_buff != buf {
    //             binder.binder_buff.insert(key, (*buf).clone());
    //             binder.binder_buff_update.push(key);
    //         }
    //     } else {
    //         binder.binder_buff.insert(key, (*buf).clone());
    //         binder.binder_buff_update.push(key);
    //     };
    // }

    // pub fn bind_resource_img(
    //     &self,
    //     set: u32,
    //     binding: u32,
    //     array_index: u32,
    //     img: &GPUImage,
    //     view_index: u32,
    // ) {
    //     let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

    //     let key = (set, binding, array_index);

    //     // if let Some(binded_img) = binder.binder_img.get(&key) {
    //     //     if binded_img.0 != view_index || binded_img.1 != *sampler || binded_img.2 != *img {
    //     //         binder
    //     //             .binder_img
    //     //             .insert(key, (view_index, sampler.clone(), (*img).clone()));
    //     //         binder.binder_img_update.push(key);
    //     //     }
    //     // } else {
    //     //     binder
    //     //         .binder_img
    //     //         .insert(key, (view_index, sampler.clone(), (*img).clone()));
    //     //     binder.binder_img_update.push(key);
    //     // };
    // }

    // pub fn bind_resource_sampler(&self, set: u32, binding: u32, array_index: u32) {
    //     // let binder = &mut self.descriptor_binders.borrow_mut()[self.get_current_frame_index()];

    //     // let key = (set, binding, array_index);

    //     // if let Some(binded_img) = binder.binder_img.get(&key) {
    //     //     if binded_img.0 != view_index || binded_img.1 != *sampler || binded_img.2 != *img {
    //     //         binder
    //     //             .binder_img
    //     //             .insert(key, (view_index, sampler.clone(), (*img).clone()));
    //     //         binder.binder_img_update.push(key);
    //     //     }
    //     // } else {
    //     //     binder
    //     //         .binder_img
    //     //         .insert(key, (view_index, sampler.clone(), (*img).clone()));
    //     //     binder.binder_img_update.push(key);
    //     // };
    // }

    // pub fn bind_resource_imgs(
    //     &self,
    //     set: u32,
    //     binding: u32,
    //     array_indices: &[u32],
    //     imgs: &[GPUImage],
    //     view_indices: &[u32],
    //     samplers: &[Sampler],
    // ) {
    //     assert!(imgs.len() == view_indices.len() && imgs.len() == samplers.len());

    //     for i in 0..imgs.len() {
    //         self.bind_resource_img(
    //             set,
    //             binding,
    //             array_indices[i],
    //             &imgs[i],
    //             view_indices[i],
    //             &samplers[i],
    //         );
    //     }
    // }

    pub fn create_shader(&self, byte_code: &[u8]) -> Shader {
        let words_from_bytes = |buf: &[u8]| -> &[u32] {
            unsafe {
                std::slice::from_raw_parts(
                    buf.as_ptr() as *const u32,
                    buf.len() / std::mem::size_of::<u32>(),
                )
            }
        };

        let code_words = words_from_bytes(byte_code);
        let module = spirv::Module::from_words(code_words);

        use spirv_cross::{glsl, spirv};
        let ast = spirv::Ast::<glsl::Target>::parse(&module).unwrap();

        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&code_words);

        let module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .expect("Shader module error")
        };

        let vshader = VulkanShader {
            device: self.device.clone(),
            module,
            ast,
        };
        Shader {
            internal: Rc::new(vshader),
        }
    }
    pub fn create_image(&self, desc: &GPUImageDesc, data: Option<&[u8]>) -> GPUImage {
        unsafe {
            let img_foramt = match desc.format {
                GPUFormat::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_UNORM,
                GPUFormat::D32_SFLOAT_S8_UINT => vk::Format::D32_SFLOAT_S8_UINT,
                GPUFormat::D24_UNORM_S8_UINT => vk::Format::D24_UNORM_S8_UINT,
            };

            let img_usage = {
                let mut flag = vk::ImageUsageFlags::default();
                if desc.usage.contains(GPUImageUsage::TRANSFER_SRC) {
                    flag |= vk::ImageUsageFlags::TRANSFER_SRC;
                }
                if desc.usage.contains(GPUImageUsage::TRANSFER_DST) {
                    flag |= vk::ImageUsageFlags::TRANSFER_DST;
                }

                if desc.usage.contains(GPUImageUsage::SAMPLED) {
                    flag |= vk::ImageUsageFlags::SAMPLED;
                }

                if desc.usage.contains(GPUImageUsage::STORAGE) {
                    flag |= vk::ImageUsageFlags::STORAGE;
                }

                if desc.usage.contains(GPUImageUsage::COLOR_ATTACHMENT) {
                    flag |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
                }

                if desc.usage.contains(GPUImageUsage::DEPTH_STENCIL_ATTACHMENT) {
                    flag |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
                }
                flag
            };

            let img_info = vk::ImageCreateInfo::builder()
                .array_layers(1)
                .mip_levels(1)
                .extent(vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: desc.depth,
                })
                .format(img_foramt)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(img_usage);
            // .queue_family_indices(&[]);

            let img = self
                .device
                .create_image(&img_info, None)
                .expect("failed to create image");
            let requirements = self.device.get_image_memory_requirements(img);

            let allocation = self
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

            if desc.usage.contains(GPUBufferUsage::SHADER_DEVICE_ADDRESS) {
                usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            }

            if desc.usage.contains(GPUBufferUsage::INDIRECT_BUFFER) {
                usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }
            // if desc.usage.contains(GPUBufferUsage::STORAGE_TEXEL_BUFFER) {
            //     usage |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
            // }

            // if desc.usage.contains(GPUBufferUsage::UNIFORM_TEXEL_BUFFER) {
            //     usage |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
            // }
            info.usage = usage;

            let buffer = self.device.create_buffer(&info, None).unwrap();
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let allocation = (*self.allocator)
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Buffer allocation",
                    requirements,
                    location,
                    linear: true,
                })
                .unwrap();

            let gpu_buffer = GPUBuffer {
                internal: Rc::new(RefCell::new(VulkanBuffer {
                    allocation: ManuallyDrop::new(allocation),
                    allocator: self.allocator.clone(),
                    buffer,
                    device: self.device.clone(),
                })),
                desc: desc.clone(),
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
                    let region = vk::BufferCopy {
                        dst_offset: 0,
                        size: content.len() as u64,
                        src_offset: 0,
                    };

                    self.copy_manager.deref().borrow_mut().copy_buffer(
                        self,
                        &gpu_buffer,
                        content,
                        region,
                    )
                } else {
                    let buff_ptr = gpu_buffer
                        .internal
                        .deref()
                        .borrow_mut()
                        .allocation
                        .mapped_ptr()
                        .unwrap();

                    buff_ptr
                        .as_ptr()
                        .cast::<u8>()
                        .copy_from_nonoverlapping(content.as_ptr(), content.len());
                }
            }

            let buffer_name = CString::new(desc.name.clone()).unwrap();
            let buffer_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(buffer.as_raw())
                .object_name(&buffer_name)
                .object_type(vk::ObjectType::BUFFER);
            self.debug_utils_loader
                .debug_utils_set_object_name(self.device.handle(), &buffer_name_info)
                .expect("object name setting failed");

            gpu_buffer
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

    pub fn get_current_vulkan_cmd(&self) -> vk::CommandBuffer {
        let cmd_index = self.current_command.get() - 1;

        let cmd = &self.command_buffers.borrow_mut()[self.get_current_frame_index()][cmd_index];
        cmd.cmd
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
    pub fn bind_swapchain(&self, cmd: Cmd, swapchain: &Swapchain) {
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
        }
    }
    pub fn begin_renderpass_sc(&self, cmd: Cmd, swapchain: &Swapchain) {
        let cmd = self.get_cmd(cmd);

        let internal = (*swapchain.internal).borrow_mut();

        unsafe {
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
                .render_pass(internal.renderpass.clone())
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

    pub fn begin_renderpass(&self, cmd: Cmd, render_pass: &RenderPass) {
        let cmd = self.get_cmd(cmd);

        let render_pass = render_pass.internal.deref().borrow();

        let swapchain = self
            .current_swapchain
            .borrow()
            .as_ref()
            .unwrap()
            .internal
            .clone();

        let swapchain = swapchain.deref().borrow();

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: swapchain.desc.clearcolor,
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
            .render_pass(render_pass.render_pass)
            .framebuffer(swapchain.framebuffers[swapchain.image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.desc.width,
                    height: swapchain.desc.height,
                },
            })
            .clear_values(&clear_values);
        unsafe {
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
                if Self::FRAME_MAX_COUNT as u32 <= self.surface_capabilities.max_image_count {
                    Self::FRAME_MAX_COUNT as u32
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

            //depth images

            let depth_images: Vec<GPUImage> = (0..present_images.len())
                .map(|_| {
                    let img_desc = GPUImageDesc {
                        format: GPUFormat::D32_SFLOAT_S8_UINT,
                        width: desc.width,
                        height: desc.height,
                        memory_location: MemLoc::GpuOnly,
                        usage: GPUImageUsage::DEPTH_STENCIL_ATTACHMENT,
                        size: (desc.width * desc.height * 40) as usize,
                        ..Default::default()
                    };
                    self.create_image(&img_desc, None)
                })
                .collect();

            let depth_image_views: Vec<vk::ImageView> = depth_images
                .iter()
                .map(|depth_image| {
                    let view_index = self.create_image_view(
                        depth_image,
                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                        1,
                        1,
                    ) as usize;
                    let img_view = depth_image.internal.deref().borrow().views.borrow()[view_index];
                    img_view
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
                    vk::AttachmentDescription {
                        format: vk::Format::D32_SFLOAT_S8_UINT,
                        samples: vk::SampleCountFlags::TYPE_1,
                        load_op: vk::AttachmentLoadOp::CLEAR,
                        store_op: vk::AttachmentStoreOp::DONT_CARE,
                        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        ..Default::default()
                    },
                ];

                let color_ref = [vk::AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }];

                let depth_ref = vk::AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                };

                let subpasses = [vk::SubpassDescription::builder()
                    .color_attachments(&color_ref)
                    .depth_stencil_attachment(&depth_ref)
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .build()];

                let dependencies = [vk::SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
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

            let framebuffers: Vec<vk::Framebuffer> = (0..present_image_views.len())
                .map(|i| {
                    let framebuffer_attachments = [present_image_views[i], depth_image_views[i]];
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
                    _depth_images: depth_images,
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

    pub fn create_imgui_render_pass(&self) -> RenderPass {
        let surface_format = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface)
                .unwrap()[0]
        };

        let render_pass = {
            let attachments = [
                vk::AttachmentDescription {
                    format: surface_format.format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::LOAD,
                    store_op: vk::AttachmentStoreOp::STORE,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    initial_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                vk::AttachmentDescription {
                    format: vk::Format::D32_SFLOAT_S8_UINT,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::LOAD,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];

            let color_ref = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];

            let depth_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpasses = [vk::SubpassDescription::builder()
                .color_attachments(&color_ref)
                .depth_stencil_attachment(&depth_ref)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()];

            let dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                ..Default::default()
            }];
            let renderpass_ci = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);
            unsafe {
                self.device
                    .create_render_pass(&renderpass_ci, None)
                    .expect("failed to create a renderpass")
            }
        };

        RenderPass {
            internal: Rc::new(RefCell::new(VkRenderPass {
                render_pass,
                device: self.device.clone(),
            })),
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

                for _i in 0..GFXDevice::COMMAND_BUFFER_MAX_COUNT {
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
                        graphics_pipeline: None,
                        compute_pipeline: None,
                        prev_pipeline_hash: 0,
                        sets: Vec::new(),
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
            let mut device_extension_names_raw =
                vec![ash::extensions::khr::Swapchain::name().as_ptr()];

            let device_extentions = instance
                .enumerate_device_extension_properties(pdevice)
                .unwrap();

            let is_vk_khr_portability_subset = device_extentions.iter().any(|ext| -> bool {
                let e = CStr::from_ptr(ext.extension_name.as_ptr());
                if e.eq(vk::KhrPortabilitySubsetFn::name()) {
                    device_extension_names_raw.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
                    return true;
                }

                false
            });

            let features11 = &mut vk::PhysicalDeviceVulkan11Features::default();
            let features12 = &mut vk::PhysicalDeviceVulkan12Features::default();
            let features2 = &mut vk::PhysicalDeviceFeatures2::builder()
                .push_next(features11)
                .push_next(features12)
                .build();
            instance.get_physical_device_features2(pdevice, features2);

            if features11.shader_draw_parameters == 0 {
                error!("shader_draw_parameters is not supported! ")
            }
            if features12.buffer_device_address == 0 {
                error!("buffer_device_address is not supported! ")
            }

            if features12.descriptor_indexing == 0 {
                error!("descriptor_indexing is not supported! ")
            }

            if features12.descriptor_binding_partially_bound == 0 {
                error!("descriptor_binding_partially_bound is not supported! ")
            }

            if features12.descriptor_binding_variable_descriptor_count == 0 {
                error!("descriptor_binding_variable_descriptor_count is not supported! ")
            }

            if features12.scalar_block_layout == 0 {
                error!("scalar_block_layout is not supported! ")
            }

            // info!("device features :{:?} ", features2);

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut ci = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(features2);

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
            let entry = { Entry::load().expect("failed to load vulkan dll") };

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
                .api_version(Self::VK_API_VERSIION);

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
            let graphics_queue_index = pdevice.1 as u32;
            let pdevice = pdevice.0;
            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();

            let device = GFXDevice::create_device(&instance, pdevice, graphics_queue_index);

            let graphics_queue = device.get_device_queue(graphics_queue_index, 0);
            let graphics_queue_properties = instance
                .get_physical_device_queue_family_properties(pdevice)
                [graphics_queue_index as usize];
            // swapchain
            let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
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
                pdevice,
                device_properties,
                device,
                swapchain_loader,
                allocator: Rc::new(RefCell::new(ManuallyDrop::new(allocator))),
                graphics_queue,
                graphics_queue_index,
                graphics_queue_properties,
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
            debug!("Destroying vulkan device");
            self.device.device_wait_idle().unwrap();

            // drop swapchain
            drop(self.current_swapchain.replace(None));

            // drop commad buffers
            {
                let mut command_buffers = self.command_buffers.borrow_mut();
                for frame_cmds in command_buffers.iter() {
                    for cmd in frame_cmds {
                        if let Some(graphics_pipeline) = cmd.graphics_pipeline {
                            self.device.destroy_pipeline(graphics_pipeline, None);
                        };
                        if let Some(compute_pipeline) = cmd.compute_pipeline {
                            self.device.destroy_pipeline(compute_pipeline, None);
                        };
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
    // set , bind , array_index
    pub binder_buff: HashMap<(u32, u32, u32), GPUBuffer>,
    // set , bind ,array_index  , view_index , sampler ,gpuimage
    pub binder_img: HashMap<(u32, u32, u32), (u32, Option<Sampler>, GPUImage)>,

    // this is so we can tell if a set needs updating
    // set , bind , array_index
    pub binder_buff_update: Vec<(u32, u32, u32)>,
    pub binder_img_update: Vec<(u32, u32, u32)>,
}

impl DescriptorBinder {
    const BINDLESS_DESCRIPTOR_MAX_COUNT: u32 = 512 * 1024;

    pub fn new(device: &ash::Device) -> Self {
        let descriptor_pool = Self::init_descriptors(device);

        Self {
            device: device.clone(),
            binder_buff: HashMap::new(),
            binder_img: HashMap::new(),
            binder_buff_update: Vec::new(),
            binder_img_update: Vec::new(),
            sets: Vec::with_capacity(16),
            descriptor_pool,
        }
    }

    fn init_descriptors(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(Self::BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        let image_sampler_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(Self::BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::SAMPLER)
            .build();

        let sampled_image_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(Self::BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .build();

        let storage_buffer_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(Self::BINDLESS_DESCRIPTOR_MAX_COUNT)
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .build();

        let pool_sizes = &[
            uniform_pool_size,
            image_sampler_pool_size,
            sampled_image_pool_size,
            storage_buffer_size,
        ];

        let ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(10);

        unsafe {
            device
                .create_descriptor_pool(&ci, None)
                .expect("couldn't create descriptor pool")
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

struct CopyManager {
    free_buffers: Vec<GPUBuffer>,
    used_buffers: Vec<GPUBuffer>,
    //needs to be dropped
    cmd: CommandBuffer,
    copy_wait_semaphores: Vec<vk::Semaphore>,
    //needs to be dropped
    copy_signal_semaphore: vk::Semaphore,
    device: ash::Device,
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
                graphics_pipeline: None,
                compute_pipeline: None,
                sets: Vec::new(),
            }
        };

        let free_buffers = Vec::with_capacity(Self::BUFFERS_COUNT);
        let used_buffers = Vec::with_capacity(Self::BUFFERS_COUNT);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let copy_semaphore = unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        let copy_signal_semaphore = copy_semaphore;

        let copy_wait_semaphores = Vec::<vk::Semaphore>::new();

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd.cmd, &begin_info).unwrap() }

        CopyManager {
            free_buffers,
            used_buffers,
            cmd,
            copy_signal_semaphore,
            copy_wait_semaphores,
            device: device.clone(),
        }
    }

    fn pick_stagging_buffer(&mut self, size: usize, gfx: &GFXDevice) -> GPUBuffer {
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
                    name: format!("stagging-buffer({})", self.used_buffers.len()),
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

    pub fn copy_buffer(
        &mut self,
        gfx: &GFXDevice,
        buffer: &GPUBuffer,
        data: &[u8],
        region: vk::BufferCopy,
    ) {
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
        render_wait_semaphores.push(self.copy_signal_semaphore);

        let cpy_cmds = slice::from_ref(&self.cmd.cmd);
        let signal_semaphores = slice::from_ref(&self.copy_signal_semaphore);

        let copy_submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(self.copy_wait_semaphores.as_slice())
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::BOTTOM_OF_PIPE])
            .command_buffers(cpy_cmds)
            .signal_semaphores(signal_semaphores)
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

impl Drop for CopyManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.cmd.command_pool, None);
            self.device
                .destroy_semaphore(self.copy_signal_semaphore, None);
        }
    }
}

#[derive(Clone)]

pub struct VkDescSet {
    set: vk::DescriptorSet,
    device: ash::Device,
    writes: Vec<vk::WriteDescriptorSet>,
    // we need these to live until we write the writes
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    img_infos: Vec<vk::DescriptorImageInfo>,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
}

impl DescSet {
    pub fn bind_resource_imgs(
        &mut self,
        binding: u32,
        array_indices: &[u32],
        imgs: &[GPUImage],
        view_indices: &[u32],
    ) {
        let mut desc_set = self.internal.borrow_mut();

        for i in 0..imgs.len() {
            let img_vk = imgs[i].internal.deref().borrow();
            let img_view_index = view_indices[i];
            let img_view_vk = img_vk.views.borrow()[img_view_index as usize];
            let arr_index = array_indices[i];

            desc_set.img_infos.push(
                vk::DescriptorImageInfo::builder()
                    .image_view(img_view_vk)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build(),
            );
            let desc_img_info = &desc_set.img_infos.as_slice()
                [desc_set.img_infos.len() - 1..desc_set.img_infos.len()];

            //update desc set
            let wds = vk::WriteDescriptorSet::builder()
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .dst_set(desc_set.set)
                .dst_binding(binding)
                .dst_array_element(arr_index)
                .image_info(desc_img_info)
                .build();

            desc_set.writes.push(wds);
        }
    }

    pub fn bind_resource_sampler(&mut self, binding: u32, array_index: u32, sampler: &Sampler) {
        let mut desc_set = self.internal.borrow_mut();
        let sampler = sampler.internal.deref().borrow().sampler;

        desc_set
            .img_infos
            .push(vk::DescriptorImageInfo::builder().sampler(sampler).build());
        let desc_img_info =
            &desc_set.img_infos.as_slice()[desc_set.img_infos.len() - 1..desc_set.img_infos.len()];

        //update desc set
        let wds = vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_set(desc_set.set)
            .dst_binding(binding)
            .dst_array_element(array_index)
            .image_info(desc_img_info)
            .build();

        desc_set.writes.push(wds);
    }

    pub fn bind_resource_buffer(&mut self, binding: u32, array_index: u32, buffer: &GPUBuffer) {
        let mut desc_set = self.internal.borrow_mut();
        let desc_type = {
            if buffer.desc.usage.contains(GPUBufferUsage::UNIFORM_BUFFER) {
                vk::DescriptorType::UNIFORM_BUFFER
            } else if buffer.desc.usage.contains(GPUBufferUsage::STORAGE_BUFFER) {
                vk::DescriptorType::STORAGE_BUFFER
            } else {
                unimplemented!();
            }
        };

        let buffer_vk = buffer.internal.deref().borrow().buffer;

        desc_set.buffer_infos.push(
            vk::DescriptorBufferInfo::builder()
                .range(vk::WHOLE_SIZE)
                .buffer(buffer_vk)
                .offset(0)
                .build(),
        );

        let buffer_info = &desc_set.buffer_infos.as_slice()
            [desc_set.buffer_infos.len() - 1..desc_set.buffer_infos.len()];

        //update desc set

        let ws = vk::WriteDescriptorSet::builder()
            .descriptor_type(desc_type)
            .dst_set(desc_set.set)
            .dst_binding(binding)
            .dst_array_element(array_index)
            .buffer_info(buffer_info)
            .build();
        desc_set.writes.push(ws);
    }

    pub fn flush(&mut self) {
        let mut desc_set = self.internal.borrow_mut();
        if !desc_set.writes.is_empty() {
            unsafe {
                desc_set
                    .device
                    .update_descriptor_sets(&desc_set.writes, &[]);
            }
            desc_set.writes.clear();
            desc_set.buffer_infos.clear();
            desc_set.img_infos.clear();
        }
    }
}

impl Drop for VkDescSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
