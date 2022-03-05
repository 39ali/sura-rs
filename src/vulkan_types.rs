use ash::vk::{self, SwapchainKHR};

use crate::device::{GPUBufferDesc, SwapchainDesc};

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
    pub(crate) allocator: crate::device::Alloc,
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
    pub allocator: crate::device::Alloc,
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
