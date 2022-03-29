extern crate bitflags;

use std::{cell::RefCell, mem::ManuallyDrop, rc::Rc};

use ash::vk::{self};
use bitflags::bitflags;
use gpu_allocator::vulkan::Allocator;

use crate::vulkan_device::{VKBuffer, VKImage, VKPipelineState, VKShader, VkSwapchain};

pub struct Renderpass {
    pub renderpass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub beginInfo: vk::RenderPassBeginInfo,
}

bitflags! {
   pub struct  GPUBufferUsage:u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = (0b1);
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST= 0b10;
        #[doc = "Can be used as TBO"]
        const UNIFORM_TEXEL_BUFFER= (0b100);
        #[doc = "Can be used as IBO"]
        const STORAGE_TEXEL_BUFFER= (0b1000);
        #[doc = "Can be used as UBO"]
        const UNIFORM_BUFFER= (0b1_0000);
        #[doc = "Can be used as SSBO"]
        const STORAGE_BUFFER= (0b10_0000);
        #[doc = "Can be used as source of fixed-function index fetch (index buffer)"]
        const INDEX_BUFFER= (0b100_0000);
        #[doc = "Can be used as source of fixed-function vertex fetch (VBO)"]
        const VERTEX_BUFFER= (0b1000_0000);
        #[doc = "Can be the source of indirect parameters (e.g. indirect buffer, parameter buffer)"]
        const INDIRECT_BUFFER= 0b1_0000_0000;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemLoc {
    /// The allocated resource is stored at an unknown memory location; let the driver decide what's the best location
    Unknown,
    /// Store the allocation in GPU only accessible memory - typically this is the faster GPU resource and this should be
    /// where most of the allocations live.
    GpuOnly,
    /// Memory useful for uploading data to the GPU and potentially for constant buffers
    CpuToGpu,
    /// Memory useful for CPU readback of data
    GpuToCpu,
}

#[derive(Clone)]
pub enum GPUIndexedBufferType {
    U32,
    U16,
    U8,
}

#[derive(Clone)]
pub enum GPUFormat {
    R8G8B8A8_UNORM,
}

bitflags! {
   pub struct  GPUImageUsage:u32{
       #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC= (0b1);
       #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST  = (0b10);
       #[doc = "Can be sampled from (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
       const SAMPLED = (0b100);
       #[doc = "Can be used as storage image (STORAGE_IMAGE descriptor type)"]
       const STORAGE= (0b1000);
       #[doc = "Can be used as framebuffer color attachment"]
       const COLOR_ATTACHMENT= (0b1_0000);
       #[doc = "Can be used as framebuffer depth/stencil attachment"]
       const DEPTH_STENCIL_ATTACHMENT= (0b10_0000);
       #[doc = "Image data not needed outside of rendering"]
       const TRANSIENT_ATTACHMENT= (0b100_0000);
       #[doc = "Can be used as framebuffer input attachment"]
       const INPUT_ATTACHMENT = (0b1000_0000);
   }

}

#[derive(Clone)]
pub struct GPUImageDesc {
    pub memory_location: MemLoc,
    pub usage: GPUImageUsage,
    pub format: GPUFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}
impl Default for GPUImageDesc {
    fn default() -> Self {
        GPUImageDesc {
            memory_location: MemLoc::Unknown,
            usage: GPUImageUsage::TRANSFER_SRC | GPUImageUsage::SAMPLED,
            format: GPUFormat::R8G8B8A8_UNORM,
            width: 0,
            height: 0,
            depth: 1,
        }
    }
}

#[derive(Clone)]
pub struct GPUBufferDesc {
    pub memory_location: MemLoc,
    pub size: u64,
    pub usage: GPUBufferUsage,
    pub index_buffer_type: Option<GPUIndexedBufferType>,
}
impl Default for GPUBufferDesc {
    fn default() -> Self {
        GPUBufferDesc {
            memory_location: MemLoc::Unknown,
            size: 0,
            usage: GPUBufferUsage::VERTEX_BUFFER,
            index_buffer_type: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct PipelineStateDesc {
    pub vertex: Option<Shader>,
    pub fragment: Option<Shader>,
    pub vertex_input_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub vertex_input_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    pub bind_point: vk::PipelineBindPoint,
}

impl std::hash::Hash for PipelineStateDesc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vertex.hash(state);
        self.fragment.hash(state);

        for v in &self.vertex_input_binding_descriptions {
            v.binding.hash(state);
            v.stride.hash(state);
            v.input_rate.hash(state);
        }

        for v in &self.vertex_input_attribute_descriptions {
            v.binding.hash(state);
            v.format.hash(state);
            v.location.hash(state);
            v.offset.hash(state);
        }

        self.bind_point.hash(state);
    }
}

#[derive(Clone)]
pub struct PipelineState {
    pub pipeline_desc: PipelineStateDesc,
    pub hash: u64,
    pub internal: Rc<RefCell<VKPipelineState>>,
}

pub struct CommandBuffer {
    pub cmd: vk::CommandBuffer,
    pub pipeline_state: Option<PipelineState>,
    pub pipeline_is_dirty: bool,
    pub prev_pipeline_hash: u64,
    pub pipeline: Option<vk::Pipeline>,
    pub command_pool: vk::CommandPool,
}

#[derive(Clone)]
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub framebuffer_count: u32,
    pub clearcolor: [f32; 4],
    pub vsync: bool,
}

impl Default for SwapchainDesc {
    fn default() -> Self {
        SwapchainDesc {
            width: 0,
            height: 0,
            framebuffer_count: 2,
            clearcolor: [1.0, 0.0, 1.0, 1.0],
            vsync: true,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Cmd(pub usize);

#[derive(Clone)]
pub struct GPUBuffer {
    pub internal: Rc<RefCell<VKBuffer>>,
}

#[derive(Clone)]
pub struct Swapchain {
    pub internal: Rc<RefCell<VkSwapchain>>,
}

#[derive(Clone, Hash)]
pub struct Shader {
    pub internal: Rc<VKShader>,
}

pub struct GPUImage {
    pub internal: Rc<RefCell<VKImage>>,
    pub desc: GPUImageDesc,
}
