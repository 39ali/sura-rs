extern crate bitflags;

use ash::vk::{self, Image, ImageView, SwapchainKHR};
use bitflags::bitflags;

use crate::device::Shader;

pub struct SwapchainData {
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain: SwapchainKHR,
    pub image_count: u32,
    pub present_images: Vec<Image>,
    pub present_image_views: Vec<ImageView>,
    pub width: u32,
    pub height: u32,
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
}

#[derive(Clone, Default)]
pub struct PipelineState {
    pub pipeline_info: vk::GraphicsPipelineCreateInfo,
    pub pipeline_desc: PipelineStateDesc,
}

pub struct CommandBuffer {
    pub cmd: vk::CommandBuffer,
    pub pipeline_state: PipelineState,
}
