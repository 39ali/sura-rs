extern crate bitflags;

use std::{cell::RefCell, rc::Rc};

use ash::vk::{self};
use bitflags::bitflags;

use super::device::{
    VKComputePipeline, VKRasterPipeline, VkBindGroup, VkRenderPass, VkSwapchain, VulkanBuffer,
    VulkanImage, VulkanSampler, VulkanShader,
};

bitflags! {
   pub struct  GPUBufferUsage:u32 {
        #[doc = "Can be used as a source of transfer operations"]
        const TRANSFER_SRC = (0b1);
        #[doc = "Can be used as a destination of transfer operations"]
        const TRANSFER_DST= 0b10;
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
        const SHADER_DEVICE_ADDRESS= 0b10_0000_0000_0000_0000;
        const ACCELERATION_STRUCTURE_STORAGE = 0b1_0000_0000_0000_0000_0000;
        const ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR  = (0b1000_0000_0000_0000_0000);
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

#[derive(Clone, Copy)]
pub enum GPUIndexedBufferType {
    U32,
    U16,
    U8,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum GPUFormat {
    R8G8B8A8_UNORM,
    R8G8B8A8_SRGB,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,

    //depth stencil
    D32_SFLOAT_S8_UINT,
    D24_UNORM_S8_UINT,
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

bitflags! {
   pub struct  ShaderStage:u32{
        const VERTEX= 0;
        const FRAGMENT = 0<<1;
        const COMPUTE = 0 <<2;
        const ALL = 0<<3;
   }

}

pub enum InputRate {
    Instance,
    Vertex,
}

pub struct VertexAttribute {
    pub location: u32,
    pub byte_offset: u32,
    pub format: GPUFormat,
}

pub struct VertexBufferLayout<'a> {
    pub stride: u64,
    pub input_rate: InputRate,
    pub attributes: &'a [VertexAttribute],
}

pub struct VertexState<'a> {
    pub shader: Shader,
    pub entry_point: String,
    pub vertex_buffer_layouts: &'a [VertexBufferLayout<'a>],
}

#[derive(Clone, Copy, Debug)]
pub enum BindingType {
    SAMPLER,
    IMAGE,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    InputAttachment,
}

pub struct BindGroupBinding {
    pub index: u32,
    pub stages: ShaderStage,
    pub ty: BindingType,
    pub count: u32,
    pub non_uniform_indexing: bool,
}

pub struct BindGroupLayout<'a> {
    pub bindings: &'a [BindGroupBinding],
}

pub struct PipelineLayout<'a> {
    pub bind_group_layouts: &'a [&'a BindGroupLayout<'a>],
}

#[derive(PartialEq, Eq)]
pub enum LoadOp {
    Clear,
    Load,
}
pub struct AttachmentOp {
    pub load: LoadOp,
    pub store: bool,
}

pub struct AttachmentLayout {
    pub format: GPUFormat,
    pub sample_count: u32,
    pub op: AttachmentOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

pub struct DepthStencilState {
    pub format: GPUFormat,
    pub sample_count: u32,
    pub op: AttachmentOp,
}

pub struct RasterPipelineDesc<'a> {
    pub vertex: Option<VertexState<'a>>,
    pub fragment: Option<Shader>,
    pub layout: PipelineLayout<'a>,
    pub attachments: Option<&'a [AttachmentLayout]>,
    pub depth_stencil: Option<DepthStencilState>,
}

pub struct TextureView<'a> {
    pub index: u32,
    pub texture: &'a GPUImage,
}

pub struct RenderAttachmentDesc<'a> {
    pub view: &'a TextureView<'a>,
    pub clear_color: &'a [f32; 4],
}

pub struct DepthStencilAttachmentDesc<'a> {
    pub view: &'a TextureView<'a>,
    pub clear_depth: f32,
    pub clear_stencil: u32,
}

pub struct RenderPassDesc<'a> {
    pub render_attachments: Option<&'a [RenderAttachmentDesc<'a>]>,
    pub depth_stencil_attachment: Option<&'a [DepthStencilAttachmentDesc<'a>]>,
}

pub struct ComputePipelineStateDesc {
    pub compute: Option<Shader>,
}

pub trait Pipeline {
    fn stage(&self) -> vk::ShaderStageFlags;
    fn layout(&self) -> vk::PipelineLayout;
    fn bind_point(&self) -> vk::PipelineBindPoint;
    fn vk_pipeline(&self) -> vk::Pipeline;
}

#[derive(Clone)]
pub struct RasterPipeline {
    pub internal: Rc<RefCell<VKRasterPipeline>>,
}

impl Pipeline for RasterPipeline {
    fn stage(&self) -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX
    }

    fn layout(&self) -> vk::PipelineLayout {
        self.internal.borrow().pipeline_layout
    }

    fn bind_point(&self) -> vk::PipelineBindPoint {
        vk::PipelineBindPoint::GRAPHICS
    }

    fn vk_pipeline(&self) -> vk::Pipeline {
        self.internal.borrow().pipeline
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    pub internal: Rc<RefCell<VKComputePipeline>>,
}
impl Pipeline for ComputePipeline {
    fn stage(&self) -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::COMPUTE
    }

    fn layout(&self) -> vk::PipelineLayout {
        self.internal.borrow().pipeline_layout
    }

    fn bind_point(&self) -> vk::PipelineBindPoint {
        vk::PipelineBindPoint::COMPUTE
    }

    fn vk_pipeline(&self) -> vk::Pipeline {
        self.internal.borrow().pipeline
    }
}

#[derive(Default)]
pub struct CommandBuffer {
    pub cmd: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
}

#[derive(Clone)]
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub clear_stencil: u32,
    pub vsync: bool,
    pub format: GPUFormat,
}

impl Default for SwapchainDesc {
    fn default() -> Self {
        SwapchainDesc {
            width: 0,
            height: 0,
            clear_color: [1.0, 0.0, 1.0, 1.0],
            clear_depth: 1.0,
            clear_stencil: 0,
            vsync: true,
            format: GPUFormat::B8G8R8A8_UNORM,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Cmd(pub usize);

#[derive(Clone)]
pub struct Swapchain {
    pub internal: Rc<RefCell<VkSwapchain>>,
    pub desc: SwapchainDesc,
}

#[derive(Clone)]
pub struct RenderPass {
    pub internal: Rc<RefCell<VkRenderPass>>,
}

#[derive(Clone, Hash)]
pub struct Shader {
    pub internal: Rc<VulkanShader>,
}

#[derive(Clone)]
pub struct GPUBufferDesc {
    pub memory_location: MemLoc,
    pub size: usize,
    pub usage: GPUBufferUsage,
    pub index_buffer_type: Option<GPUIndexedBufferType>,
    pub name: String,
}
impl Default for GPUBufferDesc {
    fn default() -> Self {
        GPUBufferDesc {
            memory_location: MemLoc::Unknown,
            size: 0,
            usage: GPUBufferUsage::VERTEX_BUFFER,
            index_buffer_type: None,
            name: String::default(),
        }
    }
}

#[derive(Clone)]
pub struct GPUBuffer {
    pub internal: Rc<RefCell<VulkanBuffer>>,
    pub desc: GPUBufferDesc,
}

impl PartialEq for GPUBuffer {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.internal, &other.internal)
    }
}

#[derive(Clone, Copy)]
pub struct GPUImageDesc {
    pub memory_location: MemLoc,
    pub usage: GPUImageUsage,
    pub format: GPUFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub size: usize,
}
impl Default for GPUImageDesc {
    fn default() -> Self {
        GPUImageDesc {
            memory_location: MemLoc::GpuOnly,
            usage: GPUImageUsage::TRANSFER_DST | GPUImageUsage::SAMPLED,
            format: GPUFormat::R8G8B8A8_UNORM,
            width: 0,
            height: 0,
            depth: 1,
            size: 0,
        }
    }
}
#[derive(Clone)]
pub struct GPUImage {
    pub internal: Rc<RefCell<VulkanImage>>,
    pub desc: GPUImageDesc,
}

impl PartialEq for GPUImage {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.internal, &other.internal)
    }
}

#[derive(Clone)]
pub struct Sampler {
    pub internal: Rc<RefCell<VulkanSampler>>,
}

impl PartialEq for Sampler {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.internal, &other.internal)
    }
}

#[derive(Clone)]
pub struct BindGroup {
    pub internal: Rc<RefCell<VkBindGroup>>,
}
