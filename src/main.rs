extern crate ash;
extern crate winit;

extern crate base64;
extern crate gltf;

extern crate custom_error;
extern crate glam;

extern crate image;
extern crate indexmap;

use custom_error::custom_error;

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

use core::slice::{self};
use std::{ffi::CString, fs, mem, path::Path, rc::Rc};

use ash::vk::{self};
use device::{GFXDevice, GPUBuffer};
use std::time::Instant;
mod device;

use indexmap::IndexMap;

mod gpuStructs;
use gpuStructs::*;
enum UriType {
    URI,
    URIDATA,
}

struct DataUri<'a> {
    mime_type: &'a str,
    kind: UriType,
    data: &'a str,
}

fn split_once(input: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut iter = input.splitn(2, delimiter);
    Some((iter.next()?, iter.next()?))
}

impl<'a> DataUri<'a> {
    const VALID_MIME_TYPES: &'a [&'a str] =
        &["application/octet-stream", "application/gltf-buffer"];

    fn parse(uri: &'a str) -> DataUri<'a> {
        if let Some(uri) = uri.strip_prefix("data:") {
            let (mime_type, data) = split_once(uri, ',').unwrap();

            if let Some(mime_type) = mime_type.strip_suffix(";base64") {
                DataUri {
                    mime_type,
                    kind: UriType::URIDATA,
                    data,
                }
            } else {
                panic!("URI data needs to be base64 encoded :{}", uri);
            }
        } else {
            DataUri {
                mime_type: "",
                kind: UriType::URI,
                data: uri,
            }
        }
    }

    fn decode(&self, parent_path: &Path) -> Result<Vec<u8>, GltfError> {
        match self.kind {
            UriType::URI => {
                let data_path = std::path::Path::new(parent_path.parent().unwrap()).join(self.data);
                Ok(fs::read(data_path).unwrap_or_else(|_| {
                    panic!("couldn't open file:{}", parent_path.to_str().unwrap())
                }))
            }
            UriType::URIDATA => {
                if DataUri::VALID_MIME_TYPES.contains(&self.mime_type) {
                    match base64::decode(self.data) {
                        Ok(d) => Ok(d),
                        Err(err) => Err(GltfError::Base64Decode { base64error: err }),
                    }
                } else {
                    Err(GltfError::MimeErr {
                        mime_type: self.mime_type.into(),
                    })
                }
            }
        }
    }
}

custom_error! {GltfError
    Base64Decode{base64error:base64::DecodeError} = "failed to decode base64:{base64error}",
    MimeErr{mime_type:String}= "Mime type:'{mime_type}' is not supported",
    SizeMismatch = "buffer size doesn't match byteLength",
    MissingBlob = "Blob is missing from gltf!",

}

pub enum VertexAttributeValues {
    F32(Vec<f32>),
    F32x2(Vec<[f32; 2]>),
    F32x3(Vec<[f32; 3]>),
    F32x4(Vec<[f32; 4]>),
}

impl VertexAttributeValues {
    // get att size per vertex in bytes
    pub fn get_element_size(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x2(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x3(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x4(v) => VertexAttributeValues::size_of_vec_element(v),
        }
    }

    // get att size for all vertices in bytes
    pub fn get_size(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => mem::size_of_val(&v),
            VertexAttributeValues::F32x2(v) => mem::size_of_val(&v),
            VertexAttributeValues::F32x3(v) => mem::size_of_val(&v),
            VertexAttributeValues::F32x4(v) => mem::size_of_val(&v),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => v.len(),
            VertexAttributeValues::F32x2(v) => v.len(),
            VertexAttributeValues::F32x3(v) => v.len(),
            VertexAttributeValues::F32x4(v) => v.len(),
        }
    }

    pub fn get_bytes(&self) -> &[u8] {
        unsafe {
            match self {
                VertexAttributeValues::F32(v) => {
                    let n_bytes = v.len() * std::mem::size_of::<f32>();
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x2(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x3(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x4(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
            }
        }
    }

    fn size_of_vec_element<T>(_v: &Vec<T>) -> usize {
        mem::size_of::<T>()
    }
}

#[derive(Debug, Clone)]
pub enum Indices {
    None,
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

struct Renderable {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<image::DynamicImage>,
}

struct Mesh {
    pub index_buffer: Indices,
    pub vertex_attributes: IndexMap<&'static str, VertexAttributeValues>,
}

impl Mesh {
    pub const ATT_POSITION: &'static str = "vertex_pos";
    pub const ATT_UV: &'static str = "vertex_uv";
    pub const ATT_NORMAL: &'static str = "vertex_normal";
    pub const ATT_TANGENT: &'static str = "vertex_tangent";

    pub fn new() -> Self {
        Mesh {
            index_buffer: Indices::None,
            vertex_attributes: IndexMap::new(),
        }
    }

    pub fn set_attribute(&mut self, name: &'static str, val: VertexAttributeValues) {
        self.vertex_attributes.insert(name, val);
    }

    pub fn stride(&self) -> usize {
        let mut stride = 0usize;
        for att in self.vertex_attributes.values() {
            stride += att.get_element_size();
        }
        stride
    }

    pub fn vertex_count(&self) -> usize {
        let mut v_count: Option<usize> = None;

        for (name, att) in &self.vertex_attributes {
            let att_len = att.len();
            if let Some(prev_count) = v_count {
                assert_eq!(prev_count,att_len, "Attribute `{}` has a different vertex count than other attributes , expected:{} , got:{}" ,name ,prev_count ,att_len  );
            }
            v_count = Some(att_len);
        }

        v_count.unwrap_or(0)
    }

    pub fn get_buffer(&self) -> Vec<u8> {
        let vertex_size = self.stride();
        let vertex_count = self.vertex_count();

        let mut buff = vec![0; vertex_count * vertex_size];

        let mut att_offset = 0;

        for att in self.vertex_attributes.values() {
            let attributes_bytes = att.get_bytes();
            let att_size = att.get_element_size();

            for (vertex_index, att_data) in attributes_bytes.chunks_exact(att_size).enumerate() {
                let offset = vertex_index * vertex_size + att_offset;
                buff[offset..offset + att_size].copy_from_slice(att_data);
            }
            att_offset += att_size;
        }

        buff
    }
}

fn load_gltf() -> Renderable {
    //docs : https://www.khronos.org/files/gltf20-reference-guide.pdf
    let path = std::path::Path::new("./models/gltf_logo/scene.gltf");
    let gltf = gltf::Gltf::open(path)
        .unwrap_or_else(|_| panic!("couldn't open gltf file:{}", path.to_str().unwrap()));
    let buffers = load_buffers(&gltf, path).unwrap();

    let mut out_meshes: Vec<Mesh> = Vec::with_capacity(gltf.meshes().len());
    let mut out_textures = Vec::with_capacity(gltf.textures().len());
    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let mut out_mesh = Mesh::new();
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            if let Some(indices) = reader.read_indices().map(|v| match v {
                gltf::mesh::util::ReadIndices::U8(it) => Indices::U8(it.collect()),
                gltf::mesh::util::ReadIndices::U16(it) => Indices::U16(it.collect()),
                gltf::mesh::util::ReadIndices::U32(it) => Indices::U32(it.collect()),
            }) {
                out_mesh.index_buffer = indices;
            }

            if let Some(vertex_attribute) = reader
                .read_positions()
                .map(|v| VertexAttributeValues::F32x3(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_POSITION, vertex_attribute);
            }

            // TODO(ALI): support other uv types
            if let Some(vertex_attribute) = reader
                .read_tex_coords(0)
                .map(|v| VertexAttributeValues::F32x2(v.into_f32().collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_UV, vertex_attribute);
            }

            if let Some(vertex_attribute) = reader
                .read_normals()
                .map(|v| VertexAttributeValues::F32x3(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_NORMAL, vertex_attribute);
            }

            if let Some(vertex_attribute) = reader
                .read_tangents()
                .map(|v| VertexAttributeValues::F32x4(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_TANGENT, vertex_attribute);
            }

            // if let Some(vertex_attribute) = reader
            //     .read_colors()
            //     .map(|v| VertexAttributeValues::F32x3(v.collect()))
            // {
            //     out_mesh.set_attribute(Mesh::Att_Position, vertex_attribute);
            // }

            out_meshes.push(out_mesh);
        }
    }

    for texture in gltf.textures() {
        let tex = match texture.source().source() {
            gltf::image::Source::View { view, mime_type } => {
                let start = view.offset() as usize;
                let end = (view.offset() + view.length()) as usize;
                let buffer = &buffers[view.buffer().index()][start..end];

                image::load_from_memory_with_format(
                    buffer,
                    image::ImageFormat::from_extension(mime_type).unwrap_or_else(|| {
                        panic!("couldn't figure out extension for :{}", mime_type)
                    }),
                )
                .unwrap()
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                let uri = DataUri::parse(uri);

                let buf = uri.decode(path).unwrap();

                let mime = match mime_type {
                    Some(t) => t,
                    None => Path::new(uri.data).extension().unwrap().to_str().unwrap(),
                };

                image::load_from_memory_with_format(
                    &buf,
                    image::ImageFormat::from_extension(mime)
                        .unwrap_or_else(|| panic!("couldn't figure out extension for :{}", mime)),
                )
                .unwrap()
            }
        };

        println!(
            "images: {} , {}  ",
            texture.name().unwrap_or("no-name"),
            texture.index(),
            // texture.sampler()
        );

        out_textures.push(tex);
    }

    Renderable {
        meshes: out_meshes,
        textures: out_textures,
    }
}

fn load_buffers(gltf: &gltf::Gltf, path: &Path) -> Result<Vec<Vec<u8>>, GltfError> {
    // doc-> https://raw.githubusercontent.com/KhronosGroup/glTF/main/specification/2.0/figures/gltfOverview-2.0.0b.png

    let mut buffer_data: Vec<Vec<u8>> = Vec::new();

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => match gltf.blob.as_deref() {
                Some(blob) => buffer_data.push(blob.into()),
                None => return Err(GltfError::MissingBlob),
            },
            gltf::buffer::Source::Uri(uri) => {
                let uri = DataUri::parse(uri);

                let buf = match uri.decode(path) {
                    Ok(buff) => buff,
                    Err(err) => return Err(err),
                };

                if buffer.length() != buf.len() {
                    return Err(GltfError::SizeMismatch);
                }
                buffer_data.push(buf)
            }
        }
    }

    Ok(buffer_data)
}

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

fn create_graphics_pipeline<'a>(
    device: &'a Rc<GFXDevice>,
    renderpass: vk::RenderPass,
    renderable: &Renderable,
) -> (
    Vec<vk::Pipeline>,
    vk::PipelineLayout,
    device::GPUBuffer<'a>,
    device::GPUBuffer<'a>,
    device::GPUBuffer<'a>,
    vk::DescriptorSet,
) {
    unsafe {
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
        let mvp_buffer = device.create_buffer::<u8>(&desc, None);

        let mut vertex_spv_file = std::io::Cursor::new(&include_bytes!("../shaders/vert.spv")[..]);
        let mut frag_spv_file = std::io::Cursor::new(&include_bytes!("../shaders/frag.spv")[..]);

        let vertex_code = ash::util::read_spv(&mut vertex_spv_file)
            .expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

        let frag_code = ash::util::read_spv(&mut frag_spv_file)
            .expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

        let vertex_shader_module = device
            .device
            .create_shader_module(&vertex_shader_info, None)
            .expect("Vertex shader module error");

        let fragment_shader_module = device
            .device
            .create_shader_module(&frag_shader_info, None)
            .expect("Fragment shader module error");

        let mvp_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();

        let bindings = &[mvp_ubo_binding];

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        let desc_set_layout = device
            .device
            .create_descriptor_set_layout(&create_info, None)
            .expect("failed to create descriptor set layout");

        let set_layouts = &[desc_set_layout];

        let desc_set = build_descriptors(device, &desc_set_layout, &mvp_buffer);

        let push_constant_ranges = &[vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<MVP>() as u32)
            .build()];

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(push_constant_ranges)
            .set_layouts(set_layouts);

        let pipeline_layout = device
            .device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: mesh.stride() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let vertex_input_attribute_descriptions = [
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
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
            vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: device.swapchain.width as f32,
            height: device.swapchain.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: device.swapchain.width,
                height: device.swapchain.height,
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

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(renderpass);

        let graphics_pipelines = device
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_info.build()],
                None,
            )
            .expect("Unable to create graphics pipeline");

        (
            graphics_pipelines,
            pipeline_layout,
            index_buffer,
            vertex_buffer,
            mvp_buffer,
            desc_set,
        )
    }
}

fn build_descriptors(
    device: &Rc<GFXDevice>,
    desc_set_layout: &vk::DescriptorSetLayout,
    uniform_buffer: &GPUBuffer,
) -> vk::DescriptorSet {
    unsafe {
        //create uniform pool
        let uniform_pool_size = vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build();

        let pool_sizes = &[uniform_pool_size];

        let ci = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(3);

        let desc_pool = device
            .device
            .create_descriptor_pool(&ci, None)
            .expect("couldn't create descrriptor pool");

        //allocate desc sets

        let desc_set_layouts = &[*desc_set_layout];
        let ci = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(desc_set_layouts)
            .build();

        let desc_sets = device
            .device
            .allocate_descriptor_sets(&ci)
            .expect("failed to allocate descriptor sets");

        // update/define desc
        let desc_buffer = vk::DescriptorBufferInfo::builder()
            .range(vk::WHOLE_SIZE)
            .buffer(uniform_buffer.buffer)
            .offset(0)
            .build();
        //update desc set
        let wds = vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .dst_set(desc_sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .buffer_info(&[desc_buffer])
            .build();
        let desc_writes = &[wds];
        device.device.update_descriptor_sets(desc_writes, &[]);

        desc_sets[0]
    }
}

fn update_uniform_buffer(width: i32, height: i32, start: &Instant) -> MVP {
    let elapsed = { start.elapsed() };

    let view = glam::Mat4::look_at_lh(
        glam::vec3(0.0f32, 2.0, -7.0),
        glam::vec3(0.0f32, 0.0, 0.0),
        glam::vec3(0.0f32, 1.0f32, 0.0f32),
    );
    let proj = glam::Mat4::perspective_lh(
        f32::to_radians(45.0f32),
        width as f32 / height as f32,
        0.01f32,
        100.0f32,
    );

    //https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    let proj = proj.mul_mat4(&glam::mat4(
        glam::vec4(1.0f32, 0.0, 0.0, 0.0),
        glam::vec4(0.0f32, -1.0, 0.0, 0.0),
        glam::vec4(0.0f32, 0.0, 1.0f32, 0.0),
        glam::vec4(0.0f32, 0.0, 0.0f32, 1.0),
    ));

    let model = glam::Quat::from_axis_angle(
        glam::vec3(0.0f32, 1.0, 0.0),
        f32::to_radians(elapsed.as_millis() as f32) * 0.05f32,
    );
    let model = glam::Mat4::from_quat(model);

    MVP { proj, view, model }
}

fn main() {
    let renderable = load_gltf();

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

        let g = device.clone();

        let pipelines = { create_graphics_pipeline(&g, renderpass, &renderable) };
        let graphic_pipeline = pipelines.0[0];
        let graphic_pipeline_layout = pipelines.1;
        let index_buffer = pipelines.2;
        let vertex_buffer = pipelines.3.buffer;
        let mut uniform_buffer = pipelines.4;
        let desc_set = pipelines.5;
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: device.swapchain.width as f32,
            height: device.swapchain.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: device.swapchain.width,
                height: device.swapchain.height,
            },
        }];

        let info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let fence = device
            .device
            .create_fence(&info, None)
            .expect("failed to create fence");

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
                            device.cmd_bind_pipeline(
                                draw_command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                graphic_pipeline,
                            );
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

                            uniform_buffer
                                .allocation
                                .mapped_slice_mut()
                                .unwrap()
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
