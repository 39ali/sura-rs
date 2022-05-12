use std::{
    cell::RefCell,
    fs::File,
    mem::{self, ManuallyDrop},
    path::Path,
    slice,
    time::Instant,
};

//TODO : remove
use sura_asset::mesh::{self, *};
use sura_backend::vulkan::vulkan_device::*;
//TODO : remove
use image::GenericImageView;
use sura_backend::ash::vk;
//TODO : remove
use log::{trace, warn};
use winit::window::Window;

use crate::buffer::{BufferBuilder, BufferData};

const albedo_index: usize = 9;
#[derive(Default, Debug)]
#[allow(dead_code)]
struct MVP {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

struct GpuMesh {
    pos_offset: u32,
    uv_offset: u32,
}

pub struct InnerData {
    pub swapchain: Swapchain,

    mvp_buffer: GPUBuffer,
    pso: PipelineState,
    mesh: LoadedTriangleMesh,
    images: Vec<GPUImage>,
    base_texture_view_index: u32,
    sampler: Sampler,
    vertex_buffer: GPUBuffer,
    index_buffer: GPUBuffer,
    draw_cmds_buffer: GPUBuffer,
    vertices_buffer: GPUBuffer,
    gpu_mesh_buffer: GPUBuffer,
    uv_buffer: GPUBuffer,
}

fn load_triangled_mesh(path: &Path) -> LoadedTriangleMesh {
    let file = File::open(&path).unwrap_or_else(|e| panic!("Could not mmap {:?}: {:?}", path, e));

    let mmap = ManuallyDrop::new(unsafe { memmap2::MmapOptions::new().map(&file).unwrap() });

    let data = unsafe { slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

    let archived = unsafe { rkyv::archived_root::<TriangleMesh>(&data[..]) };

    LoadedTriangleMesh {
        mesh: archived,
        mmap,
    }
}

pub struct Renderer {
    pub data: RefCell<Option<InnerData>>,

    start: Instant,
    win_size: winit::dpi::PhysicalSize<u32>,
    pub gfx: GFXDevice,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let start: Instant = Instant::now();

        let gfx = GFXDevice::new(window);

        Renderer {
            gfx,
            start,
            data: RefCell::new(None),
            win_size: window.inner_size(),
        }
    }

    pub fn init(&self) {
        let gfx = &self.gfx;

        trace!("loading asset ...");
        // let mesh = load_triangled_mesh(&Path::new("baked/retro_car.mesh"));
        let mesh = load_triangled_mesh(&Path::new("baked/future_car.mesh"));
        trace!("finshed loading asset");

        mesh.maps.iter().for_each(|x| {
            trace!(
                "what map :{} , dims:{:?}, size:{}",
                x.name,
                x.source.dimentions,
                x.source.source.len()
            );
        });

        let mut indices = vec![];
        indices.extend_from_slice(mesh.indices.as_slice());

        let desc = GPUBufferDesc {
            size: (indices.len()) * mem::size_of::<u32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::INDEX_BUFFER | GPUBufferUsage::TRANSFER_DST,
            index_buffer_type: Some(GPUIndexedBufferType::U32),
            ..Default::default()
        };

        trace!(
            "index desc : bsize{:} , count:{:?} ,slicebsize:{} ",
            desc.size,
            indices.len(),
            indices.as_bytes().len()
        );

        let index_buffer = gfx.create_buffer(&desc, Some(indices.as_bytes()));

        let mut positions = vec![];
        positions.extend_from_slice(mesh.positions.as_slice());

        let desc = GPUBufferDesc {
            size: positions.len() * 3 * mem::size_of::<f32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        trace!(
            "vertex desc : {:?}, count:{:?}  , slicebsize:{:}",
            desc.size,
            positions.len(),
            positions.as_bytes().len()
        );
        let vertex_buffer = gfx.create_buffer(&desc, Some(positions.as_bytes()));

        let vertex_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple.vert.spv")[..]);

        let frag_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple.frag.spv")[..]);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: self.win_size.width,
            height: self.win_size.height,
            framebuffer_count: 2,
            ..Default::default()
        });

        let pso_desc = PipelineStateDesc {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            renderpass: swapchain.internal.borrow().renderpass,
            vertex_input_binding_descriptions: vec![],
            vertex_input_attribute_descriptions: vec![],
            // vertex_input_binding_descriptions: vec![
            //     vk::VertexInputBindingDescription {
            //         binding: 0,
            //         stride: (mem::size_of::<f32>() * 3) as u32,
            //         input_rate: vk::VertexInputRate::VERTEX,
            //     },
            //     // vk::VertexInputBindingDescription {
            //     //     binding: 1,
            //     //     stride: (mem::size_of::<f32>() * 2) as u32,
            //     //     input_rate: vk::VertexInputRate::VERTEX,
            //     // },
            // ],
            // vertex_input_attribute_descriptions: vec![
            //     vk::VertexInputAttributeDescription {
            //         location: 0,
            //         binding: 0,
            //         format: vk::Format::R32G32B32_SFLOAT,
            //         offset: 0u32,
            //     },
            //     // vk::VertexInputAttributeDescription {
            //     //     location: 1,
            //     //     binding: 1,
            //     //     format: vk::Format::R32G32_SFLOAT,
            //     //     offset: 0u32,
            //     // },
            // ],
        };

        let pso = gfx.create_pipeline_state(&pso_desc);

        // create uniform
        let uniform_desc = GPUBufferDesc {
            size: std::mem::size_of::<MVP>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        };
        let mvp_buffer = gfx.create_buffer(&uniform_desc, None);

        let mut g = 0;
        let images: Vec<GPUImage> = (&mesh.maps)
            .into_iter()
            .map(|map| {
                let mut desc = GPUImageDesc::default();
                desc.width = map.source.dimentions[0];
                desc.height = map.source.dimentions[1];

                let data = &map.source.source;
                desc.size = data.len();

                let img = gfx.create_image(&desc, Some(data.as_slice()));

                trace!(
                    "map[{}] name :{} ,dims :{:?}",
                    g,
                    map.name,
                    map.source.dimentions
                );
                g += 1;
                img
            })
            .collect();

        let base_texture_view_index =
            gfx.create_image_view(&images[albedo_index], vk::ImageAspectFlags::COLOR, 1, 1);

        let sampler = gfx.create_sampler();

        //TODO: convert to gpuonly
        let vertices_desc = GPUBufferDesc {
            size: 512 * 2usize.pow(20),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::SHADER_DEVICE_ADDRESS
                | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        // bindless stuff

        // trace!("213 {:?}", mesh.uvs);
        let mut buf_builder = BufferBuilder::default();

        let pos_offset = buf_builder.add(&mesh.positions) as u32;
        let uv_offset = buf_builder.add(&mesh.uvs) as u32;

        let vertices_buffer = gfx.create_buffer(&vertices_desc, Some(buf_builder.data()));

        //TODO: convert to gpuonly
        let gpu_meshes_desc = GPUBufferDesc {
            size: 1 * mem::size_of::<GpuMesh>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let gpu_meshes = vec![GpuMesh {
            pos_offset,
            uv_offset,
        }];

        let gpu_mesh_buffer = gfx.create_buffer(&gpu_meshes_desc, Some(gpu_meshes.as_bytes()));

        //TODO: convert to gpuonly
        let draw_cmds_desc = GPUBufferDesc {
            size: 1 * mem::size_of::<vk::DrawIndexedIndirectCommand>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::INDIRECT_BUFFER
                | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let i = 0;
        let draw_cmd = vk::DrawIndexedIndirectCommand::builder()
            .index_count(mesh.indices.len() as u32)
            .instance_count(1)
            .first_index(0)
            .first_instance(i)
            .vertex_offset(0)
            .build();
        let draw_cmds = vec![draw_cmd];

        let draw_cmds_buffer = gfx.create_buffer(&draw_cmds_desc, Some(draw_cmds.as_bytes()));

        //
        let uv_desc = GPUBufferDesc {
            size: mesh.uvs.as_bytes().len(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let uv_buffer = gfx.create_buffer(&uv_desc, Some(mesh.uvs.as_bytes()));

        //

        self.data.borrow_mut().replace(InnerData {
            swapchain,
            vertex_buffer,
            index_buffer,
            pso,
            mesh,
            mvp_buffer,
            images,
            base_texture_view_index,
            sampler,
            vertices_buffer,
            gpu_mesh_buffer,
            draw_cmds_buffer,
            uv_buffer,
        });
    }

    fn update_uniform_buffer(width: i32, height: i32, start: &Instant) -> MVP {
        let elapsed = { start.elapsed() };

        let view = glam::Mat4::look_at_lh(
            glam::vec3(0.0f32, 2.0, 5.0),
            glam::vec3(0.0f32, 0.0, 0.0),
            glam::vec3(0.0f32, 1.0f32, 0.0f32),
        );
        let proj = glam::Mat4::perspective_lh(
            f32::to_radians(75.0f32),
            width as f32 / height as f32,
            0.01f32,
            1000.0f32,
        );

        // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
        let proj = proj.mul_mat4(&glam::mat4(
            glam::vec4(1.0f32, 0.0, 0.0, 0.0),
            glam::vec4(0.0f32, -1.0, 0.0, 0.0),
            glam::vec4(0.0f32, 0.0, 1.0f32, 0.0),
            glam::vec4(0.0f32, 0.0, 0.0f32, 1.0),
        ));

        let rot = glam::Quat::from_axis_angle(
            glam::vec3(0.0f32, 1.0, 0.0),
            f32::to_radians(elapsed.as_millis() as f32) * 0.05f32,
        );
        let model = glam::Mat4::from_quat(rot);

        // * glam::Mat4::from_scale(glam::vec3(0.05, 0.05, 0.05));

        MVP { proj, view, model }
    }

    pub fn render(&self) {
        let gfx = &self.gfx;
        let cmd = gfx.begin_command_buffer();
        unsafe {
            let inner_data = std::cell::Ref::map(self.data.borrow(), |f| f.as_ref().unwrap());

            let vertex_buffer = &inner_data.vertex_buffer;
            let index_buffer = &inner_data.index_buffer;
            let pso = &inner_data.pso;
            let mesh = &inner_data.mesh;
            let mvp_buffer = &inner_data.mvp_buffer;
            let swapchain = &inner_data.swapchain;
            let images = &inner_data.images;
            let base_texture_view_index = inner_data.base_texture_view_index;
            let sampler = &inner_data.sampler;

            let draw_cmds_buffer = &inner_data.draw_cmds_buffer;
            let vertices_buffer = &inner_data.vertices_buffer;
            let gpu_mesh_buffer = &inner_data.gpu_mesh_buffer;

            let uv_buffer = &inner_data.uv_buffer;

            gfx.bind_viewports(
                cmd,
                &[vk::Viewport {
                    x: 0f32,
                    y: 0f32,
                    width: self.win_size.width as f32,
                    height: self.win_size.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            gfx.bind_scissors(
                cmd,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.win_size.width,
                        height: self.win_size.height,
                    },
                }],
            );

            gfx.begin_renderpass_sc(cmd, &swapchain);

            gfx.bind_pipeline(cmd, pso);

            // // push_constants

            let vertices_ptr = {
                let info = vk::BufferDeviceAddressInfo::builder()
                    .buffer(vertices_buffer.internal.borrow().buffer);

                gfx.device.get_buffer_device_address(&info)
            };
            let p_const = vec![vertices_ptr];
            let constants = p_const.as_bytes();
            gfx.bind_push_constants(cmd, pso, constants);

            // update transform data
            let mvp = Renderer::update_uniform_buffer(
                self.win_size.width as i32,
                self.win_size.height as i32,
                &self.start,
            );
            (*mvp_buffer.internal)
                .borrow_mut()
                .allocation
                .mapped_slice_mut()
                .unwrap()[0..mem::size_of::<MVP>()]
                .copy_from_slice(slice::from_raw_parts(
                    (&mvp as *const MVP) as *const u8,
                    mem::size_of::<MVP>(),
                ));

            gfx.bind_resource_buffer(0, 0, &gpu_mesh_buffer);
            gfx.bind_resource_buffer(0, 1, &mvp_buffer);

            gfx.bind_resource_img(
                0,
                2,
                &images[albedo_index],
                base_texture_view_index,
                sampler,
            );
            // gfx.bind_resource_buffer(0, 3, &uv_buffer);

            // gfx.bind_vertex_buffer(cmd, vertex_buffer, 0);

            gfx.bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

            // gfx.draw_indexed(cmd, mesh.indices.len() as u32, 1, 0, 0, 0);
            // gfx.draw(cmd, 3 as u32, 1, 0, 0);

            let stride = mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
            gfx.draw_indexed_indirect(cmd, draw_cmds_buffer, 0, 1, stride);

            gfx.end_renderpass(cmd);

            // gfx.end_command_buffers();

            // gfx.wait_for_gpu();
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {}
}
