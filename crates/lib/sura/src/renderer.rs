use std::{
    cell::{RefCell, RefMut},
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
use sura_backend::ash::vk;
//TODO : remove
use log::{trace, warn};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    buffer::{BufferBuilder, BufferData},
    gpu_structs::{GpuMesh, MVP},
};

const albedo_index: usize = 9;

struct UploadedTriangledMesh {
    pub gpu_mesh_index: u64,
    pub materials: Vec<MeshMaterial>,
    pub transform_index: u32,
}

struct UploadedTexture {
    image: GPUImage,
    sampler: Sampler,
}

pub struct InnerData {
    pub swapchain: Swapchain,
    pso: PipelineState,
    mvp_buffer: GPUBuffer,
    // bindless data
    index_buffer: GPUBuffer,
    vertices_buffer: GPUBuffer,
    gpu_mesh_buffer: GPUBuffer,
    draw_cmds_buffer: GPUBuffer,
    //offsets
    index_buffer_offset: u64,
    vertices_buffer_offset: u64,
    gpu_mesh_buffer_index: u64,
    draw_cmds_buffer_offset: u64,
    //
    uploaded_meshes: Vec<UploadedTriangledMesh>,
    textures: Vec<UploadedTexture>,
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
    pub data: RefCell<InnerData>,

    start: Instant,
    win_size: winit::dpi::PhysicalSize<u32>,
    pub gfx: GFXDevice,
}

impl Renderer {
    const MAX_INDEX_COUNT: usize = 25 * 2usize.pow(20);
    const MAX_MESH_COUNT: usize = 1024;
    const MAX_VERTEX_DATA_SIZE: usize = 512 * 2usize.pow(20);
    const FRAME_COUNT: u32 = 2;

    pub fn new(window: &Window) -> Self {
        let start: Instant = Instant::now();

        let gfx = GFXDevice::new(window);
        let win_size = window.inner_size();
        let data = RefCell::new(Self::init(&gfx, &win_size));
        Renderer {
            gfx,
            start,
            win_size,
            data,
        }
    }

    fn init(gfx: &GFXDevice, win_size: &PhysicalSize<u32>) -> InnerData {
        let vertex_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple.vert.spv")[..]);

        let frag_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple.frag.spv")[..]);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: win_size.width,
            height: win_size.height,
            ..Default::default()
        });

        let pso_desc = PipelineStateDesc {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            renderpass: swapchain.internal.borrow().renderpass,
            vertex_input_binding_descriptions: vec![],
            vertex_input_attribute_descriptions: vec![],
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

        let desc = GPUBufferDesc {
            size: Self::MAX_INDEX_COUNT * mem::size_of::<u32>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::INDEX_BUFFER | GPUBufferUsage::TRANSFER_DST,
            index_buffer_type: Some(GPUIndexedBufferType::U32),
            ..Default::default()
        };

        let index_buffer = gfx.create_buffer(&desc, None);

        let vertices_desc = GPUBufferDesc {
            size: Self::MAX_VERTEX_DATA_SIZE,
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::SHADER_DEVICE_ADDRESS
                | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let vertices_buffer = gfx.create_buffer(&vertices_desc, None);

        let gpu_meshes_desc = GPUBufferDesc {
            size: Self::MAX_MESH_COUNT * mem::size_of::<GpuMesh>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let gpu_mesh_buffer = gfx.create_buffer(&gpu_meshes_desc, None);

        let draw_cmds_desc = GPUBufferDesc {
            size: Self::MAX_MESH_COUNT * mem::size_of::<vk::DrawIndexedIndirectCommand>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::INDIRECT_BUFFER
                | GPUBufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let draw_cmds_buffer = gfx.create_buffer(&draw_cmds_desc, None);
        //

        InnerData {
            swapchain,
            pso,
            mvp_buffer,

            index_buffer,
            vertices_buffer,
            gpu_mesh_buffer,
            draw_cmds_buffer,

            index_buffer_offset: 0,
            vertices_buffer_offset: 0,
            gpu_mesh_buffer_index: 0,
            draw_cmds_buffer_offset: 0,

            uploaded_meshes: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn add_mesh(&self, path: &Path) {
        let mesh = load_triangled_mesh(path);

        trace!("materials :{} ", mesh.materials.len());

        let mut data = self.data.borrow_mut();

        //update textures
        let sampler = self.gfx.create_sampler();

        let mut g = 0;
        let mut textures: Vec<UploadedTexture> = (&mesh.maps)
            .into_iter()
            .map(|map| {
                let mut desc = GPUImageDesc::default();
                desc.width = map.source.dimentions[0];
                desc.height = map.source.dimentions[1];

                let data = &map.source.source;
                desc.size = data.len();

                let img = self.gfx.create_image(&desc, Some(data.as_slice()));

                trace!(
                    "map[{}] name :{} ,dims :{:?}",
                    g,
                    map.name,
                    map.source.dimentions
                );
                g += 1;

                self.gfx
                    .create_image_view(&img, vk::ImageAspectFlags::COLOR, 1, 1);

                UploadedTexture {
                    image: img,
                    sampler: sampler.clone(),
                }
            })
            .collect();

        let current_textures_offset = data.textures.len();
        data.textures.append(&mut textures);

        // update materials
        let materials: Vec<MeshMaterial> = mesh
            .materials
            .iter()
            .map(|mat| {
                use rkyv::Deserialize;
                let mut deserialized: MeshMaterial =
                    mat.deserialize(&mut rkyv::Infallible).unwrap();

                for map_indx in &mut deserialized.maps_index {
                    *map_indx += current_textures_offset as u32;
                }

                deserialized
            })
            .collect();

        // update indices
        let indices = mesh.indices.as_bytes();
        self.gfx
            .copy_to_buffer(&data.index_buffer, data.index_buffer_offset, indices);
        let first_index = data.index_buffer_offset / mem::size_of::<u32>() as u64;
        data.index_buffer_offset += indices.len() as u64;

        // update vertex data
        let current_vertices_buffer_offset = data.vertices_buffer_offset;
        let mut buf_builder = BufferBuilder::default();
        let pos_offset = (buf_builder.add(&mesh.positions) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");
        let uv_offset = (buf_builder.add(&mesh.uvs) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let normal_offset = (buf_builder.add(&mesh.normals) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let colors_offset = (buf_builder.add(&mesh.colors) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let tangents_offset = (buf_builder.add(&mesh.tangents) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let material_ids_offset = (buf_builder.add(&mesh.material_ids)
            + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let materials_data_offset = (buf_builder.add(&materials) + current_vertices_buffer_offset)
            .try_into()
            .expect("number too big");

        let vertex_data = buf_builder.data();
        self.gfx.copy_to_buffer(
            &data.vertices_buffer,
            data.vertices_buffer_offset,
            vertex_data,
        );
        data.vertices_buffer_offset += vertex_data.len() as u64;

        // update gpu mesh buffer
        {
            let gpu_mesh = GpuMesh {
                pos_offset,
                uv_offset,
                normal_offset,
                colors_offset,
                tangents_offset,
                material_ids_offset,
                materials_data_offset,
            };

            let gpu_mesh_data = gpu_mesh.as_bytes();
            let gpu_mesh_buffer_offset = gpu_mesh_data.len() as u64 * data.gpu_mesh_buffer_index;

            self.gfx
                .copy_to_buffer(&data.gpu_mesh_buffer, gpu_mesh_buffer_offset, gpu_mesh_data);
            data.gpu_mesh_buffer_index += 1;
        }
        // update draw commands buffer
        {
            let draw_cmd = vk::DrawIndexedIndirectCommand::builder()
                .index_count(mesh.indices.len() as u32)
                .instance_count(1)
                .first_index(first_index as u32)
                .first_instance((data.gpu_mesh_buffer_index - 1) as u32)
                .vertex_offset(0)
                .build();
            let draw_cmds = vec![draw_cmd];
            let draw_cmd_raw = draw_cmds.as_bytes();

            self.gfx.copy_to_buffer(
                &data.draw_cmds_buffer,
                data.draw_cmds_buffer_offset,
                draw_cmd_raw,
            );
            data.draw_cmds_buffer_offset += draw_cmd_raw.len() as u64;
        }

        // update uploaded mesh
        let uploaded_mesh = UploadedTriangledMesh {
            gpu_mesh_index: data.gpu_mesh_buffer_index - 1,
            materials: Vec::new(),
            transform_index: 0,
        };
        data.uploaded_meshes.push(uploaded_mesh);
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
            let InnerData {
                index_buffer,
                vertices_buffer,
                gpu_mesh_buffer,
                draw_cmds_buffer,
                swapchain,
                pso,
                mvp_buffer,
                index_buffer_offset,
                vertices_buffer_offset,
                gpu_mesh_buffer_index,
                draw_cmds_buffer_offset,
                uploaded_meshes,
                textures,
            } = &*self.data.borrow_mut();

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

            gfx.bind_pipeline(cmd, &pso);

            // // push_constants

            let vertices_ptr = {
                let info = vk::BufferDeviceAddressInfo::builder()
                    .buffer(vertices_buffer.internal.borrow().buffer);

                gfx.device.get_buffer_device_address(&info)
            };
            let p_const = vec![vertices_ptr];
            let constants = p_const.as_bytes();
            gfx.bind_push_constants(cmd, &pso, constants);

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
                &textures[albedo_index].image,
                0,
                &textures[albedo_index].sampler,
            );

            gfx.bind_index_buffer(cmd, &index_buffer, 0, vk::IndexType::UINT32);

            gfx.draw_indexed_indirect(
                cmd,
                &draw_cmds_buffer,
                0,
                uploaded_meshes.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );

            gfx.end_renderpass(cmd);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {}
}
