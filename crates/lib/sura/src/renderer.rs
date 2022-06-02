use std::{
    cell::{RefCell, RefMut},
    fs::File,
    mem::{self, ManuallyDrop},
    path::Path,
    slice,
    time::Instant,
};

use log::{trace, warn};
use sura_asset::mesh::*;
use sura_backend::ash::vk;
use sura_backend::vulkan::vulkan_device::*;
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{
    buffer::{BufferBuilder, BufferData},
    camera::AppCamera,
    gpu_structs::{Camera, GpuMesh},
    input::Input,
};

const albedo_index: usize = 9;

#[derive(Clone, Copy)]
pub struct MeshHandle(pub u32);

struct UploadedTriangledMesh {
    pub gpu_mesh_index: u64,
    pub materials: Vec<MeshMaterial>,
    pub transform: glam::Mat4,
}

struct UploadedTexture {
    image: GPUImage,
    sampler: Sampler,
}

pub struct InnerData {
    pub swapchain: Swapchain,
    pso: PipelineState,
    camera_buffer: GPUBuffer,
    // bindless data
    index_buffer: GPUBuffer,
    vertices_buffer: GPUBuffer,
    gpu_mesh_buffer: GPUBuffer,
    draw_cmds_buffer: GPUBuffer,
    transforms_buffer: GPUBuffer,
    //offsets
    index_buffer_offset: u64,
    vertices_buffer_offset: u64,
    gpu_mesh_buffer_index: u64,
    draw_cmds_buffer_offset: u64,
    transforms_buffer_index: u64,
    //
    textures: Vec<UploadedTexture>,
    uploaded_meshes: Vec<UploadedTriangledMesh>,
    a_buffer: GPUBuffer,
    b_buffer: GPUBuffer,
    pso_compute: PipelineState,
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

        let compute_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple.comp.spv")[..]);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: win_size.width,
            height: win_size.height,
            ..Default::default()
        });

        let pso_desc = PipelineStateDesc {
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            compute: None,
            renderpass: swapchain.internal.borrow().renderpass,
            vertex_input_binding_descriptions: None,
            vertex_input_attribute_descriptions: None,
        };

        let pso = gfx.create_pipeline_state(&pso_desc);

        let desc = GPUBufferDesc {
            size: Self::MAX_INDEX_COUNT * mem::size_of::<u32>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::INDEX_BUFFER | GPUBufferUsage::TRANSFER_DST,
            index_buffer_type: Some(GPUIndexedBufferType::U32),
            name: "index_buffer".into(),
        };

        let index_buffer = gfx.create_buffer(&desc, None);

        let vertices_desc = GPUBufferDesc {
            size: Self::MAX_VERTEX_DATA_SIZE,
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::SHADER_DEVICE_ADDRESS
                | GPUBufferUsage::TRANSFER_DST,
            name: "vertices_buffer".into(),
            ..Default::default()
        };

        let vertices_buffer = gfx.create_buffer(&vertices_desc, None);

        let gpu_meshes_desc = GPUBufferDesc {
            size: Self::MAX_MESH_COUNT * mem::size_of::<GpuMesh>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER | GPUBufferUsage::TRANSFER_DST,
            name: "gpu_meshes_buffer".into(),
            ..Default::default()
        };

        let gpu_mesh_buffer = gfx.create_buffer(&gpu_meshes_desc, None);

        let draw_cmds_desc = GPUBufferDesc {
            size: Self::MAX_MESH_COUNT * mem::size_of::<vk::DrawIndexedIndirectCommand>(),
            memory_location: MemLoc::GpuOnly,
            usage: GPUBufferUsage::STORAGE_BUFFER
                | GPUBufferUsage::INDIRECT_BUFFER
                | GPUBufferUsage::TRANSFER_DST,
            name: "draw_cmds_buffer".into(),
            ..Default::default()
        };

        let draw_cmds_buffer = gfx.create_buffer(&draw_cmds_desc, None);
        //

        let transforms_uni_desc = GPUBufferDesc {
            size: Self::MAX_MESH_COUNT * mem::size_of::<glam::Mat4>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::STORAGE_BUFFER,
            name: "transforms_buffer".into(),
            ..Default::default()
        };
        let transforms_buffer = gfx.create_buffer(&transforms_uni_desc, None);

        let camera_uni_desc = GPUBufferDesc {
            size: std::mem::size_of::<Camera>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::UNIFORM_BUFFER,
            name: "camera_buffer".into(),
            ..Default::default()
        };
        let camera_buffer = gfx.create_buffer(&camera_uni_desc, None);

        // compute test
        let a_desc = GPUBufferDesc {
            size: 10 * mem::size_of::<f32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::STORAGE_BUFFER,
            name: "compute a buffer".into(),
            ..Default::default()
        };

        let a_buffer = gfx.create_buffer(&a_desc, None);

        let b_desc = GPUBufferDesc {
            size: 10 * mem::size_of::<f32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::STORAGE_BUFFER,
            name: "compute bb buffer".into(),
            ..Default::default()
        };

        let b_buffer = gfx.create_buffer(&b_desc, None);

        let pso_desc_c = PipelineStateDesc {
            compute: Some(compute_shader),
            renderpass: swapchain.internal.borrow().renderpass,
            ..Default::default()
        };

        let pso_compute = gfx.create_pipeline_state(&pso_desc_c);

        //

        InnerData {
            swapchain,
            pso,

            camera_buffer,
            transforms_buffer,

            index_buffer,
            vertices_buffer,
            gpu_mesh_buffer,
            draw_cmds_buffer,

            index_buffer_offset: 0,
            vertices_buffer_offset: 0,
            gpu_mesh_buffer_index: 0,
            draw_cmds_buffer_offset: 0,
            transforms_buffer_index: 0,
            uploaded_meshes: Vec::new(),
            textures: Vec::new(),

            ///
            a_buffer,
            b_buffer,
            pso_compute,
        }
    }

    pub fn add_mesh(&self, path: &Path) -> MeshHandle {
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
            materials,
            transform: glam::Mat4::IDENTITY,
        };
        data.uploaded_meshes.push(uploaded_mesh);

        MeshHandle((data.uploaded_meshes.len() - 1) as u32)
    }

    pub fn update_transform(&self, mesh: MeshHandle, transform: &glam::Mat4) {
        //
        let InnerData {
            transforms_buffer,
            uploaded_meshes,
            ..
        } = &mut *self.data.borrow_mut();

        let mut mesh = &mut uploaded_meshes[mesh.0 as usize];
        mesh.transform = transform.clone();
        let transform = &mesh.transform;

        unsafe {
            let mut buff = (*transforms_buffer.internal).borrow_mut();

            let first = (mem::size_of::<glam::Mat4>() * mesh.gpu_mesh_index as usize);
            let end = first + mem::size_of::<glam::Mat4>();

            let slice = &mut buff.allocation.mapped_slice_mut().unwrap()[first..end];

            let transform_cols = transform.to_cols_array();
            let size = std::mem::size_of_val(&transform_cols);
            slice.copy_from_slice(slice::from_raw_parts(
                transform_cols.as_ptr() as *const u8,
                size,
            ));
        };
    }
    pub fn update_camera(&self, camera: &dyn AppCamera) {
        let InnerData { camera_buffer, .. } = &*self.data.borrow_mut();

        let cam = Camera {
            proj: camera.projection(),
            view: camera.view(),
        };

        unsafe {
            (*camera_buffer.internal)
                .borrow_mut()
                .allocation
                .mapped_slice_mut()
                .unwrap()[0..mem::size_of::<Camera>()]
                .copy_from_slice(slice::from_raw_parts(
                    (&cam as *const Camera) as *const u8,
                    mem::size_of::<Camera>(),
                ));
        }
    }

    pub fn render(&self) {
        let ten_millis = core::time::Duration::from_millis(2000);
        std::thread::sleep(ten_millis);

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
                camera_buffer,
                transforms_buffer,
                index_buffer_offset,
                vertices_buffer_offset,
                gpu_mesh_buffer_index,
                draw_cmds_buffer_offset,
                transforms_buffer_index,
                uploaded_meshes,
                textures,
                a_buffer,
                b_buffer,
                pso_compute,
                ..
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

            gfx.bind_resource_buffer(0, 0, 0, &gpu_mesh_buffer);

            let imgs = textures
                .iter()
                .map(|tex| tex.image.clone())
                .collect::<Vec<GPUImage>>();

            let samplers = textures
                .iter()
                .map(|tex| tex.sampler.clone())
                .collect::<Vec<Sampler>>();

            let array_indices = std::iter::successors(Some(0u32), |n| Some(n + 1))
                .take(imgs.len())
                .collect::<Vec<_>>();

            let view_indices = std::iter::repeat(0u32).take(imgs.len()).collect::<Vec<_>>();

            gfx.bind_resource_imgs(0, 1, &array_indices, &imgs, &view_indices, &samplers);

            gfx.bind_resource_buffer(1, 0, 0, &camera_buffer);
            gfx.bind_resource_buffer(1, 1, 0, &transforms_buffer);

            gfx.bind_index_buffer(cmd, &index_buffer, 0, vk::IndexType::UINT32);

            gfx.draw_indexed_indirect(
                cmd,
                &draw_cmds_buffer,
                0,
                uploaded_meshes.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            gfx.end_renderpass(cmd);

            //compute test

            gfx.bind_pipeline(cmd, &pso_compute);

            let a = std::iter::successors(Some(1f32), |n| Some(n + 1.0))
                .take(10)
                .collect::<Vec<_>>();

            let a = a.as_bytes();
            gfx.copy_to_buffer(&a_buffer, 0, a);
            trace!("A buffer : {:?}", a);

            gfx.bind_resource_buffer(0, 0, 0, &a_buffer);
            gfx.bind_resource_buffer(0, 1, 0, &b_buffer);

            gfx.disptach_compute(cmd, 10, 1, 1);

            gfx.end_command_buffers();
            // gfx.wait_for_gpu();

            let b_data = {
                let ptr = (*b_buffer.internal)
                    .borrow_mut()
                    .allocation
                    .mapped_slice_mut()
                    .unwrap()
                    .as_ptr();
                let slice = slice::from_raw_parts(ptr as *const f32, 10);
                slice
            };
            trace!("B data is :{:?} ", b_data);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {}
}
