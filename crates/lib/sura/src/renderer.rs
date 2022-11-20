use std::{
    cell::RefCell,
    fs::File,
    mem::{self, ManuallyDrop},
    ops::Deref,
    path::Path,
    slice,
};

use glam::{Mat4, Vec3};
use log::{info, trace};

use sura_asset::mesh::*;
use sura_backend::ash::vk;
use sura_backend::vulkan::vulkan_device::*;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    buffer::{BufferBuilder, BufferData},
    camera::AppCamera,
    gpu_structs::{Camera, GpuMesh},
};

#[derive(Clone, Copy)]
pub struct MeshHandle(pub u32);

#[allow(dead_code)]
struct UploadedTriangledMesh {
    pub gpu_mesh_index: u64,
    pub materials: Vec<MeshMaterial>,
    pub transform: glam::Mat4,
    vertex_count: u32,
    index_count: u32,
    pub gpu_mesh: GpuMesh,
    pub index_buffer_offset_b: u64,
}

struct UploadedTexture {
    image: GPUImage,
    sampler: Sampler,
}

pub struct InnerData {
    pub swapchain: Swapchain,
    pso: RasterPipeline,
    bind_group_bindless: BindGroup,
    frame_constants_buffer: GPUBuffer,
    // bindless data
    index_buffer: GPUBuffer,
    vertices_buffer: GPUBuffer,
    gpu_mesh_buffer: GPUBuffer,
    draw_cmds_buffer: GPUBuffer,
    transforms_buffer: GPUBuffer,
    //offsets to data on GPU
    index_buffer_offset: u64,
    vertices_buffer_offset: u64,
    gpu_mesh_buffer_index: u64,
    draw_cmds_buffer_offset: u64,
    //
    textures: Vec<UploadedTexture>,
    uploaded_meshes: Vec<UploadedTriangledMesh>,
    // a_buffer: GPUBuffer,
    // b_buffer: GPUBuffer,
    // pso_compute: ComputePipeline,
    bind_group_global: BindGroup,
    time_query: VkQueryPool,
    timestamps: Vec<f64>,

    //
    light_positions: Vec<Vec3>,
    light_colors: Vec<Vec3>,
}

fn load_triangled_mesh(path: &Path) -> LoadedTriangleMesh {
    let file = File::open(&path).unwrap_or_else(|e| panic!("Could not mmap {:?}: {:?}", path, e));

    let mmap = ManuallyDrop::new(unsafe { memmap2::MmapOptions::new().map(&file).unwrap() });

    let data = unsafe { slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

    let archived = unsafe { rkyv::archived_root::<TriangleMesh>(data) };

    LoadedTriangleMesh {
        mesh: archived,
        mmap,
    }
}

pub struct Renderer {
    pub data: RefCell<InnerData>,
    win_size: winit::dpi::PhysicalSize<u32>,
    pub gfx: GFXDevice,
}

impl Renderer {
    const MAX_INDEX_COUNT: usize = 25 * 2usize.pow(20);
    const MAX_MESH_COUNT: usize = 1024;
    const MAX_VERTEX_DATA_SIZE: usize = 512 * 2usize.pow(20);

    pub fn new(window: &Window) -> Self {
        let gfx = GFXDevice::new(window);
        let win_size = window.inner_size();
        let data = RefCell::new(Self::init(&gfx, &win_size));
        Renderer {
            gfx,
            win_size,
            data,
        }
    }

    fn init(gfx: &GFXDevice, win_size: &PhysicalSize<u32>) -> InnerData {
        let vertex_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/out/simple_vs.spv")[..]);

        let _frag_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/out/pbr_ps.spv")[..]);

        let frag_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/out/pbr_ps.spv")[..]);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: win_size.width,
            height: win_size.height,
            ..Default::default()
        });

        let bind_group_layout0 = BindGroupLayout {
            bindings: &[
                BindGroupBinding {
                    index: 0,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer,
                    count: 1,
                    non_uniform_indexing: false,
                },
                BindGroupBinding {
                    index: 1,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::IMAGE,
                    count: 512 * 1024,
                    non_uniform_indexing: true,
                },
                BindGroupBinding {
                    index: 2,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer,
                    count: 1,
                    non_uniform_indexing: false,
                },
                BindGroupBinding {
                    index: 32,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::SAMPLER,
                    count: 1,
                    non_uniform_indexing: false,
                },
            ],
        };

        let bind_group_layout1 = BindGroupLayout {
            bindings: &[
                BindGroupBinding {
                    index: 0,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::UniformBuffer,
                    count: 1,
                    non_uniform_indexing: false,
                },
                BindGroupBinding {
                    index: 1,
                    stages: ShaderStage::VERTEX,
                    ty: BindingType::StorageBuffer,
                    count: 1,
                    non_uniform_indexing: false,
                },
            ],
        };

        let mut bind_group_bindless = gfx.create_bind_group(&bind_group_layout0);
        let mut bind_group_global = gfx.create_bind_group(&bind_group_layout1);

        let pso = gfx.create_raster_pipeline(&RasterPipelineDesc {
            fragment: Some(frag_shader),
            vertex: Some(VertexState {
                shader: vertex_shader,
                entry_point: (&"main").to_string(),
                vertex_buffer_layouts: &[],
            }),
            layout: PipelineLayout {
                bind_group_layouts: &[&bind_group_layout0, &bind_group_layout1],
            },
            attachments: Some(&[AttachmentLayout {
                sample_count: 1,
                format: swapchain.internal.deref().borrow().format,
                initial_layout: vk::ImageLayout::GENERAL,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                op: AttachmentOp {
                    load: LoadOp::Clear,
                    store: true,
                },
            }]),
            depth_stencil: Some(DepthStencilState {
                sample_count: 1,
                format: GPUFormat::D32_SFLOAT_S8_UINT,
                op: AttachmentOp {
                    load: LoadOp::Clear,
                    store: false,
                },
            }),
        });

        let index_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: Self::MAX_INDEX_COUNT * mem::size_of::<u32>(),
                memory_location: MemLoc::GpuOnly,
                usage: GPUBufferUsage::INDEX_BUFFER | GPUBufferUsage::TRANSFER_DST,
                index_buffer_type: Some(GPUIndexedBufferType::U32),
                name: "index_buffer".into(),
            },
            None,
        );

        let vertices_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: Self::MAX_VERTEX_DATA_SIZE,
                memory_location: MemLoc::GpuOnly,
                usage: GPUBufferUsage::STORAGE_BUFFER
                    | GPUBufferUsage::SHADER_DEVICE_ADDRESS
                    | GPUBufferUsage::TRANSFER_DST,
                name: "vertices_buffer".into(),
                ..Default::default()
            },
            None,
        );

        let gpu_mesh_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: Self::MAX_MESH_COUNT * mem::size_of::<GpuMesh>(),
                memory_location: MemLoc::GpuOnly,
                usage: GPUBufferUsage::STORAGE_BUFFER | GPUBufferUsage::TRANSFER_DST,
                name: "gpu_meshes_buffer".into(),
                ..Default::default()
            },
            None,
        );

        let draw_cmds_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: Self::MAX_MESH_COUNT * mem::size_of::<vk::DrawIndexedIndirectCommand>(),
                memory_location: MemLoc::GpuOnly,
                usage: GPUBufferUsage::STORAGE_BUFFER
                    | GPUBufferUsage::INDIRECT_BUFFER
                    | GPUBufferUsage::TRANSFER_DST,
                name: "draw_cmds_buffer".into(),
                ..Default::default()
            },
            None,
        );
        //

        let transforms_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: Self::MAX_MESH_COUNT * mem::size_of::<glam::Mat4>(),
                memory_location: MemLoc::CpuToGpu,
                usage: GPUBufferUsage::STORAGE_BUFFER,
                name: "transforms_buffer".into(),
                ..Default::default()
            },
            None,
        );

        let frame_constants_buffer = gfx.create_buffer(
            &GPUBufferDesc {
                size: std::mem::size_of::<Camera>(),
                memory_location: MemLoc::CpuToGpu,
                usage: GPUBufferUsage::UNIFORM_BUFFER,
                name: "camera_buffer".into(),
                ..Default::default()
            },
            None,
        );

        bind_group_bindless.bind_resource_buffer(0, 0, &gpu_mesh_buffer);

        bind_group_bindless.bind_resource_buffer(2, 0, &vertices_buffer);

        bind_group_global.bind_resource_buffer(0, 0, &frame_constants_buffer);
        bind_group_global.bind_resource_buffer(1, 0, &transforms_buffer);

        // time query
        let time_query = gfx.create_query(2);

        // // compute test
        let a_desc = GPUBufferDesc {
            size: 10 * mem::size_of::<f32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::STORAGE_BUFFER,
            name: "compute a buffer".into(),
            ..Default::default()
        };

        let _a_buffer = gfx.create_buffer(&a_desc, None);

        let _b_desc = GPUBufferDesc {
            size: 10 * mem::size_of::<f32>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::STORAGE_BUFFER,
            name: "compute bb buffer".into(),
            ..Default::default()
        };

        // let b_buffer = gfx.create_buffer(&b_desc, None);

        // let pso_desc_c = {
        //     // compute: Some(compute_shader),
        //     renderpass: swapchain.internal.deref().borrow().renderpass,
        //     ..Default::default()
        // };

        // let pso_compute = gfx.create_compute_pipeline(&ComputePipelineStateDesc
        //     { compute: conp }
        //     );

        InnerData {
            swapchain,
            pso,
            time_query,
            timestamps: Vec::new(),
            //
            frame_constants_buffer,
            transforms_buffer,

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
            ///
            bind_group_bindless,
            bind_group_global,
            //
            light_colors: [
                Vec3::new(300.0, 300.0, 300.0),
                Vec3::new(300.0, 300.0, 300.0),
                Vec3::new(300.0, 300.0, 300.0),
                Vec3::new(300.0, 300.0, 300.0),
            ]
            .into(),

            light_positions: [
                Vec3::new(-10.0, 10.0, -10.0),
                Vec3::new(10.0, 10.0, -10.0),
                Vec3::new(-10.0, -10.0, -10.0),
                Vec3::new(10.0, -10.0, -10.0),
            ]
            .into(),
        }
    }

    pub fn on_init(&self) {
        // add lights
        let l_len = self.data.borrow().light_positions.len();
        for i in 0..l_len {
            let mesh = self.add_mesh(Path::new("baked/sphere2.mesh"));

            let light_pos = self.data.borrow().light_positions[i];
            let transform = Mat4::from_translation(light_pos);
            self.update_transform(mesh, &transform);
        }
    }

    pub fn add_mesh(&self, path: &Path) -> MeshHandle {
        info!("adding mesh :{:?}", path);

        let mesh = load_triangled_mesh(path);

        info!("tangents :{:?}", mesh.tangents.len());

        trace!("materials :{} ", mesh.materials.len());

        let mut data = self.data.borrow_mut();

        //update textures
        let sampler = self.gfx.create_sampler();

        let mut g = 0;
        let mut textures: Vec<UploadedTexture> = mesh
            .maps
            .iter()
            .map(|map| {
                let mut desc = GPUImageDesc::default();
                desc.width = map.source.dimentions[0];
                desc.height = map.source.dimentions[1];

                if map.params.gamma == TextureGamma::Srgb {
                    desc.format = GPUFormat::R8G8B8A8_SRGB
                }

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
            index_count: mesh.indices.len() as u32,
            vertex_count: mesh.positions.len() as u32,
            index_buffer_offset_b: data.index_buffer_offset,
            gpu_mesh,
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
        mesh.transform = *transform;
        let transform = &mesh.transform;

        unsafe {
            let mut buff = (*transforms_buffer.internal).borrow_mut();

            let first = mem::size_of::<glam::Mat4>() * mesh.gpu_mesh_index as usize;
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
        let InnerData {
            frame_constants_buffer,
            light_colors,
            light_positions,
            ..
        } = &*self.data.borrow_mut();

        let light_positions = light_positions
            .iter()
            .map(|e| e.to_array())
            .collect::<Vec<[f32; 3]>>()
            .try_into()
            .unwrap();

        let light_colors = light_colors
            .iter()
            .map(|e| e.to_array())
            .collect::<Vec<[f32; 3]>>()
            .try_into()
            .unwrap();

        let cam = Camera {
            view: camera.view().to_cols_array(),
            proj: camera.projection().to_cols_array(),
            cam_pos: camera.pos().to_array(),
            light_positions,
            light_colors,
        };

        unsafe {
            (*frame_constants_buffer.internal)
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
        let gfx = &self.gfx;
        let cmd = gfx.begin_command_buffer();

        {
            let InnerData {
                index_buffer,
                draw_cmds_buffer,
                swapchain,
                pso,
                uploaded_meshes,
                textures,
                bind_group_bindless,
                bind_group_global,
                time_query,
                timestamps,
                ..
            } = &mut *self.data.borrow_mut();

            let imgs = textures
                .iter()
                .map(|tex| tex.image.clone())
                .collect::<Vec<GPUImage>>();

            if let Some(texture) = textures.get(0) {
                let sampler = &texture.sampler;
                bind_group_bindless.bind_resource_sampler(32, 0, sampler);
            }

            let array_indices = std::iter::successors(Some(0u32), |n| Some(n + 1))
                .take(imgs.len())
                .collect::<Vec<_>>();

            let view_indices = std::iter::repeat(0u32).take(imgs.len()).collect::<Vec<_>>();

            bind_group_bindless.bind_resource_imgs(1, &array_indices, &imgs, &view_indices);

            bind_group_bindless.flush();
            bind_group_global.flush();
            //

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

            gfx.get_next_img_swapchain(swapchain);

            gfx.reset_query(cmd, time_query);

            gfx.write_time_stamp(cmd, time_query, vk::PipelineStageFlags::TOP_OF_PIPE);

            gfx.begin_renderpass_sc(cmd, swapchain);
            gfx.bind_pipeline(cmd, pso);

            gfx.set_bind_groups(cmd, pso, &[bind_group_bindless, bind_group_global]);

            gfx.bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

            gfx.draw_indexed_indirect(
                cmd,
                draw_cmds_buffer,
                0,
                uploaded_meshes.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            gfx.end_renderpass(cmd);

            gfx.write_time_stamp(cmd, time_query, vk::PipelineStageFlags::BOTTOM_OF_PIPE);

            if let Some(times) = gfx.get_query_result(time_query) {
                *timestamps = times;
            }
        }
    }

    pub fn get_timestamps(&self) -> Vec<f64> {
        self.data.borrow().timestamps.clone()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {}
}
