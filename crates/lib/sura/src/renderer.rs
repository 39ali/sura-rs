use std::{
    borrow::Borrow,
    cell::{RefCell, RefMut},
    fs::File,
    mem::{self, ManuallyDrop},
    ops::Deref,
    path::Path,
    slice,
    time::Instant,
};

use glam::{Mat4, Vec3};
use log::{info, trace, warn};
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
    vertex_count: u32,
    index_count: u32,
    pub gpu_mesh: GpuMesh,
    pub index_buffer_offset_b: u64,
}

struct UploadedTexture {
    image: GPUImage,
    sampler: Sampler,
}
struct AccelStruct {
    accel_struct: vk::AccelerationStructureKHR,
    backing_buffer: GPUBuffer,
}

struct AccelStructData {
    build_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    range_info: vk::AccelerationStructureBuildRangeInfoKHR,
    accel: AccelStruct,
}
pub struct InnerData {
    pub swapchain: Swapchain,
    pso: PipelineState,
    bindless_set: DescSet,
    frame_constants_buffer: GPUBuffer,
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
    global_set: DescSet,
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
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/out/simple_vs.spv")[..]);

        let frag_shader =
            gfx.create_shader(&include_bytes!("../../../../assets/shaders/out/pbr_ps.spv")[..]);

        // let compute_shader =
        //     gfx.create_shader(&include_bytes!("../../../../assets/shaders/simple_cs.spv")[..]);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: win_size.width,
            height: win_size.height,
            ..Default::default()
        });

        let pso_desc = PipelineStateDesc {
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            compute: None,
            renderpass: swapchain.internal.deref().borrow().renderpass,
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

        let frame_constants_uni_desc = GPUBufferDesc {
            size: std::mem::size_of::<Camera>(),
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::UNIFORM_BUFFER,
            name: "camera_buffer".into(),
            ..Default::default()
        };
        let frame_constants_buffer = gfx.create_buffer(&frame_constants_uni_desc, None);

        let mut bindless_set = gfx.create_set_bindless(&[
            //meshes
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            //maps
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_count(512 * 1024)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            //vertices
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            //sampler
            vk::DescriptorSetLayoutBinding::builder()
                .binding(32)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
        ]);

        let mut global_set = gfx.create_set_bindless(&[
            //frame_constants
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            //frame_constants
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
        ]);

        bindless_set.bind_resource_buffer(0, 0, &gpu_mesh_buffer);

        bindless_set.bind_resource_buffer(2, 0, &vertices_buffer);

        global_set.bind_resource_buffer(0, 0, &frame_constants_buffer);
        global_set.bind_resource_buffer(1, 0, &transforms_buffer);

        // time query
        let time_query = gfx.create_query(2);

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
            // compute: Some(compute_shader),
            renderpass: swapchain.internal.deref().borrow().renderpass,
            ..Default::default()
        };

        let pso_compute = gfx.create_pipeline_state(&pso_desc_c);

        //

        //
        //raytracing
        // Self::rtx(gfx);
        //

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
            transforms_buffer_index: 0,
            uploaded_meshes: Vec::new(),
            textures: Vec::new(),

            ///
            a_buffer,
            b_buffer,
            pso_compute,
            global_set,
            bindless_set,

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

    //

    // fn create_blas(
    //     &self,
    //     cmd: Cmd,
    //     blases_input: &[(
    //         vk::AccelerationStructureGeometryKHRBuilder,
    //         vk::AccelerationStructureBuildRangeInfoKHR,
    //     )],
    // ) -> Vec<AccelStructData> {
    //     struct BuildASInfo {
    //         build_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    //         size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    //         range_info: vk::AccelerationStructureBuildRangeInfoKHR,
    //     }

    //     let build_as = Vec::with_capacity(blases_input.len());

    //     let mut as_total_size = 0;
    //     let mut max_scratch_size = 0;

    //     for blas in blases_input {
    //         let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
    //             .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
    //             .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
    //             .geometries(&[*blas.0])
    //             .build();

    //         let size_info = unsafe {
    //             self.gfx
    //                 .acceleration_structure_loader
    //                 .get_acceleration_structure_build_sizes(
    //                     vk::AccelerationStructureBuildTypeKHR::DEVICE,
    //                     &build_info,
    //                     &[blas.1.primitive_count],
    //                 )
    //         };

    //         max_scratch_size += size_info.build_scratch_size;
    //         as_total_size += size_info.acceleration_structure_size;

    //         let build_as_info = BuildASInfo {
    //             build_info,
    //             size_info,
    //             range_info: blas.1,
    //         };

    //         build_as.push(build_as_info);
    //     }

    //     //alloc scratch buffer
    //     let scratch_buffer = self.gfx.create_buffer(
    //         &GPUBufferDesc {
    //             memory_location: MemLoc::GpuOnly,
    //             size: max_scratch_size as usize,
    //             usage: GPUBufferUsage::SHADER_DEVICE_ADDRESS | GPUBufferUsage::STORAGE_BUFFER,
    //             index_buffer_type: None,
    //             name: "blas scratch buffer".into(),
    //         },
    //         None,
    //     );
    //     let scratch_buffer_address = self.gfx.get_buffer_address(&scratch_buffer);

    //     //create blases
    //     let cmd = self.gfx.get_cmd(cmd).cmd;

    //     let mut acceleration_structs = Vec::new();

    //     for build_as_info in build_as {
    //         let accel_struct = self.create_acceleration(
    //             as_total_size,
    //             vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    //         );
    //         build_as_info.build_info.dst_acceleration_structure = accel_struct.accel_struct;
    //         build_as_info.build_info.scratch_data.device_address = scratch_buffer_address;
    //         acceleration_structs.push(AccelStructData {
    //             range_info: build_as_info.range_info,
    //             build_info: build_as_info.build_info,
    //             size_info: build_as_info.size_info,
    //             accel: accel_struct,
    //         });

    //         unsafe {
    //             self.gfx
    //                 .acceleration_structure_loader
    //                 .cmd_build_acceleration_structures(
    //                     cmd,
    //                     &[build_as_info.build_info],
    //                     &[&[build_as_info.range_info]],
    //                 );
    //         }
    //     }

    //     //
    //     //
    //     let memory_barriers = &[vk::MemoryBarrier::builder()
    //         .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
    //         .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)
    //         .build()];

    //     unsafe {
    //         self.gfx.device.cmd_pipeline_barrier(
    //             cmd,
    //             vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
    //             vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
    //             vk::DependencyFlags::empty(),
    //             memory_barriers,
    //             &[],
    //             &[],
    //         )
    //     }

    //     acceleration_structs
    // }

    // fn to_TransformMatrixKHR(&self, mat: &Mat4) -> vk::TransformMatrixKHR {
    //     let matrix: [f32; 12] = Default::default();
    //     matrix.copy_from_slice(&mat.to_cols_array()[0..12]);
    //     vk::TransformMatrixKHR { matrix }
    // }

    // fn create_acceleration(&self, size: u64, ty: vk::AccelerationStructureTypeKHR) -> AccelStruct {
    //     let accel_buffer = self.gfx.create_buffer(
    //         &GPUBufferDesc {
    //             memory_location: MemLoc::GpuOnly,
    //             size: size as usize,
    //             usage: GPUBufferUsage::SHADER_DEVICE_ADDRESS
    //                 | GPUBufferUsage::ACCELERATION_STRUCTURE_STORAGE,
    //             index_buffer_type: None,
    //             name: "accel_backing_buffer".into(),
    //         },
    //         None,
    //     );
    //     let accel_buffer_raw = accel_buffer.internal.deref().borrow().buffer;
    //     let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
    //         .size(size as u64)
    //         .buffer(accel_buffer_raw)
    //         .ty(ty);
    //     let accel_struct = unsafe {
    //         self.gfx
    //             .acceleration_structure_loader
    //             .create_acceleration_structure(&create_info, None)
    //             .expect("error accel structure")
    //     };

    //     let accel = AccelStruct {
    //         accel_struct,
    //         backing_buffer: accel_buffer,
    //     };

    //     accel
    // }

    // fn get_blas_address(&self, accel: vk::AccelerationStructureKHR) -> vk::DeviceAddress {
    //     unsafe {
    //         self.gfx
    //             .acceleration_structure_loader
    //             .get_acceleration_structure_device_address(
    //                 &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
    //                     .acceleration_structure(accel)
    //                     .build(),
    //             )
    //     }
    // }

    // fn create_tlas(&self, cmd: Cmd, blases: Vec<AccelStructData>, transform: &Mat4) -> AccelStruct {
    //     // let as_instance = vk::AccelerationStructureInstanceKHR {
    //     //     transform: self.to_TransformMatrixKHR(transform),
    //     //     instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
    //     //     instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
    //     //         0,
    //     //         vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
    //     //     ),
    //     //     acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
    //     //         device_handle: self.get_blas_address(blas),
    //     //     },
    //     // };
    //     // self.build_tlas(cmd, &vec![as_instance])
    //     todo!()
    // }
    // fn build_tlas(
    //     &self,
    //     cmd: Cmd,
    //     instances: &Vec<vk::AccelerationStructureInstanceKHR>,
    // ) -> AccelStruct {
    //     let cmd = self.gfx.get_cmd(cmd).cmd;

    //     let b_instances = instances.as_bytes();

    //     let instances_buffer = self.gfx.create_buffer(
    //         &GPUBufferDesc {
    //             memory_location: MemLoc::GpuOnly,
    //             size: b_instances.len(),
    //             usage: GPUBufferUsage::SHADER_DEVICE_ADDRESS
    //                 | GPUBufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    //             index_buffer_type: None,
    //             name: "accel_buffer".into(),
    //         },
    //         Some(b_instances),
    //     );

    //     let memory_barriers = &[vk::MemoryBarrier::builder()
    //         .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
    //         .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
    //         .build()];

    //     unsafe {
    //         self.gfx.device.cmd_pipeline_barrier(
    //             cmd,
    //             vk::PipelineStageFlags::TRANSFER,
    //             vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
    //             vk::DependencyFlags::empty(),
    //             memory_barriers,
    //             &[],
    //             &[],
    //         )
    //     };
    //     // self.create_cmd_tlas(cmd, geometry, range_info, transform, blas)
    //     todo!()
    // }

    // //
    // fn create_cmd_tlas(
    //     &self,
    //     cmd: &CommandBuffer,
    //     instBufferAddr: vk::DeviceAddress,
    //     instance_count: u32,
    // ) -> AccelStruct {
    //     let instances_vk = vk::AccelerationStructureGeometryInstancesDataKHR::builder()
    //         .data(vk::DeviceOrHostAddressConstKHR {
    //             device_address: instBufferAddr,
    //         })
    //         .build();

    //     let mut top_as_geometry = vk::AccelerationStructureGeometryKHR::default();
    //     top_as_geometry.geometry.instances = instances_vk;
    //     top_as_geometry.geometry_type = vk::GeometryTypeKHR::INSTANCES;

    //     let geometries = &[top_as_geometry];
    //     let geometries_counts = &[instance_count];
    //     //find sizes
    //     //VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
    //     let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
    //         .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
    //         .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
    //         .geometries(geometries)
    //         .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

    //     let size_info = unsafe {
    //         self.gfx
    //             .acceleration_structure_loader
    //             .get_acceleration_structure_build_sizes(
    //                 vk::AccelerationStructureBuildTypeKHR::DEVICE,
    //                 &build_info,
    //                 geometries_counts,
    //             )
    //     };

    //     //
    //     //create tlas
    //     let accel_struct = self.create_acceleration(
    //         size_info.acceleration_structure_size,
    //         vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    //     );

    //     // Allocate the scratch memory
    //     let scratch_buffer = self.gfx.create_buffer(
    //         &GPUBufferDesc {
    //             memory_location: MemLoc::GpuOnly,
    //             size: size_info.build_scratch_size as usize,
    //             usage: GPUBufferUsage::SHADER_DEVICE_ADDRESS
    //                 | GPUBufferUsage::ACCELERATION_STRUCTURE_STORAGE,
    //             index_buffer_type: None,
    //             name: "accel_buffer".into(),
    //         },
    //         None,
    //     );
    //     let scratch_buffer_address = self.gfx.get_buffer_address(&scratch_buffer);

    //     // update build info
    //     build_info.dst_acceleration_structure(accel_struct.accel_struct);
    //     build_info.scratch_data.device_address = scratch_buffer_address;

    //     //

    //     //build tlas

    //     let build_range_infos = &[vk::AccelerationStructureBuildRangeInfoKHR {
    //         first_vertex: 0,
    //         primitive_offset: 0,
    //         transform_offset: 0,
    //         primitive_count: instance_count,
    //     }];

    //     let infos = [build_info.build()];

    //     unsafe {
    //         self.gfx
    //             .acceleration_structure_loader
    //             .cmd_build_acceleration_structures(cmd.cmd, &infos, &[build_range_infos]);
    //     };

    //     accel_struct
    // }

    // fn create_vkgeometry(
    //     &self,
    //     vertex_buffer: &GPUBuffer,
    //     index_buffer: &GPUBuffer,
    //     vertex_offset_b: u64,
    //     index_offset_b: u64,
    //     vertex_count: u32,
    //     index_count: u32,
    // ) -> (
    //     vk::AccelerationStructureGeometryKHRBuilder,
    //     vk::AccelerationStructureBuildRangeInfoKHR,
    // ) {
    //     let vertex_address = self.gfx.get_buffer_address(&vertex_buffer) + vertex_offset_b;
    //     let index_address = self.gfx.get_buffer_address(&index_buffer) + index_offset_b;

    //     //
    //     let geometry_triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
    //         .vertex_format(vk::Format::R32G32B32_SFLOAT)
    //         .vertex_data(vk::DeviceOrHostAddressConstKHR {
    //             device_address: vertex_address,
    //         })
    //         .vertex_stride(mem::size_of::<[f32; 3]>() as u64)
    //         .index_type(vk::IndexType::UINT32)
    //         .index_data(vk::DeviceOrHostAddressConstKHR {
    //             device_address: index_address,
    //         })
    //         .max_vertex(vertex_count)
    //         .build();

    //     let geometry = vk::AccelerationStructureGeometryKHR::builder()
    //         .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
    //         .flags(vk::GeometryFlagsKHR::OPAQUE)
    //         .geometry(vk::AccelerationStructureGeometryDataKHR {
    //             triangles: geometry_triangles,
    //         });

    //     let offset = vk::AccelerationStructureBuildRangeInfoKHR {
    //         first_vertex: 0,
    //         primitive_count: index_count / 3,
    //         primitive_offset: 0,
    //         transform_offset: 0,
    //     };

    //     (geometry, offset)
    // }

    // fn update_rtx(
    //     &self,
    //     cmd: Cmd,
    //     uploaded_meshes: &Vec<UploadedTriangledMesh>,
    //     index_buffer: &GPUBuffer,
    //     vertex_buffer: &GPUBuffer,
    // ) {
    //     todo!();
    //     static mut init: bool = false;
    //     // if !init {
    //     //     init = true;

    //     //     let blases = Vec::with_capacity(uploaded_meshes.capacity());

    //     //     for mesh in uploaded_meshes {
    //     //         let gpu_mesh = mesh.gpu_mesh;
    //     //         let blas = self.create_vkgeometry(
    //     //             vertex_buffer,
    //     //             index_buffer,
    //     //             gpu_mesh.pos_offset as u64,
    //     //             mesh.index_buffer_offset_b,
    //     //             mesh.vertex_count,
    //     //             mesh.index_count,
    //     //         );

    //     //         blases.push(blas);
    //     //     }

    //     //     let blases = self.create_blas(cmd, blases.as_slice());

    //     //     // let tlases = self.create_tlas(cmd, geometry, range_info, transform, blases);
    //     //     todo!()
    //     // }
    // }
    // //
    pub fn on_init(&self) {
        // add lights
        let l_len = self.data.borrow().light_positions.len();
        for i in 0..l_len {
            let mesh = self.add_mesh(&Path::new("baked/sphere2.mesh"));

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
        let mut textures: Vec<UploadedTexture> = (&mesh.maps)
            .into_iter()
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
                bindless_set,
                global_set,
                time_query,
                timestamps,
                vertices_buffer,
                ..
            } = &mut *self.data.borrow_mut();

            //
            // self.update_rtx(cmd, &uploaded_meshes, &index_buffer, &vertices_buffer);

            //

            let imgs = textures
                .iter()
                .map(|tex| tex.image.clone())
                .collect::<Vec<GPUImage>>();

            if let Some(texture) = textures.get(0) {
                let sampler = &texture.sampler;
                bindless_set.bind_resource_sampler(32, 0, sampler);
            }

            let array_indices = std::iter::successors(Some(0u32), |n| Some(n + 1))
                .take(imgs.len())
                .collect::<Vec<_>>();

            let view_indices = std::iter::repeat(0u32).take(imgs.len()).collect::<Vec<_>>();

            bindless_set.bind_resource_imgs(1, &array_indices, &imgs, &view_indices);

            bindless_set.flush();
            global_set.flush();
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

            gfx.bind_swapchain(cmd, swapchain);

            gfx.reset_query(cmd, time_query);

            gfx.write_time_stamp(cmd, time_query, vk::PipelineStageFlags::TOP_OF_PIPE);

            gfx.begin_renderpass_sc(cmd, &swapchain);
            gfx.bind_pipeline(cmd, &pso);

            // // push_constants

            // let vertices_ptr = gfx.get_buffer_address(vertices_buffer);
            // let p_const = vec![vertices_ptr];
            // let constants = p_const.as_bytes();
            // gfx.bind_push_constants(cmd, &pso, constants);

            gfx.bind_set(cmd, bindless_set, 0);
            gfx.bind_set(cmd, global_set, 1);

            gfx.bind_index_buffer(cmd, &index_buffer, 0, vk::IndexType::UINT32);

            gfx.draw_indexed_indirect(
                cmd,
                &draw_cmds_buffer,
                0,
                uploaded_meshes.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            gfx.end_renderpass(cmd);

            gfx.write_time_stamp(cmd, time_query, vk::PipelineStageFlags::BOTTOM_OF_PIPE);

            if let Some(times) = gfx.get_query_result(time_query) {
                *timestamps = times;
            }

            // .and_then(|times| {
            //     trace!("ftimes {:?}", times);
            //     for t in times.chunks(2) {
            //         trace!("time took to render {:?}ms", (t[1] - t[0]) * 1e-6);
            //     }

            //     return Some(0);
            // });

            // gfx.end_command_buffers();
            //             //compute test

            //             gfx.bind_pipeline(cmd, &pso_compute);

            //             let a = std::iter::successors(Some(1f32), |n| Some(n + 1.0))
            //                 .take(10)
            //                 .collect::<Vec<_>>();

            //             let a = a.as_bytes();
            //             gfx.copy_to_buffer(&a_buffer, 0, a);
            //             trace!("A buffer : {:?}", a);

            //             gfx.bind_resource_buffer(0, 0, 0, &a_buffer);
            //             gfx.bind_resource_buffer(0, 1, 0, &b_buffer);

            //             gfx.disptach_compute(cmd, 10, 1, 1);

            //             gfx.end_command_buffers();
            //             // gfx.wait_for_gpu();

            //             let b_data = {
            //                 let ptr = (*b_buffer.internal)
            //                     .borrow_mut()
            //                     .allocation
            //                     .mapped_slice_mut()
            //                     .unwrap()
            //                     .as_ptr();
            //                 let slice = slice::from_raw_parts(ptr as *const f32, 10);
            //                 slice
            //             };
            // // k            trace!("B data is :{:?} ", b_data);
        }
    }

    pub fn get_timestamps(&self) -> Vec<f64> {
        self.data.borrow().timestamps.clone()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {}
}
