use std::{cell::RefCell, mem, slice, time::Instant};

use crate::{
    renderable::{self, *},
    vulkan_device::*,
};
use ash::vk;
use winit::window::Window;

#[derive(Default, Debug)]
struct MVP {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}
pub struct InnerData {
    swapchain: Swapchain,
    vertex_buffer: GPUBuffer,
    index_buffer: GPUBuffer,
    pso: PipelineState,
    renderable: Renderable,
    mvp_buffer: GPUBuffer,
}

pub struct Renderer {
    data: RefCell<Option<InnerData>>,

    start: Instant,
    win_size: winit::dpi::PhysicalSize<u32>,
    gfx: GFXDevice,
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

        let renderable = renderable::load_gltf("./models/gltf_logo/scene.gltf");

        println!("{:?}", renderable.textures.len());

        assert!(
            renderable.meshes.len() == 1,
            "multiple meshes aren't supported:({})meshes",
            renderable.meshes.len()
        );
        let mesh = &renderable.meshes[0];

        let mut desc = GPUBufferDesc {
            size: 0u64,
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::INDEX_BUFFER,
            ..Default::default()
        };

        let index_buffer = match mesh.index_buffer {
            Indices::None => gfx.create_buffer::<u32>(&desc, None),
            Indices::U32(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U32);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                gfx.create_buffer(&desc, Some(b))
            }
            Indices::U16(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U16);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                gfx.create_buffer(&desc, Some(b))
            }
            Indices::U8(ref b) => {
                desc.index_buffer_type = Some(GPUIndexedBufferType::U8);
                desc.size = std::mem::size_of_val(b.as_slice()) as u64;
                gfx.create_buffer(&desc, Some(b))
            }
        };

        let mesh_buffer = mesh.get_buffer();

        //NOTE(ALI): we're using cpuTogpu because we don't support GpuOnly yet
        let desc = GPUBufferDesc {
            size: mesh_buffer.len() as u64,
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        let vertex_buffer = gfx.create_buffer(&desc, Some(&mesh_buffer));

        // create uniform
        let desc = GPUBufferDesc {
            size: std::mem::size_of::<MVP>() as u64,
            memory_location: MemLoc::CpuToGpu,
            usage: GPUBufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        };

        let vertex_shader = gfx.create_shader(&include_bytes!("../shaders/vert.spv")[..]);

        let frag_shader = gfx.create_shader(&include_bytes!("../shaders/frag.spv")[..]);

        let pso_desc = PipelineStateDesc {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            fragment: Some(frag_shader),
            vertex: Some(vertex_shader),
            vertex_input_binding_descriptions: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride: mesh.stride() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vertex_input_attribute_descriptions: vec![
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
            ],
        };
        let pso = gfx.create_pipeline_state(&pso_desc);
        let mvp_buffer = gfx.create_buffer::<u8>(&desc, None);

        let swapchain = gfx.create_swapchain(&SwapchainDesc {
            width: self.win_size.width,
            height: self.win_size.height,
            framebuffer_count: 2,
            ..Default::default()
        });

        self.data.borrow_mut().replace(InnerData {
            swapchain: (swapchain),
            vertex_buffer,
            index_buffer,
            pso,
            renderable,
            mvp_buffer,
        });
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

    pub fn render(&self) {
        let gfx = &self.gfx;
        let cmd = gfx.begin_command_buffer();
        unsafe {
            let inner_data = std::cell::Ref::map(self.data.borrow(), |f| f.as_ref().unwrap());

            let vertex_buffer = &inner_data.vertex_buffer;
            let index_buffer = &inner_data.index_buffer;
            let pso = &inner_data.pso;
            let renderable = &inner_data.renderable;
            let mvp_buffer = &inner_data.mvp_buffer;
            let swapchain = &inner_data.swapchain;

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

            gfx.begin_renderpass(cmd, &swapchain);

            gfx.bind_pipeline(cmd, pso);

            gfx.bind_vertex_buffer(cmd, vertex_buffer, 0);

            // push_constants
            let mvp = Renderer::update_uniform_buffer(
                self.win_size.width as i32,
                self.win_size.height as i32,
                &self.start,
            );
            let constants =
                slice::from_raw_parts((&mvp as *const MVP) as *const u8, mem::size_of::<MVP>());
            gfx.bind_push_constants(cmd, pso, constants);

            (*mvp_buffer.internal)
                .borrow_mut()
                .allocation
                .mapped_slice_mut()
                .unwrap()[0..mem::size_of::<MVP>()]
                .copy_from_slice(slice::from_raw_parts(
                    (&mvp as *const MVP) as *const u8,
                    mem::size_of::<MVP>(),
                ));

            gfx.bind_resource(0, 0, &mvp_buffer);

            let index_type = match index_buffer.internal.borrow().desc.index_buffer_type {
                Some(ref t) => match t {
                    GPUIndexedBufferType::U32 => vk::IndexType::UINT32,
                    GPUIndexedBufferType::U16 => vk::IndexType::UINT16,
                    GPUIndexedBufferType::U8 => vk::IndexType::UINT8_EXT,
                },
                _ => {
                    panic!("index buffer type is not defined");
                }
            };

            gfx.bind_index_buffer(cmd, index_buffer, 0, index_type);

            let index_count = match renderable.meshes[0].index_buffer {
                Indices::None => 0,
                Indices::U32(ref i) => i.len(),
                Indices::U16(ref i) => i.len(),
                Indices::U8(ref i) => i.len(),
            } as u32;

            gfx.draw_indexed(cmd, index_count, 1, 0, 0, 1);

            gfx.end_renderpass(cmd);

            gfx.end_command_buffers();

            gfx.wait_for_gpu();
        }
    }
}
