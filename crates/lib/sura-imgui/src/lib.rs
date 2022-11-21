use __core::ops::Deref;
use ash::vk;
pub use imgui::*;
use imgui_rs_vulkan_renderer::*;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use log::debug;
use std::time::Instant;

use winit::event::Event;
use {
    gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc},
    std::sync::{Arc, Mutex},
};
pub struct SuraImgui<'a> {
    imgui: Context,
    _font_size: f32,
    platform: WinitPlatform,
    renderer: Renderer,
    gfx: &'a sura_backend::vulkan::device::Device,
    last_frame: Instant,
    render_pass: sura_backend::vulkan::device::RenderPass,
    command_pool: vk::CommandPool,
}

impl<'a> SuraImgui<'a> {
    pub fn new(
        window: &winit::window::Window,
        gfx: &'a sura_backend::vulkan::device::Device,
    ) -> SuraImgui<'a> {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        }]);
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

        let instance = gfx.instance.clone();
        let device = gfx.device.clone();
        let physical_device = gfx.pdevice;
        let queue = gfx.graphics_queue;
        let graphics_q_index = gfx.graphics_queue_index;

        let render_pass = gfx.create_imgui_render_pass();
        let render_pass_vk = render_pass.internal.deref().borrow().render_pass;

        let command_pool = {
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_q_index)
                .flags(vk::CommandPoolCreateFlags::empty());
            unsafe {
                device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap()
            }
        };

        let renderer = {
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
            })
            .unwrap();

            Renderer::with_gpu_allocator(
                Arc::new(Mutex::new(allocator)),
                device,
                queue,
                command_pool,
                render_pass_vk,
                &mut imgui,
                Some(Options {
                    in_flight_frames: 1, //NOTE : swapchain might use more than one
                    ..Default::default()
                }),
            )
            .unwrap()
        };

        SuraImgui {
            renderer,
            gfx,
            render_pass,
            command_pool,
            platform,
            _font_size: font_size,
            imgui,
            last_frame: Instant::now(),
        }
    }

    pub fn on_update(&mut self, window: &winit::window::Window, event: &winit::event::Event<()>) {
        let Self {
            platform, imgui, ..
        } = self;

        platform.handle_event(imgui.io_mut(), window, event);

        match event {
            // New frame
            Event::NewEvents(_) => {
                let now = Instant::now();
                imgui.io_mut().update_delta_time(now - self.last_frame);
                self.last_frame = now;
            }

            _ => {}
        }
    }
    pub fn on_render<'app, B>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<()>,
        mut ui_callback: B,

        swapchain: &sura_backend::vulkan::device::Swapchain,
    ) where
        B: FnMut(&mut Ui) + 'app,
    {
        let Self {
            platform,
            imgui,
            gfx,
            render_pass,
            ..
        } = self;

        match event {
            // End of event processing
            Event::MainEventsCleared => {
                // Generate UI

                platform
                    .prepare_frame(imgui.io_mut(), window)
                    .expect("Failed to prepare frame");
                let mut ui = imgui.frame();
                ui_callback(&mut ui);
                platform.prepare_render(&ui, window);
                let draw_data = ui.render();

                let cmd = gfx.begin_command_buffer();

                gfx.bind_viewports(
                    cmd,
                    &[vk::Viewport {
                        x: 0f32,
                        y: 0f32,
                        width: window.inner_size().width as f32,
                        height: window.inner_size().height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                gfx.bind_scissors(
                    cmd,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: window.inner_size().width,
                            height: window.inner_size().height,
                        },
                    }],
                );

                gfx.begin_renderpass_imgui(cmd, swapchain, render_pass);

                self.renderer
                    .cmd_draw(gfx.get_current_vulkan_cmd(), draw_data)
                    .unwrap();

                gfx.end_renderpass(cmd);
            }

            _ => {}
        }
    }
}

impl Drop for SuraImgui<'_> {
    fn drop(&mut self) {
        unsafe {
            debug!("destroying SuraImgui");
            self.gfx
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
