#[macro_use]
extern crate log;

extern crate ash;
extern crate winit;

extern crate base64;
extern crate gltf;

extern crate custom_error;
extern crate glam;

extern crate image;
extern crate indexmap;

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::{cell::RefCell, rc::Rc};

use ash::vk::{self};

mod vulkan_device;

mod gpu_structs;

mod renderable;

mod renderer;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();
    // env_logger::init();
    let window_width = 1024;
    let window_height = 768;

    let events_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Sura")
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(window_width),
            f64::from(window_height),
        ))
        .build(&events_loop)
        .unwrap();

    let renderer = Rc::new(renderer::Renderer::new(&window));
    renderer.init();

    events_loop.run(move |event: Event<'_, ()>, _, control_flow| {
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
                renderer.render();
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
