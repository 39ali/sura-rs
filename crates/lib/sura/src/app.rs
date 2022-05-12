use log::{error, trace};
use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

use std::rc::Rc;

use crate::renderer;

pub trait App {
    fn on_init(&self);
    fn on_update(&self);
    fn on_render(&self, renderer: &renderer::Renderer);
    fn on_gui(&self, ui: &mut sura_imgui::Ui);
}

pub struct AppCreateInfo {
    pub title: String,
    pub window_width: u32,
    pub window_height: u32,
}

pub fn run<'app>(app: impl App + 'app, info: AppCreateInfo) {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("trace,imgui_rs_vulkan_renderer::renderer=warn"),
    )
    .init();

    let mut events_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title(info.title)
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(info.window_width),
            f64::from(info.window_height),
        ))
        .build(&events_loop)
        .unwrap();

    let renderer = &Rc::new(renderer::Renderer::new(&window));
    let mut imgui = { sura_imgui::SuraImgui::new(&window, &renderer.gfx) };
    // let imgui: &'static mut sura_imgui::SuraImgui = &mut imgui;
    renderer.init();
    app.on_init();

    events_loop.run_return(move |event: Event<'_, ()>, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        imgui.on_update(&window, &event);
        app.on_update();

        match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                        *control_flow = ControlFlow::Exit;
                    }
                }

                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                app.on_render(&renderer);
                renderer.render();

                // imgui pass
                let ui_callback = |ui: &mut sura_imgui::Ui| {
                    app.on_gui(ui);
                };
                imgui.on_render(&window, &event, ui_callback);

                renderer.gfx.end_command_buffers();
                renderer.gfx.wait_for_gpu();
                // *control_flow = ControlFlow::Exit;
            }

            Event::LoopDestroyed => {}

            // Event::RedrawRequested(_) => {
            //     // Redraw the application.
            //     //
            //     // It's preferable for applications that do not render continuously to render in
            //     // this event rather than in MainEventsCleared, since rendering in here allows
            //     // the program to gracefully handle redraws requested by the OS.
            // }
            _ => (),
        };
    });
}
