use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

use std::rc::Rc;

use crate::{
    input::Input,
    renderer::{self, Renderer},
};

pub trait App {
    fn on_init(&mut self, renderer: &Renderer);
    fn on_update(&self);
    fn on_render(&self, renderer: &Renderer);
    fn on_gui(&self, ui: &mut sura_imgui::Ui);
}

pub struct AppCreateInfo {
    pub title: String,
    pub window_width: u32,
    pub window_height: u32,
}

pub fn run<'app>(mut app: impl App + 'app, info: AppCreateInfo) {
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

    let renderer = &(renderer::Renderer::new(&window));

    let mut imgui = {
        let gfx = &renderer.gfx;
        sura_imgui::SuraImgui::new(&window, gfx)
    };

    let mut input = Input::default();

    app.on_init(&renderer);

    events_loop.run_return(move |event: Event<'_, ()>, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        input.on_update(&event);
        imgui.on_update(&window, &event);
        renderer.on_update(&input);
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

                // render pass
                renderer.render();
                // imgui pass
                {
                    let ui_callback = |ui: &mut sura_imgui::Ui| {
                        app.on_gui(ui);
                    };
                    imgui.on_render(&window, &event, ui_callback);
                }
                // push cmds to queue
                renderer.gfx.end_command_buffers();
                renderer.gfx.wait_for_gpu();

                input.on_clear();
            }

            Event::LoopDestroyed => {}

            _ => (),
        };
    });
}
