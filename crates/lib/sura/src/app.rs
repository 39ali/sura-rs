use std::cell::{RefCell, RefMut};

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

use crate::{
    camera::{AppCamera, FreeCamera},
    input::Input,
    renderer::{self, Renderer},
};

pub trait App {
    fn on_init(&mut self, renderer: &Renderer);
    fn on_update(&mut self, renderer: &Renderer, input: &Input, state: RefMut<AppState>);
    fn on_render(&mut self, app_state: &Renderer);
    fn on_gui(&mut self, renderer: &Renderer, input: &Input, ui: &mut sura_imgui::Ui);
}

pub struct AppCreateInfo {
    pub title: String,
    pub window_width: u32,
    pub window_height: u32,
}

pub struct AppState {
    pub camera: Box<dyn AppCamera>,
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
    let win_size = window.inner_size();

    // TODO: doesn't work ?
    window.set_cursor_visible(false);
    window.set_cursor_grab(true).unwrap();

    let renderer = &(renderer::Renderer::new(&window));

    let mut input = Input::default();

    let mut imgui = {
        let gfx = &renderer.gfx;
        sura_imgui::SuraImgui::new(&window, gfx)
    };

    let app_state = RefCell::new(AppState {
        camera: Box::new(FreeCamera::new(
            win_size.width as f32 / win_size.height as f32,
        )),
    });

    renderer.on_init();
    app.on_init(renderer);

    events_loop.run_return(move |event: Event<'_, ()>, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        //handle events
        {
            input.on_update(&event);
            imgui.on_update(&window, &event);
        }

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
                //updates
                app.on_update(renderer, &input, app_state.borrow_mut());
                //camera
                {
                    let cam = &mut *app_state.borrow_mut().camera;
                    cam.on_update(&input);
                    renderer.update_camera(cam);
                }
                //

                app.on_render(renderer);

                // render pass
                renderer.render();
                // imgui pass
                {
                    let ui_callback = |ui: &mut sura_imgui::Ui| {
                        app.on_gui(renderer, &input, ui);
                    };
                    imgui.on_render(&window, &event, ui_callback);
                }
                // push cmds to queue
                renderer.gfx.end_command_buffers();
                // panic!();
            }

            Event::LoopDestroyed => {}

            _ => (),
        };
    });
}
