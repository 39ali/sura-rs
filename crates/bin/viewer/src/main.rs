use std::{cell::RefMut, path::Path, time::Instant};

use sura::{
    app::AppState,
    camera::FreeCamera,
    camera::OrbitCamera,
    gui::{self, *},
    input::Input,
    math::{self, Vec3},
    renderer::{MeshHandle, Renderer},
};

struct Viewer {
    meshes: Vec<MeshHandle>,
    timer: Instant,
    free_camera: FreeCamera,
    orbit_camera: OrbitCamera,
    show_gui: bool,
}

impl sura::app::App for Viewer {
    fn on_init(&mut self, renderer: &Renderer) {
        self.meshes
            .push(renderer.add_mesh(Path::new("baked/mystical_sphere.mesh")));

        // self.meshes
        //     .push(renderer.add_mesh(&Path::new("baked/future_car.mesh")));
    }

    fn on_update(&mut self, _renderer: &Renderer, input: &Input, mut state: RefMut<AppState>) {
        if input.is_pressed(winit::event::VirtualKeyCode::LControl) {
            state.camera = Box::new(self.orbit_camera);
        } else if input.is_released(winit::event::VirtualKeyCode::LControl) {
            state.camera = Box::new(self.free_camera);
        }

        if input.is_pressed(winit::event::VirtualKeyCode::Key1) {
            self.show_gui = !self.show_gui;
            state.camera.lock_camera(self.show_gui);
        }
    }

    fn on_render(&mut self, renderer: &Renderer) {
        let elapsed = { self.timer.elapsed() };
        let rot = math::Quat::from_axis_angle(
            math::vec3(0.0f32, 1.0, 0.0),
            f32::to_radians(elapsed.as_millis() as f32) * 0.05f32,
        );
        let transform = math::Mat4::from_translation(math::Vec3::new(0f32, 0.0, 10.0))
            * math::Mat4::from_scale(Vec3::splat(10.0))
            * math::Mat4::from_quat(rot);

        renderer.update_transform(self.meshes[0], &transform);

        // let transform = math::Mat4::from_translation(math::Vec3::new(2f32, 0.0, 0.0))
        //     * math::Mat4::from_quat(rot);
        // renderer.update_transform(self.meshes[1], &transform);
    }

    fn on_gui(&mut self, renderer: &Renderer, _input: &Input, ui: &mut gui::Ui) {
        if self.show_gui {
            gui::Window::new("Hello world")
                .position([5.0, 5.0], Condition::FirstUseEver)
                .size([400.0, 600.0], Condition::FirstUseEver)
                .build(ui, || {
                    // ui.text_wrapped("Hello world!");
                    // ui.text_wrapped("こんにちは世界！");
                    // if ui.button("hey") {}

                    // gui::Slider::new("i32 value with range", -999.0, 999.0)
                    //     .build(ui, &mut self.val);

                    // ui.button("This...is...imgui-rs!");

                    // ui.separator();
                    // let mouse_pos = ui.io().mouse_pos;
                    // ui.text(format!(
                    //     "Mouse Position: ({:.1},{:.1})",
                    //     mouse_pos[0], mouse_pos[1]
                    // ));

                    // draw timestamps !
                    ui.dummy([1.0, 10.0]);
                    ui.text("GPU");
                    ui.separator();
                    let timestamps = renderer.get_timestamps();

                    let pipeline_time = if timestamps.len() > 1 {
                        (timestamps[1] - timestamps[0]) * 1e-6
                    } else {
                        0.0
                    };
                    ui.text(format!("full pipeline time: ({:.5}ms)", pipeline_time));
                });
        }
    }
}

fn main() {
    let app_info = sura::app::AppCreateInfo {
        window_width: 1024,
        window_height: 768,
        title: "Sura viewer".into(),
    };

    let aspect_ratio = app_info.window_width as f32 / app_info.window_height as f32;

    let viewer = Viewer {
        meshes: Vec::new(),
        timer: Instant::now(),
        orbit_camera: OrbitCamera::new(aspect_ratio),
        free_camera: FreeCamera::new(aspect_ratio),
        show_gui: false,
    };

    sura::app::run(viewer, app_info);
}
