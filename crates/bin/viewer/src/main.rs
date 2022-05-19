use std::{cell::RefMut, path::Path, time::Instant};

use log::trace;
use sura::{
    app::AppState,
    camera::FreeCamera,
    camera::OrbitCamera,
    gui,
    input::Input,
    math,
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
            .push(renderer.add_mesh(&Path::new("baked/future_car.mesh")));

        self.meshes
            .push(renderer.add_mesh(&Path::new("baked/future_car.mesh")));
    }

    fn on_update(&mut self, renderer: &Renderer, input: &Input, mut state: RefMut<AppState>) {
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
        let transform = math::Mat4::from_quat(rot);

        renderer.update_transform(self.meshes[0], &transform);

        let transform = math::Mat4::from_translation(math::Vec3::new(2f32, 0.0, 0.0))
            * math::Mat4::from_quat(rot);
        renderer.update_transform(self.meshes[1], &transform);
    }

    fn on_gui(&mut self, input: &Input, ui: &mut gui::Ui) {
        if self.show_gui {
            ui.show_demo_window(&mut self.show_gui);
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
        show_gui: true,
    };

    sura::app::run(viewer, app_info);
}
