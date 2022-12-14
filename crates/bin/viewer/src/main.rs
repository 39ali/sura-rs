use std::{cell::RefMut, path::Path, time::Instant};

use sura::{
    app::AppState,
    camera::FreeCamera,
    camera::OrbitCamera,
    gui::{self, *},
    input::Input,
    math::{self, Vec3},
    renderer::{MeshHandle, Renderer},
    Light,
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

        self.meshes
            .push(renderer.add_mesh(Path::new("baked/sponza.mesh")));

        // renderer.add_light(Light::create_directional_light(
        //     &Vec3::new(0.0, -1.0, 0.0),
        //     1000.0,
        //     &Vec3::new(1.0, 1.0, 1.),
        // ));

        // renderer.add_light(Light::create_spot_light(
        //     &Vec3::new(0.0, 3.0, 0.0),
        //     &Vec3::new(0.0, 1.0, 0.0),
        //     100.0,
        //     &Vec3::new(1.0, 1.0, 1.),
        //     10.0,
        //     5.01f32.to_radians(),
        //     10.0f32.to_radians(),
        // ));

        for _i in 0..1 {
            renderer.add_light(Light::create_point_light(
                &Vec3::new(0.0, 2.0, 0.0),
                1000.0,
                &Vec3::new(1.0, 1.0, 1.),
                1000.0,
            ));
        }
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
        let transform = math::Mat4::from_translation(math::Vec3::new(0f32, 2.0, 0.0))
            * math::Mat4::from_scale(Vec3::splat(3.0))
            * math::Mat4::from_quat(rot);

        renderer.update_transform(self.meshes[0], &transform);
    }

    fn on_gui(&mut self, renderer: &Renderer, _input: &Input, ui: &mut gui::Ui) {
        if self.show_gui {
            ui.window("Hello world")
                .position([5.0, 5.0], Condition::FirstUseEver)
                .size([400.0, 600.0], Condition::FirstUseEver)
                .build(|| {
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

                    renderer.gui(ui);
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
