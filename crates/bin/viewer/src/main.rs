use std::{path::Path, time::Instant};

use sura::{
    glam, gui,
    renderer::{MeshHandle, Renderer},
};

struct Viewer {
    meshes: Vec<MeshHandle>,
    timer: Instant,
}

impl sura::app::App for Viewer {
    fn on_init(&mut self, renderer: &Renderer) {
        self.meshes
            .push(renderer.add_mesh(&Path::new("baked/future_car.mesh")));

        self.meshes
            .push(renderer.add_mesh(&Path::new("baked/future_car.mesh")));
    }

    fn on_update(&self) {}

    fn on_render(&self, renderer: &Renderer) {
        let elapsed = { self.timer.elapsed() };
        let rot = glam::Quat::from_axis_angle(
            glam::vec3(0.0f32, 1.0, 0.0),
            f32::to_radians(elapsed.as_millis() as f32) * 0.05f32,
        );
        let transform = glam::Mat4::from_quat(rot);

        renderer.update_transform(self.meshes[0], &transform);

        let transform = glam::Mat4::from_translation(glam::Vec3::new(2f32, 0.0, 0.0))
            * glam::Mat4::from_quat(rot);
        renderer.update_transform(self.meshes[1], &transform);
    }

    fn on_gui(&self, ui: &mut gui::Ui) {
        let mut run = true;
        ui.show_demo_window(&mut run);
    }
}

fn main() {
    let viewer = Viewer {
        meshes: Vec::new(),
        timer: Instant::now(),
    };
    let app_info = sura::app::AppCreateInfo {
        window_width: 1024,
        window_height: 768,
        title: "Sura viewer".into(),
    };

    sura::app::run(viewer, app_info);
}
