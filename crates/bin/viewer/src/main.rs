use std::path::Path;

use sura::{
    gui,
    renderer::{self, Renderer},
};

struct Viewer;

impl sura::app::App for Viewer {
    fn on_init(&self, renderer: &Renderer) {
        renderer.add_mesh(&Path::new("baked/future_car.mesh"));
    }

    fn on_update(&self) {}

    fn on_render(&self, renderer: &Renderer) {}

    fn on_gui(&self, ui: &mut gui::Ui) {
        let mut run = true;
        ui.show_demo_window(&mut run);
    }
}

fn main() {
    let viewer = Viewer {};
    let app_info = sura::app::AppCreateInfo {
        window_width: 1024,
        window_height: 768,
        title: "Sura viewer".into(),
    };

    sura::app::run(viewer, app_info);
}
