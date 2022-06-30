use glam::{Mat4, Vec3};

use crate::input::Input;

use super::{camera::Camera, AppCamera};

#[derive(Clone, Copy)]
pub struct OrbitCamera {
    cam: Camera,
    target: Vec3,
    radius: f32,
    disable: bool,
}

impl OrbitCamera {
    pub fn new(aspect_ratio: f32) -> Self {
        let mut cam = Camera::new(aspect_ratio);
        let radius = -10f32;
        cam.pos = Vec3::Z * radius;

        Self {
            cam,
            disable: false,
            target: Vec3::default(),
            radius: 10.32,
        }
    }

    pub fn set_target(&mut self, t: &Vec3) {
        self.target = t.clone();
    }
}

impl AppCamera for OrbitCamera {
    fn on_update(&mut self, input: &Input) {
        let mouse_motion = input.get_mouse_motion();
        let mouse_wheel = input.get_mouse_wheel_motion();
        let mut update = false;

        if mouse_wheel.1 != 0.0 {
            self.radius += mouse_wheel.1;
            update = true;
        }

        if mouse_motion != (0.0, 0.0) {
            self.cam.yaw(f32::to_radians(0.5 * mouse_motion.0 as f32));
            self.cam.pitch(f32::to_radians(0.5 * mouse_motion.1 as f32));

            update = true;
        }

        if update {
            self.cam.pos = self.target + self.cam.forward() * -self.radius;
        }
    }

    fn pos(&self) -> glam::Vec3 {
        self.cam.pos
    }

    fn projection(&self) -> Mat4 {
        self.cam.projection()
    }

    fn view(&self) -> Mat4 {
        glam::Mat4::look_at_lh(
            self.cam.pos,
            self.cam.pos + self.cam.forward(),
            self.cam.up(),
        )
    }

    fn lock_camera(&mut self, d: bool) {
        self.disable = d;
    }

    fn get_lock_camera(&self) -> bool {
        self.disable
    }
}
