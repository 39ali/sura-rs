use glam::{Mat4, Vec3};
use log::trace;
use winit::event::{Event, VirtualKeyCode, WindowEvent};

use crate::input::Input;

use super::camera::Camera;

pub struct FreeCamera {
    cam: Camera,
}

impl FreeCamera {
    pub fn new(aspect_ratio: f32) -> Self {
        let mut cam = Camera::new(aspect_ratio);
        cam.pos -= Vec3::new(0.0, 0.0, 10.0);
        Self { cam }
    }

    pub fn on_update(&mut self, input: &Input) {
        let mut mov_speed = 0.05;
        let roll_speed = 1.0;
        if input.is_pressed(VirtualKeyCode::LShift) {
            mov_speed = 0.1;
        }

        let forward = self.cam.forward() * mov_speed;
        let right = self.cam.right() * mov_speed;

        if input.is_pressed(VirtualKeyCode::W) {
            self.cam.move_pos(&forward);
        }

        if input.is_pressed(VirtualKeyCode::S) {
            self.cam.move_pos(&-forward);
        }

        if input.is_pressed(VirtualKeyCode::A) {
            self.cam.move_pos(&-right);
        }

        if input.is_pressed(VirtualKeyCode::D) {
            self.cam.move_pos(&right);
        }

        if input.is_pressed(VirtualKeyCode::Q) {
            self.cam.roll(f32::to_radians(roll_speed));
        }

        if input.is_pressed(VirtualKeyCode::E) {
            self.cam.roll(-f32::to_radians(roll_speed));
        }

        let mouse_motion = input.get_mouse_motion();
        if mouse_motion != (0.0, 0.0) {
            self.cam.yaw(f32::to_radians(0.1 * mouse_motion.0 as f32));

            self.cam.pitch(f32::to_radians(0.1 * mouse_motion.1 as f32));
        }
    }

    pub fn projection(&self) -> Mat4 {
        self.cam.projection()
    }

    pub fn view(&self) -> Mat4 {
        self.cam.view()
    }
}
