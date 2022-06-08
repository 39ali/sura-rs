use glam::{Mat4, Vec3};
use winit::event::VirtualKeyCode;

use crate::input::Input;

use super::{camera::Camera, AppCamera};

#[derive(Clone, Copy)]
pub struct FreeCamera {
    cam: Camera,
    disable: bool,
}

impl FreeCamera {
    pub fn new(aspect_ratio: f32) -> Self {
        let mut cam = Camera::new(aspect_ratio);
        cam.pos -= Vec3::new(0.0, 0.0, 10.0);
        Self {
            cam,
            disable: false,
        }
    }
}

impl AppCamera for FreeCamera {
    fn on_update(&mut self, input: &Input) {
        if self.disable {
            return;
        }
        let mut mov_speed = 0.05;
        let roll_speed = 1.0;
        if input.is_pressed_repeat(VirtualKeyCode::LShift) {
            mov_speed = 0.1;
        }

        let forward = self.cam.forward() * mov_speed;
        let right = self.cam.right() * mov_speed;

        if input.is_pressed_repeat(VirtualKeyCode::W) {
            self.cam.move_pos(&forward);
        }

        if input.is_pressed_repeat(VirtualKeyCode::S) {
            self.cam.move_pos(&-forward);
        }

        if input.is_pressed_repeat(VirtualKeyCode::A) {
            self.cam.move_pos(&-right);
        }

        if input.is_pressed_repeat(VirtualKeyCode::D) {
            self.cam.move_pos(&right);
        }

        if input.is_pressed_repeat(VirtualKeyCode::Q) {
            self.cam.roll(f32::to_radians(roll_speed));
        }

        if input.is_pressed_repeat(VirtualKeyCode::E) {
            self.cam.roll(-f32::to_radians(roll_speed));
        }

        let mouse_motion = input.get_mouse_motion();
        if mouse_motion != (0.0, 0.0) {
            self.cam.yaw(f32::to_radians(0.5 * mouse_motion.0 as f32));

            self.cam.pitch(f32::to_radians(0.5 * mouse_motion.1 as f32));
        }
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
