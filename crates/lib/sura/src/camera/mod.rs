mod camera;
mod free_camera;
mod orbit_camera;

pub use free_camera::*;
pub use orbit_camera::*;

pub trait AppCamera {
    fn on_update(&mut self, input: &crate::input::Input);
    fn projection(&self) -> glam::Mat4;
    fn view(&self) -> glam::Mat4;

    fn lock_camera(&mut self, b: bool);
    fn get_lock_camera(&self) -> bool;
}
