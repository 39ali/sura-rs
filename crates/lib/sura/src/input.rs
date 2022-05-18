use std::collections::HashMap;

use winit::event::DeviceEvent::MouseMotion;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
#[derive(Default)]
pub struct Input {
    map: HashMap<VirtualKeyCode, bool>,
    mouse_pos_delta: (f64, f64),
}

impl Input {
    pub fn on_update(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    self.map.insert(
                        input.virtual_keycode.unwrap(),
                        input.state == ElementState::Pressed,
                    );
                }

                _ => (),
            },

            Event::DeviceEvent { ref event, .. } => match event {
                MouseMotion { delta } => {
                    self.mouse_pos_delta = *delta;
                }
                _ => {}
            },

            _ => (),
        };
    }

    pub fn on_clear(&mut self) {
        self.mouse_pos_delta = (0.0, 0.0);
    }
    pub fn is_pressed(&self, key: VirtualKeyCode) -> bool {
        if let Some(k) = self.map.get(&key) {
            *k
        } else {
            false
        }
    }

    pub fn get_mouse_motion(&self) -> (f64, f64) {
        self.mouse_pos_delta
    }
}
