use std::cell::RefCell;
use std::collections::HashMap;

use log::trace;
use winit::event::DeviceEvent::MouseMotion;
use winit::event::{ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent};

#[derive(Debug)]
enum KeyState {
    Pressed,
    Released,
    Repeat,
}

#[derive(Default)]
pub struct Input {
    map: RefCell<HashMap<VirtualKeyCode, KeyState>>,
    //handles if key was pressed once
    map_once: RefCell<HashMap<VirtualKeyCode, KeyState>>,

    mouse_motion_delta: (f64, f64),
    mouse_wheel_delta: (f32, f32),
    mouse_button: HashMap<MouseButton, bool>,
}

impl Input {
    pub fn on_update(&mut self, event: &Event<()>) {
        // trace!("new event :{:?}", event);
        match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    let key = input.virtual_keycode.unwrap();
                    let state = match input.state {
                        ElementState::Pressed => {
                            if let Some(state) = self.map.borrow().get(&key) {
                                match state {
                                    KeyState::Pressed => KeyState::Repeat,
                                    KeyState::Released => KeyState::Pressed,
                                    KeyState::Repeat => KeyState::Repeat,
                                }
                            } else {
                                KeyState::Pressed
                            }
                        }
                        ElementState::Released => KeyState::Released,
                    };

                    self.map.borrow_mut().insert(key, state);

                    // map_once
                    {
                        let state = match input.state {
                            ElementState::Pressed => {
                                if let Some(state) = self.map_once.borrow().get(&key) {
                                    match state {
                                        KeyState::Pressed => KeyState::Repeat,
                                        KeyState::Released => KeyState::Pressed,
                                        KeyState::Repeat => KeyState::Repeat,
                                    }
                                } else {
                                    KeyState::Pressed
                                }
                            }
                            ElementState::Released => KeyState::Released,
                        };

                        self.map_once.borrow_mut().insert(key, state);
                    }
                }

                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => {
                        self.mouse_wheel_delta = (*x, *y)
                    }
                    _ => {}
                },

                WindowEvent::MouseInput { button, state, .. } => {
                    self.mouse_button
                        .insert(*button, *state == ElementState::Pressed);
                }

                _ => (),
            },

            Event::DeviceEvent { ref event, .. } => match event {
                MouseMotion { delta } => {
                    self.mouse_motion_delta = *delta;
                }
                _ => {}
            },

            Event::RedrawEventsCleared => {
                self.on_clear();
            }

            _ => {}
        };
    }

    pub fn on_clear(&mut self) {
        for state in self.map_once.borrow_mut().values_mut() {
            match state {
                KeyState::Pressed => *state = KeyState::Repeat,
                _ => {}
            }
        }
        self.mouse_motion_delta = (0.0, 0.0);
        self.mouse_wheel_delta = (0.0, 0.0);
    }

    // only register one key press
    pub fn is_pressed(&self, key: VirtualKeyCode) -> bool {
        if let Some(state) = self.map_once.borrow_mut().get(&key) {
            match state {
                KeyState::Pressed => true,
                KeyState::Released => false,
                KeyState::Repeat => false,
            }
        } else {
            false
        }
    }
    // register repeats as true
    pub fn is_pressed_repeat(&self, key: VirtualKeyCode) -> bool {
        if let Some(state) = self.map.borrow_mut().get(&key) {
            match state {
                KeyState::Pressed => true,
                KeyState::Released => false,
                KeyState::Repeat => true,
            }
        } else {
            false
        }
    }

    pub fn is_released(&self, key: VirtualKeyCode) -> bool {
        let mut map = self.map.borrow_mut();
        if let Some(state) = map.get(&key) {
            match state {
                KeyState::Pressed => false,
                KeyState::Repeat => false,
                KeyState::Released => {
                    map.remove(&key);
                    true
                }
            }
        } else {
            false
        }
    }

    pub fn is_mouse_pressed(&self, mouse: MouseButton) -> bool {
        if let Some(k) = self.mouse_button.get(&mouse) {
            *k
        } else {
            false
        }
    }

    pub fn get_mouse_motion(&self) -> (f64, f64) {
        self.mouse_motion_delta
    }

    pub fn get_mouse_wheel_motion(&self) -> (f32, f32) {
        self.mouse_wheel_delta
    }
}
