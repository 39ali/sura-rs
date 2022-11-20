use glam::*;

#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect_ratio: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

impl Camera {
    const UP: Vec3 = Vec3::Y;
    const FORWARD: Vec3 = Vec3::Z;
    const RIGHT: Vec3 = Vec3::X;
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            pos: Vec3::default(),
            fov: f32::to_radians(75.0),
            near: 0.01,
            far: 1000.0,
            aspect_ratio,
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
        }
    }

    pub fn projection(&self) -> Mat4 {
        let proj = glam::Mat4::perspective_lh(self.fov, self.aspect_ratio, self.near, self.far);

        // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/

        proj.mul_mat4(&glam::mat4(
            glam::vec4(1.0f32, 0.0, 0.0, 0.0),
            glam::vec4(0.0f32, -1.0, 0.0, 0.0),
            glam::vec4(0.0f32, 0.0, 1.0f32, 0.0),
            glam::vec4(0.0f32, 0.0, 0.0f32, 1.0),
        ))
    }

    pub fn move_pos(&mut self, dir: &Vec3) {
        self.pos += *dir;
    }

    pub fn rot(&self) -> Quat {
        let yaw = glam::Quat::from_axis_angle(Self::UP, self.yaw);
        let pitch = glam::Quat::from_axis_angle(Self::RIGHT, self.pitch);
        let roll = glam::Quat::from_axis_angle(Self::FORWARD, self.roll);

        let orientation = yaw * pitch * roll;

        orientation.normalize()
    }

    pub fn forward(&self) -> Vec3 {
        self.rot().mul_vec3(Self::FORWARD)
    }

    pub fn right(&self) -> Vec3 {
        self.rot().mul_vec3(Self::RIGHT)
    }
    pub fn up(&self) -> Vec3 {
        self.rot().mul_vec3(Self::UP)
    }
    #[allow(dead_code)]
    pub fn left(&self) -> Vec3 {
        -self.right()
    }
    #[allow(dead_code)]
    pub fn down(&self) -> Vec3 {
        -self.up()
    }
    // up and down
    pub fn pitch(&mut self, angle: f32) {
        self.pitch += angle;
    }
    // left and right
    pub fn yaw(&mut self, angle: f32) {
        self.yaw += angle;
    }
    // around the camera z axis
    pub fn roll(&mut self, angle: f32) {
        self.roll += angle;
    }
}
