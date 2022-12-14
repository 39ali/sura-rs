use glam::Vec3;

#[allow(dead_code)]
#[derive(Default, Debug)]
#[repr(C)]
pub struct FrameConstant {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub cam_pos: [f32; 3],
    pub light_count: i32,
}
#[allow(dead_code)]
#[derive(Default, Debug)]
#[repr(C)]
pub struct GpuMesh {
    pub pos_offset: u32,
    pub uv_offset: u32,
    pub normal_offset: u32,
    pub colors_offset: u32,
    pub tangents_offset: u32,
    pub materials_data_offset: u32,
    pub material_ids_offset: u32,
}

impl crate::buffer::BufferData for GpuMesh {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<GpuMesh>(),
            )
        }
    }

    fn bsize(&self) -> usize {
        std::mem::size_of::<GpuMesh>()
    }
}

#[repr(C)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

pub struct Light {
    pub ty: LightType,
    pub pos: glam::Vec3,
    pub direction: glam::Vec3,
    pub color: glam::Vec3,
    pub intensity: f32,
    pub inv_radius: f32,
    pub angle_scale: f32,
    pub angle_offset: f32,
}

impl Light {
    pub fn create_directional_light(direction: &Vec3, illuminance: f32, color: &Vec3) -> Self {
        Light {
            ty: LightType::Directional,
            pos: Vec3::ZERO,
            direction: direction.normalize(),
            intensity: illuminance, //Illuminance(lx)
            color: *color,
            inv_radius: 0.,
            angle_scale: 0.,
            angle_offset: 0.,
        }
    }

    pub fn create_point_light(pos: &Vec3, luminous_power: f32, color: &Vec3, radius: f32) -> Self {
        Light {
            ty: LightType::Point,
            pos: *pos,
            direction: Vec3::ZERO,
            intensity: luminous_power / (4.0 * std::f32::consts::PI), //Candela (cd)
            color: *color,
            inv_radius: 1.0 / f32::max(radius, 1e-2),
            angle_scale: 0.,
            angle_offset: 1.,
        }
    }

    /// angle : in radians
    pub fn create_spot_light(
        pos: &Vec3,
        direction: &Vec3,
        luminous_power: f32,
        color: &Vec3,
        radius: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        let cos_inner = f32::cos(inner_angle);
        let cos_outer = f32::cos(outer_angle);
        let angle_scale = 1.0 / f32::max(1e-4, cos_inner - cos_outer);
        let angle_offset = -cos_outer * angle_scale;

        Light {
            ty: LightType::Spot,
            pos: *pos,
            direction: direction.normalize(),
            intensity: luminous_power / (std::f32::consts::PI), //Candela (cd)
            color: *color,
            inv_radius: 1.0 / f32::max(radius, 1e-2),
            angle_scale,
            angle_offset,
        }
    }
}

struct LightHandle(u16);
