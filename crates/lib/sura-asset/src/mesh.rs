use custom_error::custom_error;
use glam::{Mat4, Vec3, Vec4};

use log::{error, info, trace, warn};
use memmap2::Mmap;
use rkyv::{Archive, Deserialize, Serialize};
use std::{
    fs::{self},
    mem::ManuallyDrop,
    ops::Deref,
    path::Path,
};

enum UriType {
    URI,
    URIDATA,
}

struct DataUri<'a> {
    mime_type: &'a str,
    kind: UriType,
    data: &'a str,
}

fn split_once(input: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut iter = input.splitn(2, delimiter);
    Some((iter.next()?, iter.next()?))
}

impl<'a> DataUri<'a> {
    const VALID_MIME_TYPES: &'a [&'a str] =
        &["application/octet-stream", "application/gltf-buffer"];

    fn parse(uri: &'a str) -> DataUri<'a> {
        if let Some(uri) = uri.strip_prefix("data:") {
            let (mime_type, data) = split_once(uri, ',').unwrap();

            if let Some(mime_type) = mime_type.strip_suffix(";base64") {
                DataUri {
                    mime_type,
                    kind: UriType::URIDATA,
                    data,
                }
            } else {
                panic!("URI data needs to be base64 encoded :{}", uri);
            }
        } else {
            DataUri {
                mime_type: "",
                kind: UriType::URI,
                data: uri,
            }
        }
    }

    fn decode(&self, parent_path: &std::path::Path) -> Result<Vec<u8>, GltfError> {
        match self.kind {
            UriType::URI => {
                let data_path = std::path::Path::new(parent_path.parent().unwrap()).join(self.data);
                Ok(fs::read(data_path).unwrap_or_else(|_| {
                    panic!("couldn't open file:{}", parent_path.to_str().unwrap())
                }))
            }
            UriType::URIDATA => {
                if DataUri::VALID_MIME_TYPES.contains(&self.mime_type) {
                    match base64::decode(self.data) {
                        Ok(d) => Ok(d),
                        Err(err) => Err(GltfError::Base64Decode { base64error: err }),
                    }
                } else {
                    Err(GltfError::MimeErr {
                        mime_type: self.mime_type.into(),
                    })
                }
            }
        }
    }
}

custom_error! {GltfError
    Base64Decode{base64error:base64::DecodeError} = "failed to decode base64:{base64error}",
    MimeErr{mime_type:String}= "Mime type:'{mime_type}' is not supported",
    SizeMismatch = "buffer size doesn't match byteLength",
    MissingBlob = "Blob is missing from gltf!",

}

// enum VertexAttributeValues {
//     F32(Vec<f32>),
//     F32x2(Vec<[f32; 2]>),
//     F32x3(Vec<[f32; 3]>),
//     F32x4(Vec<[f32; 4]>),
// }

// impl VertexAttributeValues {
//     // get att size per vertex in bytes
//     pub fn get_element_size(&self) -> usize {
//         match self {
//             VertexAttributeValues::F32(v) => VertexAttributeValues::size_of_vec_element(v),
//             VertexAttributeValues::F32x2(v) => VertexAttributeValues::size_of_vec_element(v),
//             VertexAttributeValues::F32x3(v) => VertexAttributeValues::size_of_vec_element(v),
//             VertexAttributeValues::F32x4(v) => VertexAttributeValues::size_of_vec_element(v),
//         }
//     }

//     pub fn len(&self) -> usize {
//         match self {
//             VertexAttributeValues::F32(v) => v.len(),
//             VertexAttributeValues::F32x2(v) => v.len(),
//             VertexAttributeValues::F32x3(v) => v.len(),
//             VertexAttributeValues::F32x4(v) => v.len(),
//         }
//     }

//     pub fn get_bytes(&self) -> &[u8] {
//         unsafe {
//             match self {
//                 VertexAttributeValues::F32(v) => {
//                     let n_bytes = v.len() * std::mem::size_of::<f32>();
//                     slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
//                 }
//                 VertexAttributeValues::F32x2(v) => {
//                     let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
//                     slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
//                 }
//                 VertexAttributeValues::F32x3(v) => {
//                     let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
//                     slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
//                 }
//                 VertexAttributeValues::F32x4(v) => {
//                     let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
//                     slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
//                 }
//             }
//         }
//     }

//     fn size_of_vec_element<T>(_v: &Vec<T>) -> usize {
//         mem::size_of::<T>()
//     }
// }

// #[derive(Debug, Clone)]
// enum Indices {
//     None,
//     U32(Vec<u32>),
//     U16(Vec<u16>),
//     U8(Vec<u8>),
// }

// struct Renderable {
//     pub meshes: Vec<Mesh>,
//     pub textures: Vec<image::DynamicImage>,
// }

// struct Mesh {
//     pub index_buffer: Indices,
//     pub vertex_attributes: IndexMap<&'static str, VertexAttributeValues>,
// }

// impl Mesh {
//     pub const ATT_POSITION: &'static str = "vertex_pos";
//     pub const ATT_UV: &'static str = "vertex_uv";
//     pub const ATT_NORMAL: &'static str = "vertex_normal";
//     pub const ATT_TANGENT: &'static str = "vertex_tangent";

//     pub fn new() -> Self {
//         Mesh {
//             index_buffer: Indices::None,
//             vertex_attributes: IndexMap::new(),
//         }
//     }

//     pub fn set_attribute(&mut self, name: &'static str, val: VertexAttributeValues) {
//         self.vertex_attributes.insert(name, val);
//     }

//     pub fn stride(&self) -> usize {
//         trace!(
//             "self.vertex_attributes :{:?}",
//             self.vertex_attributes.keys()
//         );
//         let mut stride = 0usize;
//         for att in self.vertex_attributes.values() {
//             stride += att.get_element_size();
//         }
//         stride
//     }

//     pub fn vertex_count(&self) -> usize {
//         let mut v_count: Option<usize> = None;

//         for (name, att) in &self.vertex_attributes {
//             let att_len = att.len();
//             if let Some(prev_count) = v_count {
//                 assert_eq!(prev_count,att_len, "Attribute `{}` has a different vertex count than other attributes , expected:{} , got:{}" ,name ,prev_count ,att_len  );
//             }
//             v_count = Some(att_len);
//         }

//         v_count.unwrap_or(0)
//     }

//     pub fn get_vertex_buffer(&self) -> Vec<u8> {
//         let vertex_size = self.stride();
//         let vertex_count = self.vertex_count();

//         let mut buff = vec![0; vertex_count * vertex_size];

//         let mut att_offset = 0;

//         for att in self.vertex_attributes.values() {
//             let attributes_bytes = att.get_bytes();
//             let att_size = att.get_element_size();

//             for (vertex_index, att_data) in attributes_bytes.chunks_exact(att_size).enumerate() {
//                 let offset = vertex_index * vertex_size + att_offset;
//                 buff[offset..offset + att_size].copy_from_slice(att_data);
//             }
//             att_offset += att_size;
//         }

//         buff
//     }

//     pub fn get_vertex_attribute(&self, att_name: &str) -> &VertexAttributeValues {
//         self.vertex_attributes
//             .get(att_name)
//             .expect("vertex attribute name doesn't exist")
//     }
// }

fn load_gltf_material(
    mat: &gltf::material::Material,
    textures: &Vec<image::DynamicImage>,
) -> (Vec<MaterialMap>, MeshMaterial) {
    let base_color_factor: [f32; 4] = mat.pbr_metallic_roughness().base_color_factor();
    let roughness_factor: f32 = mat.pbr_metallic_roughness().roughness_factor();
    let metalness_factor: f32 = mat.pbr_metallic_roughness().metallic_factor();
    let emissive_factors: [f32; 3] = mat.emissive_factor();

    let mesh_material = MeshMaterial {
        base_color_factor,
        roughness_factor,
        metalness_factor,
        emissive_factors,
        maps_index: [0, 1, 2, 3, 4],
    };

    let albedo_map: MaterialMap = mat
        .pbr_metallic_roughness()
        .base_color_texture()
        .map_or_else(
            || {
                warn!("doesn't have base_color_texture(albedo) using default value");
                MaterialMap::create_placeholder([255, 255, 255, 255], "albedo_map".into(), 1000)
            },
            |tex| {
                if tex.texture_transform().is_some() {
                    warn!("we don't use texture transform");
                }

                let raw_texture = textures[tex.texture().source().index()].to_rgba8();

                let source = RawRgba8Image {
                    source: raw_texture.to_vec(),
                    dimentions: [raw_texture.width(), raw_texture.height()],
                };
                MaterialMap {
                    source,
                    params: TextureParams {
                        gamma: TextureGamma::Srgb,
                        mips: true,
                    },
                    source_index: tex.texture().source().index() as u32,
                    name: "albedo_map".into(),
                }
            },
        );

    let normal_map: MaterialMap = mat.normal_texture().map_or_else(
        || {
            warn!("doesn't have normal_texture using default value");
            MaterialMap::create_placeholder([127, 127, 255, 255], "normal_map".into(), 999)
        },
        |tex| {
            let raw_texture = textures[tex.texture().source().index()].to_rgba8();

            let source = RawRgba8Image {
                source: raw_texture.to_vec(),
                dimentions: [raw_texture.width(), raw_texture.height()],
            };
            MaterialMap {
                source,
                params: TextureParams {
                    gamma: TextureGamma::Linear,
                    mips: true,
                },
                source_index: tex.texture().source().index() as u32,
                name: "normal_map".into(),
            }
        },
    );

    let spec_map: MaterialMap = mat
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map_or_else(
            || {
                warn!("doesn't have metallic_roughness_texture(specular) using default value");
                MaterialMap::create_placeholder([127, 127, 255, 255], "spec_map".into(), 998)
            },
            |tex| {
                let raw_texture = textures[tex.texture().source().index()].to_rgba8();

                let source = RawRgba8Image {
                    source: raw_texture.to_vec(),
                    dimentions: [raw_texture.width(), raw_texture.height()],
                };
                MaterialMap {
                    source,
                    params: TextureParams {
                        gamma: TextureGamma::Linear,
                        mips: true,
                    },
                    source_index: tex.texture().source().index() as u32,
                    name: "spec_map".into(),
                }
            },
        );

    let emissive_map: MaterialMap = mat.emissive_texture().map_or_else(
        || {
            warn!("doesn't have emissive_map using default value");
            MaterialMap::create_placeholder([255, 255, 255, 255], "emissive_map".into(), 997)
        },
        |tex| {
            let raw_texture = textures[tex.texture().source().index()].to_rgba8();

            let source = RawRgba8Image {
                source: raw_texture.to_vec(),
                dimentions: [raw_texture.width(), raw_texture.height()],
            };
            MaterialMap {
                source,
                params: TextureParams {
                    gamma: TextureGamma::Linear,
                    mips: true,
                },
                source_index: tex.texture().source().index() as u32,
                name: "emissive_map".into(),
            }
        },
    );

    let occlusion_map: MaterialMap = mat.occlusion_texture().map_or_else(
        || {
            warn!("doesn't have occlusion_map using default value");
            MaterialMap::create_placeholder([255, 255, 255, 255], "occlusion_map".into(), 996)
        },
        |tex| {
            let raw_texture = textures[tex.texture().source().index()].to_rgba8();
            let source = RawRgba8Image {
                source: raw_texture.to_vec(),
                dimentions: [raw_texture.width(), raw_texture.height()],
            };
            MaterialMap {
                source,
                params: TextureParams {
                    gamma: TextureGamma::Linear,
                    mips: true,
                },
                source_index: tex.texture().source().index() as u32,
                name: "occlusion_map".into(),
            }
        },
    );

    let material_maps: Vec<MaterialMap> = vec![
        normal_map,
        spec_map,
        albedo_map,
        emissive_map,
        occlusion_map,
    ];
    (material_maps, mesh_material)
}

fn parse_node(
    node: &gltf::scene::Node,
    transform: Mat4,
    buffers: &Vec<Vec<u8>>,
    textures: &Vec<image::DynamicImage>,
    out: &mut TriangleMesh,
) {
    if let Some(mesh) = node.mesh() {
        info!("processing mesh : {:}", mesh.name().unwrap_or_default());

        let flip_winding_order = transform.determinant() < 0.0;

        if flip_winding_order {
            error!("we don't support flip_winding_order")
        }
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            //collect positions (required)
            let positions: Vec<[f32; 3]> = if let Some(pos_iter) = reader.read_positions() {
                pos_iter.collect()
            } else {
                warn!("no positions available (required)");
                return;
            };

            //collect indices
            let mut indices: Vec<u32> = {
                if let Some(indices_r) = reader.read_indices() {
                    indices_r.into_u32().collect()
                } else {
                    warn!("no indices available, using positions");
                    (0..positions.len() as u32).collect()
                }
            };

            //collect uvs
            let mut uvs: Vec<[f32; 2]> = {
                if let Some(indices_r) = reader.read_tex_coords(0) {
                    indices_r.into_f32().collect()
                } else {
                    warn!("no uvs available, using default values");
                    vec![[0.0, 0.0]; positions.len()]
                }
            };

            //collect normals
            let normals: Vec<[f32; 3]> = {
                if let Some(indices_r) = reader.read_normals() {
                    indices_r.collect()
                } else {
                    warn!("no normals available, using default values");
                    return;
                }
            };

            //collect tangents
            let tangents: Vec<[f32; 4]> = {
                if let Some(val) = reader.read_tangents() {
                    val.collect()
                } else {
                    warn!("no tangents available,manually calculating tangents..");
                    let mut tangents = vec![[1.0, 0.0, 0.0, 0.0]; positions.len()];

                    mikktspace::generate_tangents(&mut TangentCalcContext {
                        indices: indices.as_slice(),
                        positions: positions.as_slice(),
                        normals: normals.as_slice(),
                        uvs: uvs.as_slice(),
                        tangents: tangents.as_mut_slice(),
                    });

                    tangents
                }
            };

            //collect colors
            let mut colors: Vec<[f32; 4]> = {
                if let Some(indices_r) = reader.read_colors(0) {
                    indices_r.into_rgba_f32().collect()
                } else {
                    warn!("no colors available, using default values");
                    vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                }
            };

            //collect material
            let cur_material_index = out.materials.len() as u32;
            // Collect material ids
            let mut material_ids = vec![cur_material_index; positions.len()];

            let (material_maps, mut material) = load_gltf_material(&primitive.material(), textures);
            {
                let mut map_base = out.maps.len() as u32;
                // only add unique maps
                {
                    for (map_index, map) in material_maps.iter().enumerate() {
                        match out
                            .maps
                            .iter_mut()
                            .position(|o_map| o_map.source_index == map.source_index)
                        {
                            Some(new_map_index) => {
                                material.maps_index[map_index] = new_map_index as u32;
                            }
                            None => {
                                material.maps_index[map_index] = map_base;
                                map_base += 1;
                                // info!(
                                //     " what map :{} , dims:{:?}, size:{}",
                                //     map.name,
                                //     map.source.dimentions,
                                //     map.source.source.len()
                                // );
                                out.maps.push(map.clone());
                            }
                        }
                    }
                };

                out.materials.push(material);
            }

            //write to out
            {
                let base_index = out.positions.len() as u32;
                for i in &mut indices {
                    *i += base_index;
                }
                out.indices.append(&mut indices);

                for v in positions {
                    let pos = (transform * Vec3::from(v).extend(1.0)).truncate();
                    out.positions.push(pos.into());
                }

                for v in normals {
                    let norm = (transform * Vec3::from(v).extend(0.0))
                        .truncate()
                        .normalize();
                    out.normals.push(norm.into());
                }

                for v in tangents {
                    let v = Vec4::from(v);
                    let t = (transform * v.truncate().extend(0.0))
                        .truncate()
                        .normalize();
                    out.tangents.push(t.extend(v.w).into());
                }

                out.uvs.append(&mut uvs);
                out.colors.append(&mut colors);
                out.material_ids.append(&mut material_ids);
            }
        }
    }
}

fn parse_gltf_node(
    node: &gltf::scene::Node,
    transform: Mat4,
    buffers: &Vec<Vec<u8>>,
    textures: &Vec<image::DynamicImage>,
    out: &mut TriangleMesh,
) {
    let transform = transform * Mat4::from_cols_array_2d(&node.transform().matrix());

    parse_node(node, transform, buffers, textures, out);

    for ref child in node.children() {
        parse_gltf_node(child, transform, buffers, textures, out);
    }
}

pub fn load_gltf(path: &str) -> Result<TriangleMesh, String> {
    //docs : https://www.khronos.org/files/gltf20-reference-guide.pdf
    let path = std::path::Path::new(path);
    let gltf = gltf::Gltf::open(path)
        .unwrap_or_else(|_| panic!("couldn't open gltf file:{}", path.to_str().unwrap()));
    let buffers = load_gltf_buffers(&gltf, path).unwrap();
    let textures = load_gltf_textures(&gltf, &buffers, path);

    if let Some(scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
        let mut out = TriangleMesh::default();

        let transform = glam::Mat4::IDENTITY;
        for ref node in scene.nodes() {
            parse_gltf_node(node, transform, &buffers, &textures, &mut out)
        }
        Ok(out)
    } else {
        Err(format!("no default scene found for {:?}", path.to_owned()))
    }
}

fn load_gltf_textures(
    gltf: &gltf::Gltf,
    buffers: &Vec<Vec<u8>>,
    path: &Path,
) -> Vec<image::DynamicImage> {
    let mut out_textures = Vec::with_capacity(gltf.textures().len());
    for texture in gltf.textures() {
        let tex = match texture.source().source() {
            gltf::image::Source::View { view, mime_type } => {
                let start = view.offset() as usize;
                let end = (view.offset() + view.length()) as usize;
                let buffer = &buffers[view.buffer().index()][start..end];

                image::load_from_memory_with_format(
                    buffer,
                    image::ImageFormat::from_extension(mime_type).unwrap_or_else(|| {
                        panic!(
                            "couldn't figure out extension for:{}, from:{:?} ",
                            mime_type,
                            view.name()
                        )
                    }),
                )
                .unwrap()
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                let uri = DataUri::parse(uri);

                let buf = uri.decode(path).unwrap();

                let mime = match mime_type {
                    Some(t) => t,
                    None => Path::new(uri.data).extension().unwrap().to_str().unwrap(),
                };

                //fix mime
                let mime = {
                    if mime.contains("png") {
                        "png"
                    } else if mime.contains("jpeg") || mime.contains("jpg") {
                        "jpeg"
                    } else if mime.contains("tga") {
                        "tga"
                    } else if mime.contains("hdr") {
                        "hdr"
                    } else {
                        mime
                    }
                };

                info!("{mime:?}");

                image::load_from_memory_with_format(
                    &buf,
                    image::ImageFormat::from_extension(mime).unwrap_or_else(|| {
                        panic!(
                            "couldn't figure out extension for:{}, from:{:?} ",
                            mime, uri.data
                        )
                    }),
                )
                .unwrap()
            }
        };

        out_textures.push(tex);
    }
    out_textures
}

fn load_gltf_buffers(gltf: &gltf::Gltf, path: &Path) -> Result<Vec<Vec<u8>>, GltfError> {
    // doc-> https://raw.githubusercontent.com/KhronosGroup/glTF/main/specification/2.0/figures/gltfOverview-2.0.0b.png

    let mut buffer_data: Vec<Vec<u8>> = Vec::new();

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => match gltf.blob.as_deref() {
                Some(blob) => buffer_data.push(blob.into()),
                None => return Err(GltfError::MissingBlob),
            },
            gltf::buffer::Source::Uri(uri) => {
                let uri = DataUri::parse(uri);

                let buf = match uri.decode(path) {
                    Ok(buff) => buff,
                    Err(err) => return Err(err),
                };

                if buffer.length() != buf.len() {
                    return Err(GltfError::SizeMismatch);
                }
                buffer_data.push(buf)
            }
        }
    }

    Ok(buffer_data)
}

#[derive(Default, Clone, Archive, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct RawRgba8Image {
    pub source: Vec<u8>,
    pub dimentions: [u32; 2],
}

#[derive(Archive, Clone, Deserialize, Serialize, Debug, PartialEq, Eq)]
#[archive(compare(PartialEq))]
pub enum TextureGamma {
    Linear,
    Srgb,
}

impl Default for TextureGamma {
    fn default() -> TextureGamma {
        TextureGamma::Linear
    }
}

#[derive(Default, Clone, Archive, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct TextureParams {
    pub gamma: TextureGamma,
    pub mips: bool,
}

#[derive(Default, Clone, Archive, Deserialize, Serialize, Debug, PartialEq)]
pub struct MaterialMap {
    pub source: RawRgba8Image,
    pub params: TextureParams,

    //used to sort map
    pub source_index: u32,
    //for debugging
    pub name: String,
}

impl MaterialMap {
    fn create_placeholder(color: [u8; 4], name: String, source_index: u32) -> MaterialMap {
        let mut mat = MaterialMap::default();
        mat.source.dimentions = [1, 1];
        mat.source.source = Vec::from(color);
        mat.source_index = source_index;
        mat.name = name;
        mat
    }
}

#[derive(Default, Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[repr(C)]
pub struct MeshMaterial {
    pub base_color_factor: [f32; 4],
    /*
    maps_index is:
    normal_map,
    spec_map,
    albedo_map,
    emissive_map,
    occlusion_map,
    */
    pub maps_index: [u32; 5],
    pub roughness_factor: f32,
    pub metalness_factor: f32,
    pub emissive_factors: [f32; 3],
}

impl MeshMaterial {
    #[allow(dead_code)]
    pub const NORMAL_MAP_INDEX: u32 = 0;
    #[allow(dead_code)]
    pub const SPECULAR_MAP_INDEX: u32 = 1;
    #[allow(dead_code)]
    pub const ALBEDO_MAP_INDEX: u32 = 2;
    #[allow(dead_code)]
    pub const EMISSIVE_MAP_INDEX: u32 = 3;
    #[allow(dead_code)]
    pub const OCCLUSION_MAP_INDEX: u32 = 4;
}

#[derive(Default, Archive, Deserialize, Serialize, Debug, PartialEq)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    pub tangents: Vec<[f32; 4]>,
    pub material_ids: Vec<u32>,
    pub materials: Vec<MeshMaterial>,
    pub maps: Vec<MaterialMap>,
}

pub struct LoadedTriangleMesh {
    pub mesh: &'static ArchivedTriangleMesh,
    pub mmap: ManuallyDrop<Mmap>,
}

impl Deref for LoadedTriangleMesh {
    type Target = ArchivedTriangleMesh;

    fn deref(&self) -> &'static Self::Target {
        self.mesh
    }
}

impl Drop for LoadedTriangleMesh {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.mmap);
        }
    }
}

struct TangentCalcContext<'a> {
    indices: &'a [u32],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tangents: &'a mut [[f32; 4]],
}

impl<'a> mikktspace::Geometry for TangentCalcContext<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
    }
}
