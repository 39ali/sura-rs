use custom_error::custom_error;
use indexmap::IndexMap;
use std::{fs, mem, path::Path, slice};

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

pub enum VertexAttributeValues {
    F32(Vec<f32>),
    F32x2(Vec<[f32; 2]>),
    F32x3(Vec<[f32; 3]>),
    F32x4(Vec<[f32; 4]>),
}

impl VertexAttributeValues {
    // get att size per vertex in bytes
    pub fn get_element_size(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x2(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x3(v) => VertexAttributeValues::size_of_vec_element(v),
            VertexAttributeValues::F32x4(v) => VertexAttributeValues::size_of_vec_element(v),
        }
    }

    // get att size for all vertices in bytes
    pub fn get_size(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => mem::size_of_val(&v),
            VertexAttributeValues::F32x2(v) => mem::size_of_val(&v),
            VertexAttributeValues::F32x3(v) => std::mem::size_of_val(&v),
            VertexAttributeValues::F32x4(v) => mem::size_of_val(&v),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            VertexAttributeValues::F32(v) => v.len(),
            VertexAttributeValues::F32x2(v) => v.len(),
            VertexAttributeValues::F32x3(v) => v.len(),
            VertexAttributeValues::F32x4(v) => v.len(),
        }
    }

    pub fn get_bytes(&self) -> &[u8] {
        unsafe {
            match self {
                VertexAttributeValues::F32(v) => {
                    let n_bytes = v.len() * std::mem::size_of::<f32>();
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x2(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x3(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
                VertexAttributeValues::F32x4(v) => {
                    let n_bytes = v.len() * VertexAttributeValues::size_of_vec_element(v);
                    slice::from_raw_parts((v.as_ptr()) as *const u8, n_bytes)
                }
            }
        }
    }

    fn size_of_vec_element<T>(_v: &Vec<T>) -> usize {
        mem::size_of::<T>()
    }
}

#[derive(Debug, Clone)]
pub enum Indices {
    None,
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

pub struct Renderable {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<image::DynamicImage>,
}

pub struct Mesh {
    pub index_buffer: Indices,
    pub vertex_attributes: IndexMap<&'static str, VertexAttributeValues>,
}

impl Mesh {
    pub const ATT_POSITION: &'static str = "vertex_pos";
    pub const ATT_UV: &'static str = "vertex_uv";
    pub const ATT_NORMAL: &'static str = "vertex_normal";
    pub const ATT_TANGENT: &'static str = "vertex_tangent";

    pub fn new() -> Self {
        Mesh {
            index_buffer: Indices::None,
            vertex_attributes: IndexMap::new(),
        }
    }

    pub fn set_attribute(&mut self, name: &'static str, val: VertexAttributeValues) {
        self.vertex_attributes.insert(name, val);
    }

    pub fn stride(&self) -> usize {
        let mut stride = 0usize;
        for att in self.vertex_attributes.values() {
            stride += att.get_element_size();
        }
        stride
    }

    pub fn vertex_count(&self) -> usize {
        let mut v_count: Option<usize> = None;

        for (name, att) in &self.vertex_attributes {
            let att_len = att.len();
            if let Some(prev_count) = v_count {
                assert_eq!(prev_count,att_len, "Attribute `{}` has a different vertex count than other attributes , expected:{} , got:{}" ,name ,prev_count ,att_len  );
            }
            v_count = Some(att_len);
        }

        v_count.unwrap_or(0)
    }

    pub fn get_buffer(&self) -> Vec<u8> {
        let vertex_size = self.stride();
        let vertex_count = self.vertex_count();

        let mut buff = vec![0; vertex_count * vertex_size];

        let mut att_offset = 0;

        for att in self.vertex_attributes.values() {
            let attributes_bytes = att.get_bytes();
            let att_size = att.get_element_size();

            for (vertex_index, att_data) in attributes_bytes.chunks_exact(att_size).enumerate() {
                let offset = vertex_index * vertex_size + att_offset;
                buff[offset..offset + att_size].copy_from_slice(att_data);
            }
            att_offset += att_size;
        }

        buff
    }
}

pub fn load_gltf(path: &str) -> Renderable {
    //docs : https://www.khronos.org/files/gltf20-reference-guide.pdf
    let path = std::path::Path::new(path);
    let gltf = gltf::Gltf::open(path)
        .unwrap_or_else(|_| panic!("couldn't open gltf file:{}", path.to_str().unwrap()));
    let buffers = load_buffers(&gltf, path).unwrap();

    let mut out_meshes: Vec<Mesh> = Vec::with_capacity(gltf.meshes().len());
    let mut out_textures = Vec::with_capacity(gltf.textures().len());
    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let mut out_mesh = Mesh::new();
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            if let Some(indices) = reader.read_indices().map(|v| match v {
                gltf::mesh::util::ReadIndices::U8(it) => Indices::U8(it.collect()),
                gltf::mesh::util::ReadIndices::U16(it) => Indices::U16(it.collect()),
                gltf::mesh::util::ReadIndices::U32(it) => Indices::U32(it.collect()),
            }) {
                out_mesh.index_buffer = indices;
            }

            if let Some(vertex_attribute) = reader
                .read_positions()
                .map(|v| VertexAttributeValues::F32x3(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_POSITION, vertex_attribute);
            }

            // TODO(ALI): support other uv types
            if let Some(vertex_attribute) = reader
                .read_tex_coords(0)
                .map(|v| VertexAttributeValues::F32x2(v.into_f32().collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_UV, vertex_attribute);
            }

            if let Some(vertex_attribute) = reader
                .read_normals()
                .map(|v| VertexAttributeValues::F32x3(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_NORMAL, vertex_attribute);
            }

            if let Some(vertex_attribute) = reader
                .read_tangents()
                .map(|v| VertexAttributeValues::F32x4(v.collect()))
            {
                out_mesh.set_attribute(Mesh::ATT_TANGENT, vertex_attribute);
            }

            // if let Some(vertex_attribute) = reader
            //     .read_colors()
            //     .map(|v| VertexAttributeValues::F32x3(v.collect()))
            // {
            //     out_mesh.set_attribute(Mesh::Att_Position, vertex_attribute);
            // }

            out_meshes.push(out_mesh);
        }
    }

    for texture in gltf.textures() {
        let tex = match texture.source().source() {
            gltf::image::Source::View { view, mime_type } => {
                let start = view.offset() as usize;
                let end = (view.offset() + view.length()) as usize;
                let buffer = &buffers[view.buffer().index()][start..end];

                image::load_from_memory_with_format(
                    buffer,
                    image::ImageFormat::from_extension(mime_type).unwrap_or_else(|| {
                        panic!("couldn't figure out extension for :{}", mime_type)
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

                image::load_from_memory_with_format(
                    &buf,
                    image::ImageFormat::from_extension(mime)
                        .unwrap_or_else(|| panic!("couldn't figure out extension for :{}", mime)),
                )
                .unwrap()
            }
        };

        println!(
            "images: {} , {}  ",
            texture.name().unwrap_or("no-name"),
            texture.index(),
            // texture.sampler()
        );

        out_textures.push(tex);
    }

    Renderable {
        meshes: out_meshes,
        textures: out_textures,
    }
}

fn load_buffers(gltf: &gltf::Gltf, path: &Path) -> Result<Vec<Vec<u8>>, GltfError> {
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
