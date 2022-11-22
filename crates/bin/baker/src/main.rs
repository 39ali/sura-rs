use std::{fs::File, io::Write, mem, path::PathBuf, time::Instant};

use env_logger::Env;
use log::info;
use sura_asset::mesh::*;

use clap::Parser;

use rkyv::Deserialize;

#[derive(Parser, Debug)]
#[clap(author, version, about ="bake .mesh files", long_about = None)]
struct Args {
    #[clap(short = 's', long)]
    scene: PathBuf,

    #[clap(short = 'o', long)]
    out: String,
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if args.scene.as_path().to_str().unwrap().is_empty() {
        panic!("scene arg is empty")
    }

    let start = Instant::now();

    info!("parsing {:?} ...", &args.scene);

    let tri_mesh = load_gltf(args.scene.to_str().unwrap())
        .expect("failed to parse gltf file into TriangleMesh");

    use rkyv::ser::{serializers::AllocSerializer, Serializer};

    let mut serializer = AllocSerializer::<0>::default();
    serializer.serialize_value(&tri_mesh).unwrap();
    let bytes = serializer.into_serializer().into_inner();

    std::fs::create_dir_all("baked").expect("failed to create baked dir!");

    let mut file = File::create(format!("baked/{}.mesh", args.out)).unwrap();
    file.write(&bytes[..]).expect("failed  to write mesh");

    let archived = unsafe { rkyv::archived_root::<TriangleMesh>(&bytes[..]) };
    let deserialized: TriangleMesh = archived.deserialize(&mut rkyv::Infallible).unwrap();

    let duration = start.elapsed();
    info!("done!, baking took: {:?}", duration);
    assert_eq!(deserialized, tri_mesh);

    info!(
        "vertices count:{} ,indices:{}",
        tri_mesh.positions.len(),
        tri_mesh.indices.len()
    );

    info!(
        "materials count:{}, size:{}mb",
        tri_mesh.materials.len(),
        to_mb((tri_mesh.materials.len() * mem::size_of::<MeshMaterial>()) as f32)
    );

    info!(
        "maps count:{}, size:{}mb",
        tri_mesh.maps.len(),
        tri_mesh.maps.iter().fold(0f32, |accum, map| {
            accum + to_mb(map.source.source.len() as f32)
        })
    );

    for (mat_indx, mat) in tri_mesh.materials.iter().enumerate() {
        log::debug!("mat [{:?}]", mat_indx);
        for map_inx in mat.maps_index.iter() {
            let map = &tri_mesh.maps[*map_inx as usize];
            log::debug!("map name :{}, dims :{:?}", map.name, map.source.dimentions);
        }
    }
}

fn to_mb(n: f32) -> f32 {
    n * 2f32.powi(-20)
}
