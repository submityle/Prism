//! Reproducible CPU benchmark for the Mesh virtual-geometry build contracts.
//!
//! The benchmark intentionally reports the measured CPU portion only. GPU
//! timestamps, VRAM, upload bandwidth, page misses, and power belong to the
//! runtime capture protocol and cannot be inferred from an asset-build timer.

#![expect(
    clippy::print_stdout,
    reason = "the benchmark output is the measured report"
)]

use std::env;
use std::hint::black_box;
use std::time::Instant;

use prism_asset_build_core::{
    build_static_virtual_geometry, build_static_virtual_geometry_lod_chain,
    StaticClusterBuildSettings, StaticVirtualGeometryLodSet,
};
use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues, POSITION_ATTRIBUTE};
use prism_render_architecture::virtual_geometry::{GeometryLodPolicy, LodProjection};
use prism_render_driver::{PrimitiveTopology, VertexFormat};

const DEFAULT_TRIANGLES: usize = 8_192;
const SAMPLES: usize = 24;

fn source_mesh(triangle_count: usize) -> Mesh {
    let mut positions = Vec::with_capacity(triangle_count * 3);
    let mut indices = Vec::with_capacity(triangle_count * 3);
    for triangle in 0..triangle_count {
        let base = (triangle * 3) as u32;
        let x = triangle as f32;
        positions.extend_from_slice(&[[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 0.5, 0.0]]);
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            VertexAttributeValues::Float32x3(positions),
        )
        .expect("benchmark positions fit")
        .indexed(Indices::Uint32(indices))
}

fn attributed_grid(side: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for y in 0..=side {
        for x in 0..=side {
            positions.push([x as f32, y as f32, 0.0]);
            uvs.push([x as f32 / side as f32, y as f32 / side as f32]);
        }
    }
    for y in 0..side {
        for x in 0..side {
            let a = y * (side + 1) + x;
            indices.extend([a, a + 1, a + side + 2, a, a + side + 2, a + side + 1]);
        }
    }
    let vertex_count = positions.len();
    Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            VertexAttributeValues::Float32x3(positions),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1),
            VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 1.0]; vertex_count]),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Uv", VertexFormat::Float32x2, 2),
            VertexAttributeValues::Float32x2(uvs),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Tangent", VertexFormat::Float32x4, 3),
            VertexAttributeValues::Float32x4(vec![[1.0, 0.0, 0.0, 1.0]; vertex_count]),
        )
        .unwrap()
        .indexed(Indices::Uint32(indices))
}

fn percentile(samples: &mut [u128], numerator: usize, denominator: usize) -> u128 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * numerator).div_ceil(denominator);
    samples[index]
}

fn main() {
    let triangles = env::var("PRISM_MESH_BENCH_TRIANGLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TRIANGLES);
    assert!(triangles > 0, "PRISM_MESH_BENCH_TRIANGLES must be positive");
    assert!(
        triangles <= u32::MAX as usize / 3,
        "triangle count exceeds index ABI"
    );
    let source = source_mesh(triangles);
    let settings = StaticClusterBuildSettings::default();

    let mut build_samples = [0_u128; SAMPLES];
    let mut cluster_count = 0_usize;
    let mut page_count = 0_usize;
    for sample in &mut build_samples {
        let start = Instant::now();
        let asset = build_static_virtual_geometry(7, &source, settings).expect("build asset");
        *sample = start.elapsed().as_nanos();
        cluster_count = asset.clusters.len();
        page_count = asset.pages.len();
        black_box(asset);
    }

    let grid_side: u32 = env::var("PRISM_MESH_BENCH_GRID_SIDE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    assert!(
        (8..=512).contains(&grid_side),
        "grid side must be in 8..=512"
    );
    let lod_source = attributed_grid(grid_side);
    let faces = grid_side * grid_side * 2;
    let lod_targets = [faces, faces / 2, faces / 4];
    // Warm up before timing. Validate achieved counts: disconnected triangle
    // soup can exercise the selector without ever exercising simplification.
    let lods = build_static_virtual_geometry_lod_chain(11, &lod_source, settings, &lod_targets)
        .expect("build LOD assets");
    let actual: Vec<_> = lods.iter().map(|level| level.achieved_triangles).collect();
    assert!(
        actual.windows(2).all(|p| p[1] < p[0]),
        "benchmark did not simplify"
    );
    let mut lod_build_samples = [0_u128; SAMPLES];
    for sample in &mut lod_build_samples {
        let start = Instant::now();
        let rebuilt =
            build_static_virtual_geometry_lod_chain(11, &lod_source, settings, &lod_targets)
                .expect("build attributed LOD assets");
        *sample = start.elapsed().as_nanos();
        assert_eq!(rebuilt, lods, "LOD artifact must be deterministic");
        black_box(rebuilt);
    }
    let lod_set = StaticVirtualGeometryLodSet::new(lods).expect("index LOD assets");
    let mut lod_samples = [0_u128; SAMPLES];
    for sample in &mut lod_samples {
        let start = Instant::now();
        for _ in 0..64 {
            black_box(lod_set.select(
                LodProjection::from_focal_length_pixels(1_000.0),
                10.0,
                1.0,
                GeometryLodPolicy {
                    target_error_pixels: 0.5,
                    hysteresis_pixels: 0.1,
                    prefetch_velocity_scale: 2.0,
                },
                None,
            ));
        }
        *sample = start.elapsed().as_nanos();
    }

    println!("prism_asset_build_core mesh_virtual_geometry");
    println!("  triangles         : {triangles}");
    println!("  clusters          : {cluster_count}");
    println!("  pages             : {page_count}");
    println!("  samples           : {SAMPLES}");
    println!("  lod grid side     : {grid_side}");
    println!("  lod targets       : {lod_targets:?}");
    println!("  lod achieved      : {actual:?}");
    println!(
        "  build p50 ns      : {}",
        percentile(&mut build_samples, 50, 100)
    );
    println!(
        "  build p95 ns      : {}",
        percentile(&mut build_samples, 95, 100)
    );
    println!(
        "  build p99 ns      : {}",
        percentile(&mut build_samples, 99, 100)
    );
    println!(
        "  lod build p50 ns  : {}",
        percentile(&mut lod_build_samples, 50, 100)
    );
    println!(
        "  lod build p95 ns  : {}",
        percentile(&mut lod_build_samples, 95, 100)
    );
    println!(
        "  lod build p99 ns  : {}",
        percentile(&mut lod_build_samples, 99, 100)
    );
    println!(
        "  select x64 p50 ns : {}",
        percentile(&mut lod_samples, 50, 100)
    );
    println!(
        "  select x64 p95 ns : {}",
        percentile(&mut lod_samples, 95, 100)
    );
    println!(
        "  select x64 p99 ns : {}",
        percentile(&mut lod_samples, 99, 100)
    );
    println!("  gpu fields        : runtime capture required");
}
