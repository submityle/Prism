//! Physical-GPU parity for the sparse SDF brick lookup contract.
#![cfg(feature = "gpu")]

use prism_world_gpu::{GpuContext, GpuSdfError, GpuSdfLookup, GpuSdfQuery, GpuSdfQueryResult};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

fn context() -> GpuContext {
    GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical SDF GPU acquisition failed")
    .expect("physical GPU required; no silent skip")
}

fn grid() -> SparseSdfBrickGrid {
    let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
    grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![-9, 7, 1, 2, 3, 4, 5, 6])
        .unwrap();
    grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![3; 8])
        .unwrap();
    grid
}

#[test]
fn gpu_lookup_matches_cpu_samples_and_preserves_unknown() {
    let context = context();
    eprintln!("[prism_world_gpu::sdf] {:?}", context.evidence());
    let mut lookup = GpuSdfLookup::new(&context, 4, 8, 2, 2).unwrap();
    let grid = grid();
    lookup.upload_grid(&grid).unwrap();
    assert_eq!(lookup.brick_count(), 2);

    let queries = [
        GpuSdfQuery {
            voxel: [-2, 0, 0],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [-1, 1, 1],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [0, 0, 0],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [4, 0, 0],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [0, 0, 0],
            mip: 1,
        },
    ];
    let evaluation = lookup.evaluate_with_readback(&context, &queries).unwrap();
    assert_eq!(
        evaluation.results,
        vec![
            GpuSdfQueryResult::Value(-9),
            GpuSdfQueryResult::Value(6),
            GpuSdfQueryResult::Value(3),
            GpuSdfQueryResult::Unknown,
            GpuSdfQueryResult::Unknown,
        ]
    );
    assert_eq!(evaluation.telemetry.brick_count, 2);
    assert_eq!(evaluation.telemetry.query_count, queries.len());
    assert_eq!(evaluation.telemetry.brick_upload_bytes, 2 * (16 + 16 + 16));
    assert_eq!(
        evaluation.telemetry.query_upload_bytes,
        queries.len() as u64 * 16
    );
    assert_eq!(
        evaluation.telemetry.readback_bytes,
        queries.len() as u64 * 16
    );
    assert_eq!(evaluation.telemetry.dispatch_workgroups, 1);
}

#[test]
fn invalid_grid_does_not_replace_uploaded_pool() {
    let context = context();
    let mut lookup = GpuSdfLookup::new(&context, 2, 1, 2, 2).unwrap();
    lookup.upload_grid(&grid()).unwrap();
    let wrong = SparseSdfBrickGrid::new(4, 2).unwrap();
    assert!(matches!(
        lookup.upload_grid(&wrong),
        Err(GpuSdfError::GridMismatch { .. })
    ));
    assert_eq!(lookup.brick_count(), 2);
}
