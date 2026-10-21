//! Physical-GPU parity and lifecycle coverage for toroidal SDF slots.
#![cfg(feature = "gpu")]

use std::sync::mpsc;

use prism_world_gpu::{
    GpuContext, GpuSdfError, GpuSdfQuery, GpuSdfQueryResult, GpuSdfToroidalLookup,
};
use prism_world_residency::{BrickCoord, SdfClipmapConfig, SparseSdfBrickGrid};

fn context() -> GpuContext {
    GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical SDF GPU acquisition failed")
    .expect("physical GPU required; no silent skip")
}

fn readback(
    context: &GpuContext,
    lookup: &mut GpuSdfToroidalLookup,
    queries: &[GpuSdfQuery],
) -> Vec<GpuSdfQueryResult> {
    lookup.write_queries(queries).unwrap();
    let bytes = (queries.len() * 16).max(16) as u64;
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF toroidal test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SDF toroidal test encoder"),
        });
    lookup.encode(&mut encoder).unwrap();
    encoder.copy_buffer_to_buffer(lookup.results(), 0, &staging, 0, queries.len() as u64 * 16);
    context.queue().submit([encoder.finish()]);
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let (sender, receiver) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    let mut results = Vec::with_capacity(queries.len());
    for (index, record) in mapped.chunks_exact(16).take(queries.len()).enumerate() {
        let status = u32::from_le_bytes(record[0..4].try_into().unwrap());
        let value = i32::from_le_bytes(record[4..8].try_into().unwrap());
        results.push(match status {
            0 if value == 0 => GpuSdfQueryResult::Unknown,
            1 if (i16::MIN as i32..=i16::MAX as i32).contains(&value) => {
                GpuSdfQueryResult::Value(value as i16)
            }
            _ => panic!("invalid toroidal GPU result record {index}: {status}/{value}"),
        });
    }
    drop(mapped);
    staging.unmap();
    results
}

fn query(voxel: [i32; 3], mip: u8) -> GpuSdfQuery {
    GpuSdfQuery { voxel, mip }
}

#[test]
fn toroidal_lookup_rolls_slots_and_explicit_dirty_uploads_match_cpu_grid() {
    let context = context();
    eprintln!("[prism_world_gpu::sdf_toroidal] {:?}", context.evidence());
    let config = SdfClipmapConfig::new(2, 1, 4096).unwrap();
    let mut lookup = GpuSdfToroidalLookup::new(&context, config, 2, 8, 4096).unwrap();
    let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
    grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![-9, 7, 1, 2, 3, 4, 5, 6])
        .unwrap();
    grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![3; 8])
        .unwrap();
    grid.insert_brick(BrickCoord::new(0, 0, 0, 1), vec![13; 8])
        .unwrap();

    assert_eq!(
        lookup
            .update_center(&grid, BrickCoord::new(0, 0, 0, 0))
            .unwrap(),
        624
    );
    assert_eq!(lookup.slot_count(), 18);
    assert_eq!(lookup.brick_count(), 3);
    assert_eq!(
        readback(
            &context,
            &mut lookup,
            &[
                query([-2, 0, 0], 0),
                query([-1, 0, 0], 0),
                query([0, 0, 0], 0),
                query([4, 0, 0], 0),
                query([0, 0, 0], 1),
            ],
        ),
        vec![
            GpuSdfQueryResult::Value(-9),
            GpuSdfQueryResult::Value(7),
            GpuSdfQueryResult::Value(3),
            GpuSdfQueryResult::Unknown,
            GpuSdfQueryResult::Value(13),
        ]
    );

    assert_eq!(
        lookup
            .update_center(&grid, BrickCoord::new(1, 0, 0, 0))
            .unwrap(),
        96
    );
    assert_eq!(lookup.center(), Some(BrickCoord::new(1, 0, 0, 0)));
    assert_eq!(lookup.brick_count(), 2);
    assert_eq!(
        readback(
            &context,
            &mut lookup,
            &[
                query([-2, 0, 0], 0),
                query([0, 0, 0], 0),
                query([4, 0, 0], 0)
            ],
        ),
        vec![
            GpuSdfQueryResult::Unknown,
            GpuSdfQueryResult::Value(3),
            GpuSdfQueryResult::Unknown,
        ]
    );

    let entering = BrickCoord::new(2, 0, 0, 0);
    grid.insert_brick(entering, vec![21; 8]).unwrap();
    assert_eq!(lookup.upload_bricks(&grid, &[entering]).unwrap(), 48);
    assert_eq!(lookup.brick_count(), 3);
    assert_eq!(
        readback(&context, &mut lookup, &[query([4, 1, 1], 0)]),
        vec![GpuSdfQueryResult::Value(21)]
    );

    assert!(grid.remove_brick(entering).unwrap());
    assert_eq!(lookup.upload_bricks(&grid, &[entering]).unwrap(), 32);
    assert_eq!(lookup.brick_count(), 2);
    assert_eq!(
        readback(&context, &mut lookup, &[query([4, 1, 1], 0)]),
        vec![GpuSdfQueryResult::Unknown]
    );
}

#[test]
fn toroidal_upload_validation_is_atomic_and_budgeted() {
    let context = context();
    let config = SdfClipmapConfig::new(1, 1, 1024).unwrap();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    let mut too_small = GpuSdfToroidalLookup::new(&context, config, 2, 1, 1).unwrap();
    assert!(matches!(
        too_small.update_center(&grid, BrickCoord::new(0, 0, 0, 0)),
        Err(GpuSdfError::UploadBudgetExceeded { .. })
    ));
    assert_eq!(too_small.center(), None);
    assert_eq!(too_small.brick_count(), 0);
    assert_eq!(too_small.last_upload_bytes(), 0);

    let mut lookup = GpuSdfToroidalLookup::new(&context, config, 2, 1, 2048).unwrap();
    lookup
        .update_center(&grid, BrickCoord::new(0, 0, 0, 0))
        .unwrap();
    let resident = BrickCoord::new(0, 0, 0, 0);
    let prior_upload = lookup.last_upload_bytes();
    assert!(matches!(
        lookup.upload_bricks(&grid, &[resident, resident]),
        Err(GpuSdfError::DuplicateBrickUpdate { .. })
    ));
    assert!(matches!(
        lookup.upload_bricks(&grid, &[BrickCoord::new(4, 0, 0, 0)]),
        Err(GpuSdfError::BrickOutsideClipmap { .. })
    ));
    assert_eq!(lookup.center(), Some(BrickCoord::new(0, 0, 0, 0)));
    assert_eq!(lookup.brick_count(), 0);
    assert_eq!(lookup.last_upload_bytes(), prior_upload);

    grid.insert_brick(resident, vec![5; 8]).unwrap();
    assert_eq!(lookup.upload_bricks(&grid, &[resident]).unwrap(), 48);
    assert_eq!(lookup.brick_count(), 1);
}
