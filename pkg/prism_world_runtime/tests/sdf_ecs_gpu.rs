//! End-to-end ECS request, physical GPU dispatch, readback, and publication.
#![cfg(feature = "sdf_ecs_gpu")]

use prism_ecs::prelude::World;
use prism_world_gpu::{GpuAcquireError, GpuContext};
use prism_world_residency::{BrickCoord, SdfClipmapConfig, SparseSdfBrickGrid};
use prism_world_runtime::{
    WorldSdfClipmapConfig, WorldSdfEcsGpuBridge, WorldSdfEcsGpuConfig, WorldSdfEcsGpuError,
    WorldSdfEcsGpuPolicy, WorldSdfEcsRequest, WorldSdfEcsSample, WorldSdfEcsSampleValue,
    WorldSdfFocusConfig,
};

fn context() -> Option<GpuContext> {
    let context = match GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    }) {
        Ok(Some(context)) => context,
        Ok(None) | Err(GpuAcquireError::Adapter(_)) | Err(GpuAcquireError::NonHardware(_)) => {
            eprintln!("physical SDF ECS GPU unavailable; skipping hardware coverage");
            return None;
        }
        Err(error) => panic!("physical SDF ECS GPU acquisition failed: {error}"),
    };
    assert!(
        context.evidence().is_hardware(),
        "software adapter is not coverage"
    );
    eprintln!(
        "[prism_world_runtime::sdf_ecs_gpu] {:?}",
        context.evidence()
    );
    Some(context)
}

fn config() -> WorldSdfEcsGpuConfig {
    WorldSdfEcsGpuConfig {
        focus: WorldSdfFocusConfig::new([0; 3], 1, 1, [0; 3]).unwrap(),
        clipmap: WorldSdfClipmapConfig::new(
            SdfClipmapConfig::new(2, 1, 1 << 20).unwrap(),
            18,
            4096,
        )
        .unwrap(),
        max_queries: 16,
        max_upload_bytes: 4096,
    }
}

fn brick(x: i32, z: i32, mip: u8) -> BrickCoord {
    BrickCoord::new(x, 0, z, mip)
}

fn grid() -> SparseSdfBrickGrid {
    let mut grid = SparseSdfBrickGrid::new(1, 2).unwrap();
    for x in -1..=1 {
        for z in -1..=1 {
            grid.insert_brick(brick(x, z, 0), vec![(x * 10 + z) as i16])
                .unwrap();
        }
    }
    grid.insert_brick(brick(0, 0, 1), vec![i16::MIN]).unwrap();
    grid
}

fn request(voxel: [i32; 3], mip: u8, request_id: u64) -> WorldSdfEcsRequest {
    WorldSdfEcsRequest {
        voxel,
        mip,
        request_id,
    }
}

#[test]
fn physical_gpu_evidence_is_required_and_skip_policy_is_explicit() {
    let required = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    };
    match WorldSdfEcsGpuBridge::from_headless(
        WorldSdfEcsGpuPolicy::SkipUnavailable,
        required.clone(),
        config(),
    ) {
        Ok(Some(bridge)) => assert!(bridge.focus().ring().allocated_buffer_bytes() > 0),
        Ok(None) => {
            let strict = WorldSdfEcsGpuBridge::from_headless(
                WorldSdfEcsGpuPolicy::Strict,
                required,
                config(),
            );
            assert!(matches!(strict, Err(WorldSdfEcsGpuError::NoPhysicalGpu)));
        }
        Err(error) => panic!("unexpected policy result: {error}"),
    }
}

#[test]
fn ecs_requests_dispatch_readback_publish_unknown_and_multimip_values() {
    let Some(context) = context() else { return };
    let mut bridge = WorldSdfEcsGpuBridge::new(context, config()).unwrap();
    let grid = grid();
    bridge.update_focus(&grid, [0; 3]).unwrap();

    let mut world = World::new();
    let min_entity = world.spawn(request([0, 0, 0], 1, 101));
    let max_entity = world.spawn(request([1, 0, 0], 0, 102));
    let unknown_entity = world.spawn(request([4, 0, 0], 0, 103));
    let missing_entity = world.spawn(request([0, 0, 1], 1, 104));

    let report = bridge.sample_world(&mut world).unwrap();
    assert_eq!(report.sampled_entities, 4);
    assert_eq!(
        world.get::<WorldSdfEcsSample>(min_entity),
        Some(&WorldSdfEcsSample {
            request_id: 101,
            value: WorldSdfEcsSampleValue::Value(i16::MIN),
        })
    );
    assert_eq!(
        world.get::<WorldSdfEcsSample>(max_entity),
        Some(&WorldSdfEcsSample {
            request_id: 102,
            value: WorldSdfEcsSampleValue::Value(10),
        })
    );
    assert_eq!(
        world.get::<WorldSdfEcsSample>(unknown_entity),
        Some(&WorldSdfEcsSample {
            request_id: 103,
            value: WorldSdfEcsSampleValue::Unknown,
        })
    );
    assert_eq!(
        world.get::<WorldSdfEcsSample>(missing_entity),
        Some(&WorldSdfEcsSample {
            request_id: 104,
            value: WorldSdfEcsSampleValue::Unknown,
        })
    );
}

#[test]
fn dirty_fence_requires_flush_and_entity_recycle_does_not_reuse_old_result() {
    let Some(context) = context() else { return };
    let mut bridge = WorldSdfEcsGpuBridge::new(context, config()).unwrap();
    let mut grid = grid();
    bridge.update_focus(&grid, [0; 3]).unwrap();

    let mut world = World::new();
    let old = world.spawn(request([0, 0, 0], 0, 1));
    bridge.sample_world(&mut world).unwrap();
    assert_eq!(
        world.get::<WorldSdfEcsSample>(old).unwrap().value,
        WorldSdfEcsSampleValue::Value(0)
    );
    assert!(world.despawn(old));

    grid.insert_brick(brick(0, 0, 0), vec![i16::MAX]).unwrap();
    bridge.mark_dirty(&[brick(0, 0, 0)]).unwrap();
    world.spawn(request([0, 0, 0], 0, 2));
    assert!(matches!(
        bridge.sample_world(&mut world),
        Err(WorldSdfEcsGpuError::Submit(
            prism_world_runtime::WorldSdfGpuClipmapError::PendingDirty { .. }
        ))
    ));

    bridge.flush_dirty(&grid, 1, 4096).unwrap();
    let recycled = world.spawn(request([0, 0, 0], 0, 3));
    bridge.sample_world(&mut world).unwrap();
    assert_eq!(
        world.get::<WorldSdfEcsSample>(recycled).unwrap().value,
        WorldSdfEcsSampleValue::Value(i16::MAX)
    );
    assert_ne!(
        old, recycled,
        "entity generations must prevent stale identity reuse"
    );
}

#[test]
fn empty_batches_and_invalid_mips_fail_without_gpu_work() {
    let Some(context) = context() else { return };
    let mut bridge = WorldSdfEcsGpuBridge::new(context, config()).unwrap();
    let mut empty = World::new();
    assert_eq!(bridge.sample_world(&mut empty).unwrap().sampled_entities, 0);

    let mut invalid = World::new();
    invalid.spawn(request([0, 0, 0], 2, 7));
    assert!(matches!(
        bridge.sample_world(&mut invalid),
        Err(WorldSdfEcsGpuError::InvalidRequestMip { index: 0, mip: 2 })
    ));
}
