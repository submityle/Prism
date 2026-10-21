//! Strict hardware evidence for object edits, local rebakes and GPU publication.
#![cfg(feature = "gpu_proxy")]

use prism_world_gpu::{GpuContext, GpuSdfError, GpuSdfQuery, GpuSdfQueryResult};
use prism_world_residency::{BrickCoord, SdfClipmapConfig};
use prism_world_runtime::{
    WorldSdfClipmapConfig, WorldSdfClipmapError, WorldSdfFocusConfig, WorldSdfFocusUpdateError,
    WorldSdfGpuClipmapError, WorldSdfGpuObjects, WorldSdfGpuObjectsConfig, WorldSdfGpuObjectsError,
    WorldSdfLocalVolume, WorldSdfObject, WorldSdfObjectBakeBudget, WorldSdfObjectConfig,
    WorldSdfObjectError, WorldSdfObjectLimits, WorldSdfObjectResource,
};
use std::{
    cell::Cell,
    mem::size_of,
    sync::{mpsc, Arc},
    time::Duration,
};
use GpuSdfQueryResult::{Unknown, Value};

const ORIGIN: [i64; 3] = [100, -200, 300];

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("SDF object GPU acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(
        context.evidence().is_hardware(),
        "software adapter is not coverage"
    );
    eprintln!(
        "[prism_world_runtime::sdf_objects] {:?}",
        context.evidence()
    );
    context
}

fn config() -> WorldSdfGpuObjectsConfig {
    WorldSdfGpuObjectsConfig {
        objects: WorldSdfObjectConfig {
            origin_mm: ORIGIN,
            voxel_size_mm: 10,
            distance_scale_mm: 2,
            brick_size: 1,
            mip_levels: 2,
        },
        limits: WorldSdfObjectLimits {
            max_objects: 32,
            max_bricks: 256,
            max_links: 512,
            max_dirty_bricks: 256,
            max_edit_bricks: 512,
            max_edit_scratch_bytes: 1 << 20,
        },
        focus: WorldSdfFocusConfig::new(ORIGIN, 10, 1, [0; 3]).unwrap(),
        clipmap: WorldSdfClipmapConfig::new(
            SdfClipmapConfig::new(2, 2, 1 << 20).unwrap(),
            50,
            1 << 20,
        )
        .unwrap(),
        max_queries: 32,
        max_upload_bytes: 1 << 20,
        max_commit_scratch_bytes: 4096,
    }
}

fn budget(count: usize) -> WorldSdfObjectBakeBudget {
    WorldSdfObjectBakeBudget {
        max_bricks: count,
        max_evaluations: 1 << 20,
        max_scratch_bytes: 1 << 20,
    }
}

fn object(value: i16, min: i64, max: i64) -> WorldSdfObject<i16> {
    WorldSdfObject {
        translation_mm: ORIGIN,
        local_min_mm: [min, 0, 0],
        local_max_mm: [max, 0, 0],
        source: Arc::new(value),
    }
}

fn bake(owner: &mut WorldSdfGpuObjects<i16>, count: usize) {
    let update = owner
        .prepare(budget(count))
        .unwrap()
        .evaluate(|value, _| Ok::<_, ()>(*value))
        .unwrap();
    owner.commit(update).unwrap();
}

fn q(x: i32, mip: u8) -> GpuSdfQuery {
    GpuSdfQuery {
        voxel: [x, 0, 0],
        mip,
    }
}

fn encoder(context: &GpuContext) -> wgpu::CommandEncoder {
    context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SDF object consumer"),
        })
}

struct Readback {
    buffer: wgpu::Buffer,
    count: usize,
}

// Each submission owns its witness. Later uploads are queued before any map/wait.
fn submit<T>(
    context: &GpuContext,
    owner: &mut WorldSdfGpuObjects<T>,
    queries: &[GpuSdfQuery],
) -> Readback {
    let bytes = (queries.len() * 16) as u64;
    let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF objects independent witness"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let called = Cell::new(false);
    owner
        .submit_queries(queries, encoder(context), |results, encoder| {
            called.set(true);
            encoder.copy_buffer_to_buffer(results, 0, &buffer, 0, bytes);
        })
        .unwrap();
    assert!(called.get());
    Readback {
        buffer,
        count: queries.len(),
    }
}

impl Readback {
    fn read(self, context: &GpuContext) -> Vec<GpuSdfQueryResult> {
        let (sender, receiver) = mpsc::channel();
        self.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                sender.send(result).unwrap();
            });
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        receiver
            .recv_timeout(Duration::from_secs(30))
            .unwrap()
            .unwrap();
        let mapped = self.buffer.get_mapped_range(..).unwrap();
        assert_eq!(mapped.len(), self.count * 16);
        let result = mapped
            .chunks_exact(16)
            .map(|bytes| {
                assert_eq!(&bytes[8..16], &[0; 8]);
                let status = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let value = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
                match status {
                    0 => {
                        assert_eq!(value, 0);
                        Unknown
                    }
                    1 => Value(i16::try_from(value).unwrap()),
                    _ => panic!("invalid query status {status}"),
                }
            })
            .collect();
        drop(mapped);
        self.buffer.unmap();
        result
    }
}

fn assert_cpu_closed<T>(context: &GpuContext, owner: &mut WorldSdfGpuObjects<T>) {
    let called = Cell::new(false);
    assert!(matches!(
        owner.submit_queries(&[q(0, 0)], encoder(context), |_, _| called.set(true)),
        Err(WorldSdfGpuObjectsError::Objects(
            WorldSdfObjectError::PendingBake { .. }
        ))
    ));
    assert!(!called.get());
    assert!(matches!(
        owner.update_focus(ORIGIN),
        Err(WorldSdfGpuObjectsError::Objects(
            WorldSdfObjectError::PendingBake { .. }
        ))
    ));
    assert!(matches!(
        owner.flush_dirty(100, 1 << 20),
        Err(WorldSdfGpuObjectsError::Objects(
            WorldSdfObjectError::PendingBake { .. }
        ))
    ));
}

fn assert_gpu_closed<T>(context: &GpuContext, owner: &mut WorldSdfGpuObjects<T>) {
    let called = Cell::new(false);
    assert!(matches!(
        owner.submit_queries(&[q(0, 0)], encoder(context), |_, _| called.set(true)),
        Err(WorldSdfGpuObjectsError::Upload(
            WorldSdfGpuClipmapError::PendingDirty { .. }
        ))
    ));
    assert!(!called.get());
}

#[test]
fn overlap_move_delete_and_unknown_are_ordered_without_host_waits() {
    let context = context();
    let mut owner = WorldSdfGpuObjects::new(&context, config()).unwrap();
    owner.upsert(8, object(40, -10, 20)).unwrap();
    owner.upsert(2, object(-30, 0, 10)).unwrap();
    assert_cpu_closed(&context, &mut owner);
    bake(&mut owner, 100);
    owner.update_focus(ORIGIN).unwrap();
    let bytes = owner.focus().ring().allocated_buffer_bytes();
    let queries = [
        q(-1, 0),
        q(0, 0),
        q(1, 0),
        q(2, 0),
        q(-1, 1),
        q(0, 1),
        q(1, 1),
    ];
    let first = submit(&context, &mut owner, &queries);

    owner.remove(2).unwrap();
    assert_cpu_closed(&context, &mut owner);
    bake(&mut owner, 100);
    assert_gpu_closed(&context, &mut owner);
    owner.flush_dirty(100, 1 << 20).unwrap();
    let restored = submit(&context, &mut owner, &queries);

    let mut moved = object(40, -10, 20);
    moved.translation_mm[0] += 20;
    owner.upsert(8, moved).unwrap();
    bake(&mut owner, 100);
    owner.flush_dirty(100, 1 << 20).unwrap();
    let moved = submit(&context, &mut owner, &queries);

    owner.remove(8).unwrap();
    bake(&mut owner, 100);
    owner.flush_dirty(100, 1 << 20).unwrap();
    let deleted = submit(&context, &mut owner, &queries);
    assert_eq!(owner.store().reserved_bricks(), 0);
    assert_eq!(owner.store().grid().brick_count(), 0);
    assert_eq!(owner.focus().ring().allocated_buffer_bytes(), bytes);

    // Read all witnesses only after the last edit/upload/query submission.
    assert_eq!(
        first.read(&context),
        [
            Value(40),
            Value(-30),
            Value(-30),
            Value(40),
            Value(i16::MAX),
            Value(-30),
            Value(40)
        ]
    );
    assert_eq!(
        restored.read(&context),
        [
            Value(40),
            Value(40),
            Value(40),
            Value(40),
            Value(i16::MAX),
            Value(40),
            Value(40)
        ]
    );
    assert_eq!(
        moved.read(&context),
        [
            Unknown,
            Unknown,
            Value(40),
            Value(40),
            Unknown,
            Value(i16::MAX),
            Value(40)
        ]
    );
    assert_eq!(deleted.read(&context), [Unknown; 7]);
}

#[test]
fn partial_composition_then_partial_upload_keeps_both_fences_closed() {
    let context = context();
    let mut cfg = config();
    cfg.objects.mip_levels = 1;
    cfg.clipmap =
        WorldSdfClipmapConfig::new(SdfClipmapConfig::new(1, 2, 4096).unwrap(), 25, 4096).unwrap();
    cfg.max_commit_scratch_bytes = size_of::<BrickCoord>();
    let mut owner = WorldSdfGpuObjects::new(&context, cfg).unwrap();
    owner.update_focus(ORIGIN).unwrap();
    owner.upsert(1, object(-9, 0, 10)).unwrap();
    let revision = owner.store().revision();
    let gpu_revision = owner.focus().ring().scheduler().revision();
    let update = owner
        .prepare(budget(2))
        .unwrap()
        .evaluate(|v, _| Ok::<_, ()>(*v))
        .unwrap();
    assert!(matches!(
        owner.commit(update),
        Err(WorldSdfGpuObjectsError::Objects(
            WorldSdfObjectError::BudgetExceeded {
                resource: WorldSdfObjectResource::ScratchBytes,
                ..
            }
        ))
    ));
    assert_eq!(owner.store().revision(), revision);
    assert_eq!(owner.focus().ring().scheduler().revision(), gpu_revision);
    assert_eq!(owner.store().grid().brick_count(), 0);
    bake(&mut owner, 1);
    assert_cpu_closed(&context, &mut owner);
    bake(&mut owner, 1);
    assert_gpu_closed(&context, &mut owner);
    assert_eq!(owner.flush_dirty(1, 36).unwrap().pending_bricks, 1);
    assert_gpu_closed(&context, &mut owner);
    assert_eq!(owner.flush_dirty(1, 36).unwrap().pending_bricks, 0);
    let readback = submit(&context, &mut owner, &[q(0, 0), q(1, 0)]);
    assert_eq!(readback.read(&context), [Value(-9), Value(-9)]);
}

#[test]
fn mark_request_failure_and_stale_work_preserve_both_owners() {
    let context = context();
    let mut cfg = config();
    cfg.clipmap = cfg.clipmap.with_mark_request_budget(1, 4096).unwrap();
    let mut owner = WorldSdfGpuObjects::new(&context, cfg).unwrap();
    owner.upsert(1, object(5, 0, 10)).unwrap();
    bake(&mut owner, 100);
    owner.update_focus(ORIGIN).unwrap();
    let old = submit(&context, &mut owner, &[q(0, 0), q(1, 0), q(0, 1)]);
    owner.upsert(1, object(17, 0, 10)).unwrap();
    bake(&mut owner, 1);
    let revision = owner.store().revision();
    let gpu_revision = owner.focus().ring().scheduler().revision();
    let dirty = owner.store().dirty_bricks().to_vec();
    let gpu_dirty = owner.focus().ring().scheduler().dirty_bricks().to_vec();
    assert_eq!(dirty.len(), 2);
    assert_eq!(gpu_dirty.len(), 1);
    let coords = [q(0, 0), q(1, 0), q(0, 1)];
    let values = coords.map(|q| {
        owner
            .store()
            .grid()
            .brick_samples(BrickCoord {
                x: q.voxel[0],
                y: 0,
                z: 0,
                mip: q.mip,
            })
            .unwrap()
            .unwrap()
            .to_vec()
    });
    let update = owner
        .prepare(budget(100))
        .unwrap()
        .evaluate(|v, _| Ok::<_, ()>(*v))
        .unwrap();
    assert!(matches!(
        owner.commit(update),
        Err(WorldSdfGpuObjectsError::Upload(
            WorldSdfGpuClipmapError::Scheduler(WorldSdfClipmapError::MarkRequestCapacityExceeded {
                required: 2,
                capacity: 1
            })
        ))
    ));
    assert_eq!(owner.store().revision(), revision);
    assert_eq!(owner.focus().ring().scheduler().revision(), gpu_revision);
    assert_eq!(owner.store().dirty_bricks(), dirty);
    assert_eq!(owner.focus().ring().scheduler().dirty_bricks(), gpu_dirty);
    assert_eq!(owner.store().grid().brick_count(), 3);
    for (q, values) in coords.into_iter().zip(values) {
        assert_eq!(
            owner
                .store()
                .grid()
                .brick_samples(BrickCoord {
                    x: q.voxel[0],
                    y: 0,
                    z: 0,
                    mip: q.mip,
                })
                .unwrap()
                .unwrap(),
            values
        );
    }

    let stale = owner
        .prepare(budget(1))
        .unwrap()
        .evaluate(|v, _| Ok::<_, ()>(*v))
        .unwrap();
    owner.upsert(1, object(-17, 0, 10)).unwrap();
    let revision = owner.store().revision();
    assert!(matches!(
        owner.commit(stale),
        Err(WorldSdfGpuObjectsError::Objects(
            WorldSdfObjectError::StaleWork
        ))
    ));
    assert_eq!(owner.store().revision(), revision);
    assert_eq!(owner.focus().ring().scheduler().revision(), gpu_revision);
    while !owner.store().dirty_bricks().is_empty() {
        bake(&mut owner, 1);
    }
    owner.flush_dirty(100, 4096).unwrap();
    let readback = submit(&context, &mut owner, &[q(0, 0), q(1, 0), q(0, 1)]);
    assert_eq!(old.read(&context), [Value(5); 3]);
    assert_eq!(readback.read(&context), [Value(-17); 3]);
}

#[test]
fn failed_upload_roll_and_invalid_queries_preserve_previous_generation() {
    let context = context();
    let mut cfg = config();
    cfg.objects.mip_levels = 1;
    cfg.clipmap =
        WorldSdfClipmapConfig::new(SdfClipmapConfig::new(1, 1, 4096).unwrap(), 9, 4096).unwrap();
    // Initial window: nine keys plus four present samples. A disjoint, fully
    // occupied window needs 324 bytes and must fail before moving either center.
    cfg.max_upload_bytes = 304;
    cfg.max_queries = 2;
    let mut owner = WorldSdfGpuObjects::new(&context, cfg).unwrap();
    let field = |value| WorldSdfObject {
        translation_mm: ORIGIN,
        local_min_mm: [0; 3],
        local_max_mm: [100, 0, 100],
        source: Arc::new(value),
    };
    owner.upsert(1, field(3)).unwrap();
    bake(&mut owner, 200);
    owner.update_focus(ORIGIN).unwrap();
    let old = submit(&context, &mut owner, &[q(0, 0)]);
    let gpu_bricks = owner.focus().ring().brick_count();
    let gpu_bytes = owner.focus().ring().allocated_buffer_bytes();
    owner.upsert(1, field(-7)).unwrap();
    bake(&mut owner, 200);
    let revision = owner.focus().ring().scheduler().revision();
    let dirty = owner.focus().ring().scheduler().dirty_bricks().to_vec();
    assert_eq!(dirty.len(), 4);
    assert_eq!(
        owner.flush_dirty(1, 35),
        Err(WorldSdfGpuObjectsError::Upload(
            WorldSdfGpuClipmapError::UploadBudgetTooSmall {
                required: 36,
                budget: 35
            }
        ))
    );
    assert!(matches!(
        owner.update_focus([ORIGIN[0] + 30, ORIGIN[1], ORIGIN[2] + 30]),
        Err(WorldSdfGpuObjectsError::Focus(
            WorldSdfFocusUpdateError::Apply(WorldSdfGpuClipmapError::Backend(
                GpuSdfError::UploadBudgetExceeded {
                    required: 324,
                    budget: 304,
                }
            ))
        ))
    ));
    assert_eq!(owner.focus().ring().scheduler().revision(), revision);
    assert_eq!(owner.focus().ring().scheduler().dirty_bricks(), dirty);
    assert_eq!(owner.focus().ring().brick_count(), gpu_bricks);
    assert_eq!(owner.focus().ring().allocated_buffer_bytes(), gpu_bytes);
    assert_eq!(
        owner.focus().focus().center(),
        owner.focus().ring().scheduler().center()
    );
    assert_eq!(
        owner.focus().ring().scheduler().center(),
        Some(BrickCoord {
            x: 0,
            y: 0,
            z: 0,
            mip: 0,
        })
    );
    assert_gpu_closed(&context, &mut owner);
    // A smaller roll uploads entering keys, retaining all four edited keys.
    owner
        .update_focus([ORIGIN[0] + 10, ORIGIN[1], ORIGIN[2]])
        .unwrap();
    assert_eq!(owner.focus().ring().scheduler().dirty_bricks(), dirty);
    assert_gpu_closed(&context, &mut owner);
    assert_eq!(owner.flush_dirty(2, 72).unwrap().pending_bricks, 2);
    assert_eq!(owner.flush_dirty(2, 72).unwrap().pending_bricks, 0);
    let current = submit(&context, &mut owner, &[q(0, 0), q(2, 0)]);
    for (queries, expected) in [
        (
            vec![q(0, 0); 3],
            GpuSdfError::QueryCapacityExceeded {
                required: 3,
                capacity: 2,
            },
        ),
        (
            vec![q(0, 0), q(0, 1)],
            GpuSdfError::InvalidQueryMip { index: 1, mip: 1 },
        ),
    ] {
        let called = Cell::new(false);
        assert_eq!(
            owner
                .submit_queries(&queries, encoder(&context), |_, _| called.set(true))
                .unwrap_err(),
            WorldSdfGpuObjectsError::Upload(WorldSdfGpuClipmapError::Backend(expected))
        );
        assert!(!called.get());
    }
    let recovered = submit(&context, &mut owner, &[q(1, 0)]);
    assert_eq!(old.read(&context), [Value(3)]);
    assert_eq!(current.read(&context), [Value(-7); 2]);
    assert_eq!(recovered.read(&context), [Value(-7)]);
}

#[test]
fn off_window_changes_are_loaded_on_entry_including_mixed_batches() {
    let context = context();
    let mut owner = WorldSdfGpuObjects::new(&context, config()).unwrap();
    owner.update_focus(ORIGIN).unwrap();
    // One object spans resident and nonresident coordinates at both mips.
    owner.upsert(1, object(31, 0, 100)).unwrap();
    bake(&mut owner, 100);
    owner.flush_dirty(100, 4096).unwrap();
    let inside = submit(&context, &mut owner, &[q(0, 0), q(2, 0), q(10, 0)]);
    owner
        .update_focus([ORIGIN[0] + 100, ORIGIN[1], ORIGIN[2]])
        .unwrap();
    let entered = submit(&context, &mut owner, &[q(10, 0), q(5, 1), q(0, 0)]);
    owner.remove(1).unwrap();
    bake(&mut owner, 100);
    owner.flush_dirty(100, 4096).unwrap();
    owner.update_focus(ORIGIN).unwrap();
    let removed = submit(&context, &mut owner, &[q(0, 0), q(0, 1)]);
    assert_eq!(inside.read(&context), [Value(31), Value(31), Unknown]);
    assert_eq!(entered.read(&context), [Value(31), Value(31), Unknown]);
    assert_eq!(removed.read(&context), [Unknown; 2]);
}

#[test]
fn local_volume_sampling_reaches_gpu_and_geometry_mismatch_is_rejected() {
    let context = context();
    for mismatch in 0..4 {
        let mut cfg = config();
        match mismatch {
            0 => cfg.objects.origin_mm[0] += 1,
            1 => cfg.objects.voxel_size_mm += 1,
            2 => cfg.objects.brick_size += 1,
            _ => cfg.objects.mip_levels += 1,
        }
        assert!(matches!(
            WorldSdfGpuObjects::<i16>::new(&context, cfg),
            Err(WorldSdfGpuObjectsError::GeometryMismatch)
        ));
    }
    let mut cfg = config();
    cfg.objects.brick_size = 2;
    cfg.focus = WorldSdfFocusConfig::new(ORIGIN, 10, 2, [0; 3]).unwrap();
    let mut owner = WorldSdfGpuObjects::new(&context, cfg).unwrap();
    let mut samples = vec![i16::MAX; 27];
    samples[13] = -123;
    let volume = WorldSdfLocalVolume::new([3; 3], 10, samples, 27).unwrap();
    owner
        .upsert(
            1,
            WorldSdfObject {
                translation_mm: ORIGIN,
                local_min_mm: [0; 3],
                local_max_mm: volume.max_local_mm(),
                source: Arc::new(volume),
            },
        )
        .unwrap();
    let update = owner
        .prepare(budget(100))
        .unwrap()
        .evaluate(|volume, point| Ok::<_, ()>(volume.sample(point.local_position_mm)))
        .unwrap();
    owner.commit(update).unwrap();
    owner.update_focus(ORIGIN).unwrap();
    let queries = [
        GpuSdfQuery {
            voxel: [1; 3],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [0; 3],
            mip: 0,
        },
        GpuSdfQuery {
            voxel: [0; 3],
            mip: 1,
        },
        GpuSdfQuery {
            voxel: [-1; 3],
            mip: 0,
        },
    ];
    let readback = submit(&context, &mut owner, &queries);
    assert_eq!(
        readback.read(&context),
        [Value(-123), Value(i16::MAX), Value(i16::MAX), Unknown]
    );
}
