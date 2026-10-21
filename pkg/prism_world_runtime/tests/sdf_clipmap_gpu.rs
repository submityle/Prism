//! Strict physical-GPU coverage of the runtime-owned SDF upload/consumer boundary.
#![cfg(feature = "gpu_proxy")]

use std::{cell::Cell, sync::mpsc, time::Duration};

use prism_world_gpu::{GpuContext, GpuSdfError, GpuSdfQuery, GpuSdfQueryResult};
use prism_world_residency::{BrickCoord, SdfClipmapConfig, SparseSdfBrickGrid};
use prism_world_runtime::{
    WorldSdfClipmapConfig, WorldSdfClipmapUploadError, WorldSdfGpuClipmap, WorldSdfGpuClipmapError,
};

use GpuSdfQueryResult::{Unknown, Value};

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical SDF GPU acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(context.evidence().is_hardware());
    eprintln!(
        "[prism_world_runtime::sdf_clipmap] {:?}",
        context.evidence()
    );
    context
}

fn config(mips: u8) -> WorldSdfClipmapConfig {
    // Radius one: nine XZ slots per mip. Keep planner/dirty budgets ample so
    // upload-budget failures below come from the physical backend, not planning.
    WorldSdfClipmapConfig::new(
        SdfClipmapConfig::new(mips, 1, 4096).unwrap(),
        9 * usize::from(mips),
        4096,
    )
    .unwrap()
}

fn coord(x: i32) -> BrickCoord {
    BrickCoord::new(x, 0, 0, 0)
}

fn query(voxel: [i32; 3], mip: u8) -> GpuSdfQuery {
    GpuSdfQuery { voxel, mip }
}

fn encoder(context: &GpuContext) -> wgpu::CommandEncoder {
    context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("runtime SDF integration consumer"),
        })
}

struct Readback {
    buffer: wgpu::Buffer,
    count: usize,
}

// This helper must neither poll nor map: callers can submit more work and edit
// the same ring before observing any of the previous submissions.
fn submit_copy(
    context: &GpuContext,
    owner: &mut WorldSdfGpuClipmap,
    queries: &[GpuSdfQuery],
) -> Readback {
    assert!(!queries.is_empty());
    let bytes = queries.len() as u64 * 16;
    let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("runtime SDF independent consumer readback"),
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
    assert!(
        called.get(),
        "successful submission must encode its consumer"
    );
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
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("physical SDF GPU readback poll failed");
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("physical SDF GPU map callback missing")
            .expect("physical SDF GPU map failed");
        let mapped = self.buffer.get_mapped_range(..).unwrap();
        assert_eq!(mapped.len(), self.count * 16);
        let results = mapped
            .chunks_exact(16)
            .enumerate()
            .map(|(index, record)| {
                let status = u32::from_le_bytes(record[0..4].try_into().unwrap());
                let value = i32::from_le_bytes(record[4..8].try_into().unwrap());
                assert_eq!(&record[8..16], &[0; 8], "reserved ABI words at {index}");
                match status {
                    0 if value == 0 => Unknown,
                    1 => Value(i16::try_from(value).expect("GPU sample exceeds signed i16")),
                    _ => panic!("invalid runtime SDF GPU record {index}: {status}/{value}"),
                }
            })
            .collect();
        drop(mapped);
        self.buffer.unmap();
        results
    }
}

fn assert_rejected(
    context: &GpuContext,
    owner: &mut WorldSdfGpuClipmap,
    queries: &[GpuSdfQuery],
    expected: WorldSdfGpuClipmapError,
) {
    let called = Cell::new(false);
    let result = owner.submit_queries(queries, encoder(context), |_, _| called.set(true));
    assert!(
        !called.get(),
        "rejected submission must not call its consumer"
    );
    assert_eq!(result.expect_err("submission must fail closed"), expected);
}

#[test]
fn first_window_negative_samples_mips_and_rolling_slot_aliases() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
    let negative = BrickCoord::new(-1, 0, -1, 0);
    let coarse = BrickCoord::new(-1, 0, -1, 1);
    let entering = BrickCoord::new(2, 0, -1, 0);
    grid.insert_brick(negative, vec![i16::MIN, i16::MAX, -3, 4, -5, 6, -7, 8])
        .unwrap();
    grid.insert_brick(coord(0), vec![19; 8]).unwrap();
    grid.insert_brick(coarse, vec![-71; 8]).unwrap();
    // Off-window source data must not be uploaded before its key enters.
    grid.insert_brick(entering, vec![31; 8]).unwrap();
    let mut owner = WorldSdfGpuClipmap::new(&context, config(2), 2, 8, 4096).unwrap();
    let first = owner.update_center(&grid, coord(0)).unwrap();
    assert_eq!(first.uploaded_bytes, 18 * 32 + 3 * 16);
    assert_eq!(first.pending_bricks, 0);
    assert_eq!(owner.brick_count(), 3);
    let aliased_slot = owner.scheduler().slot_index(negative).unwrap();
    let queries = [
        query([-2, 0, -2], 0),
        query([-1, 0, -2], 0),
        query([-2, 1, -1], 0),
        query([-1, 1, -1], 0),
        query([-1, 1, -1], 1),
        query([0, 0, 0], 0),
        query([4, 0, -2], 0),
        query([-1, -1, -1], 0),
    ];
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![
            Value(i16::MIN),
            Value(i16::MAX),
            Value(-7),
            Value(8),
            Value(-71),
            Value(19),
            Unknown,
            Unknown
        ]
    );
    let same = owner.update_center(&grid, coord(0)).unwrap();
    assert_eq!(same.uploaded_bytes, 0);
    assert_eq!(same.revision, first.revision);

    // -1 and 2 alias the same modulo-three X slot; their full keys must not.
    let rolled = owner.update_center(&grid, coord(1)).unwrap();
    assert_eq!(rolled.uploaded_bytes, 3 * 32 + 16);
    assert_eq!(rolled.pending_bricks, 0);
    assert_eq!(owner.scheduler().center(), Some(coord(1)));
    assert_eq!(owner.scheduler().slot_index(negative), None);
    assert_eq!(owner.scheduler().slot_index(entering), Some(aliased_slot));
    assert_eq!(owner.brick_count(), 3);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![
            Unknown,
            Unknown,
            Unknown,
            Unknown,
            Value(-71),
            Value(19),
            Value(31),
            Unknown
        ]
    );

    // Roll that slot again, this time to absent x=5. Stale samples must not
    // resurrect either old key, and the coarse mip now leaves its window too.
    owner.update_center(&grid, coord(4)).unwrap();
    assert_eq!(owner.brick_count(), 0);
    assert_eq!(
        submit_copy(
            &context,
            &mut owner,
            &[
                query([4, 0, -2], 0),
                query([10, 0, -2], 0),
                query([-1, 1, -1], 1)
            ],
        )
        .read(&context),
        vec![Unknown; 3]
    );
    owner.update_center(&grid, coord(0)).unwrap();
    assert_eq!(
        submit_copy(&context, &mut owner, &[queries[0], queries[4], queries[5]]).read(&context),
        vec![Value(i16::MIN), Value(-71), Value(19)]
    );
}

#[test]
fn pending_edits_gate_consumers_and_flush_obeys_bytes_count_and_deletion() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    for x in -1..=1 {
        grid.insert_brick(coord(x), vec![1; 8]).unwrap();
    }
    let mut owner = WorldSdfGpuClipmap::new(&context, config(1), 2, 3, 4096).unwrap();
    owner.update_center(&grid, coord(0)).unwrap();
    for (x, value) in [(-1, 11), (0, 22), (1, 33)] {
        grid.insert_brick(coord(x), vec![value; 8]).unwrap();
    }
    owner.mark_dirty(&[coord(1), coord(-1), coord(0)]).unwrap();
    let queries = [
        query([-2, 0, 0], 0),
        query([0, 0, 0], 0),
        query([2, 0, 0], 0),
    ];
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 3 },
    );

    // Byte budget binds first (two present bricks cost 96), count binds next.
    let bytes_limited = owner.flush_dirty(&grid, usize::MAX, 95).unwrap();
    assert_eq!(
        (bytes_limited.uploaded_bytes, bytes_limited.pending_bricks),
        (48, 2)
    );
    let count_limited = owner.flush_dirty(&grid, 1, 4096).unwrap();
    assert_eq!(
        (count_limited.uploaded_bytes, count_limited.pending_bricks),
        (48, 1)
    );
    assert_eq!(owner.scheduler().dirty_bricks(), &[coord(-1)]);
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );

    // Re-edit an already-pending brick: upload the latest source, not a cached
    // copy captured when it was marked, and fence the revision again.
    grid.insert_brick(coord(-1), vec![111; 8]).unwrap();
    let before_mark = owner.scheduler().revision();
    assert_eq!(owner.mark_dirty(&[coord(-1)]).unwrap(), 0);
    assert_eq!(owner.scheduler().revision(), before_mark + 1);
    let revision = owner.scheduler().revision();
    assert_eq!(
        owner.flush_dirty(&grid, 1, 47),
        Err(WorldSdfClipmapUploadError::UploadBudgetTooSmall {
            required: 48,
            budget: 47
        })
    );
    assert_eq!(owner.scheduler().revision(), revision);
    assert_eq!(owner.scheduler().dirty_bricks(), &[coord(-1)]);
    let last = owner.flush_dirty(&grid, 1, 48).unwrap();
    assert_eq!((last.uploaded_bytes, last.pending_bricks), (48, 0));
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(111), Value(22), Value(33)]
    );

    assert!(grid.remove_brick(coord(0)).unwrap());
    owner.mark_dirty(&[coord(0)]).unwrap();
    let removed = owner.flush_dirty(&grid, 1, 32).unwrap();
    assert_eq!((removed.uploaded_bytes, removed.pending_bricks), (32, 0));
    assert_eq!(owner.brick_count(), 2);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(111), Unknown, Value(33)]
    );
}

#[test]
fn upload_budget_failure_preserves_initial_and_committed_center_dirty_and_gpu_data() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(coord(0), vec![7; 8]).unwrap();
    let mut too_small = WorldSdfGpuClipmap::new(&context, config(1), 2, 1, 303).unwrap();
    assert_eq!(
        too_small.update_center(&grid, coord(0)),
        Err(WorldSdfClipmapUploadError::Backend(
            GpuSdfError::UploadBudgetExceeded {
                required: 304,
                budget: 303
            }
        ))
    );
    assert_eq!(too_small.scheduler().center(), None);
    assert_eq!(too_small.scheduler().revision(), 0);
    assert!(too_small.scheduler().dirty_bricks().is_empty());
    assert_eq!(too_small.brick_count(), 0);
    assert_rejected(
        &context,
        &mut too_small,
        &[query([0, 0, 0], 0)],
        WorldSdfClipmapUploadError::Uninitialized,
    );

    let mut owner = WorldSdfGpuClipmap::new(&context, config(1), 2, 3, 304).unwrap();
    owner.update_center(&grid, coord(0)).unwrap();
    let allocated = owner.allocated_buffer_bytes();
    // Make a disjoint roll more expensive than initialization, with a pending
    // retained edit that a failed transaction must not discard.
    for x in 9..=11 {
        grid.insert_brick(coord(x), vec![22; 8]).unwrap();
    }
    // Keep x=0 clean as a GPU witness: flushing x=1 after failure must not
    // accidentally repair a partially overwritten x=0 slot and mask the bug.
    grid.insert_brick(coord(1), vec![13; 8]).unwrap();
    owner.mark_dirty(&[coord(1)]).unwrap();
    let revision = owner.scheduler().revision();
    assert_eq!(
        owner.update_center(&grid, coord(10)),
        Err(WorldSdfClipmapUploadError::Backend(
            GpuSdfError::UploadBudgetExceeded {
                required: 336,
                budget: 304
            }
        ))
    );
    assert_eq!(owner.scheduler().center(), Some(coord(0)));
    assert_eq!(owner.scheduler().revision(), revision);
    assert_eq!(owner.scheduler().dirty_bricks(), &[coord(1)]);
    assert_eq!(owner.brick_count(), 1);
    assert_eq!(owner.allocated_buffer_bytes(), allocated);
    let queries = [
        query([0, 0, 0], 0),
        query([2, 0, 0], 0),
        query([20, 0, 0], 0),
    ];
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );
    owner.flush_dirty(&grid, 1, 48).unwrap();
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(7), Value(13), Unknown]
    );
    // A small roll proves the backend planner also retained the old center:
    // it must keep x=0, not treat it as an already-uploaded entering key.
    let small_roll = owner.update_center(&grid, coord(1)).unwrap();
    assert_eq!(small_roll.uploaded_bytes, 96);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(7), Value(13), Unknown]
    );

    assert!(grid.remove_brick(coord(9)).unwrap());
    assert!(grid.remove_brick(coord(11)).unwrap());
    let retry = owner.update_center(&grid, coord(10)).unwrap();
    assert_eq!((retry.uploaded_bytes, retry.pending_bricks), (304, 0));
    assert_eq!(owner.scheduler().center(), Some(coord(10)));
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Unknown, Unknown, Value(22)]
    );
}

#[test]
fn back_to_back_submissions_survive_later_writes_and_invalid_queries_recover() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(coord(0), vec![-13, 17, 1, 2, 3, 4, 5, 6])
        .unwrap();
    grid.insert_brick(coord(1), vec![29; 8]).unwrap();
    let mut owner = WorldSdfGpuClipmap::new(&context, config(1), 2, 4, 4096).unwrap();
    owner.update_center(&grid, coord(0)).unwrap();

    // Deliberately no read/map/poll between these submissions and all later
    // edits. Vary both query order and count to catch overwritten uniforms too.
    let first = submit_copy(
        &context,
        &mut owner,
        &[
            query([0, 0, 0], 0),
            query([1, 0, 0], 0),
            query([2, 0, 0], 0),
        ],
    );
    let second = submit_copy(&context, &mut owner, &[query([2, 0, 0], 0)]);
    grid.insert_brick(coord(0), vec![101; 8]).unwrap();
    assert!(grid.remove_brick(coord(1)).unwrap());
    owner.mark_dirty(&[coord(0), coord(1)]).unwrap();
    let edited = owner.flush_dirty(&grid, 2, 80).unwrap();
    assert_eq!((edited.uploaded_bytes, edited.pending_bricks), (80, 0));
    let third = submit_copy(
        &context,
        &mut owner,
        &[
            query([2, 0, 0], 0),
            query([0, 0, 0], 0),
            query([1, 0, 0], 0),
            query([100, 0, 0], 0),
        ],
    );

    let revision = owner.scheduler().revision();
    // A valid prefix followed by a bad mip must not publish a partial query
    // batch or invoke the consumer. Capacity rejection must be recoverable too.
    assert_rejected(
        &context,
        &mut owner,
        &[query([2, 0, 0], 0), query([0, 0, 0], 1)],
        WorldSdfClipmapUploadError::Backend(GpuSdfError::InvalidQueryMip { index: 1, mip: 1 }),
    );
    assert_rejected(
        &context,
        &mut owner,
        &[query([2, 0, 0], 0); 5],
        WorldSdfClipmapUploadError::Backend(GpuSdfError::QueryCapacityExceeded {
            required: 5,
            capacity: 4,
        }),
    );
    assert_eq!(owner.scheduler().revision(), revision);
    assert_eq!(owner.scheduler().center(), Some(coord(0)));
    assert!(owner.scheduler().dirty_bricks().is_empty());
    let recovered = submit_copy(&context, &mut owner, &[query([0, 0, 0], 0)]);

    // The first device wait happens only now, after every later queue write.
    assert_eq!(first.read(&context), vec![Value(-13), Value(17), Value(29)]);
    assert_eq!(second.read(&context), vec![Value(29)]);
    assert_eq!(
        third.read(&context),
        vec![Unknown, Value(101), Value(101), Unknown]
    );
    assert_eq!(recovered.read(&context), vec![Value(101)]);
}
