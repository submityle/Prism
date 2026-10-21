//! Strict physical-GPU coverage of world-mm focus and paired publication.
#![cfg(feature = "gpu_proxy")]

use std::{cell::Cell, sync::mpsc, time::Duration};

use prism_world_gpu::{GpuContext, GpuSdfError, GpuSdfQuery, GpuSdfQueryResult};
use prism_world_residency::{BrickCoord, ClipmapAxis, SdfClipmapConfig, SparseSdfBrickGrid};
use prism_world_runtime::{
    WorldSdfClipmapConfig, WorldSdfClipmapUploadError, WorldSdfFocusConfig, WorldSdfFocusDriver,
    WorldSdfFocusError, WorldSdfFocusUpdateError, WorldSdfGpuClipmapError, WorldSdfGpuFocus,
};

use GpuSdfQueryResult::{Unknown, Value};

const ORIGIN: [i64; 3] = [1_000, -2_000, 3_000];

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical SDF focus GPU acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(
        context.evidence().is_hardware(),
        "software adapters are not coverage"
    );
    eprintln!("[prism_world_runtime::sdf_focus] {:?}", context.evidence());
    context
}

fn config(mips: u8, radius: u32) -> WorldSdfClipmapConfig {
    let side = usize::try_from(radius).unwrap() * 2 + 1;
    WorldSdfClipmapConfig::new(
        SdfClipmapConfig::new(mips, radius, 4096).unwrap(),
        side * side * usize::from(mips),
        4096,
    )
    .unwrap()
}

fn focus(voxel_mm: u32, brick_size: u8, margins: [u64; 3]) -> WorldSdfFocusConfig {
    WorldSdfFocusConfig::new(ORIGIN, voxel_mm, brick_size, margins).unwrap()
}

fn position(relative: [i64; 3]) -> [i64; 3] {
    std::array::from_fn(|i| ORIGIN[i].checked_add(relative[i]).unwrap())
}

fn brick(x: i32, y: i32, z: i32, mip: u8) -> BrickCoord {
    BrickCoord::new(x, y, z, mip)
}

fn x_brick(x: i32) -> BrickCoord {
    brick(x, 0, 0, 0)
}

fn query(voxel: [i32; 3], mip: u8) -> GpuSdfQuery {
    GpuSdfQuery { voxel, mip }
}

fn encoder(context: &GpuContext) -> wgpu::CommandEncoder {
    context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("world-mm SDF focus integration consumer"),
        })
}

#[derive(Debug, Eq, PartialEq)]
struct Snapshot {
    focus: Option<BrickCoord>,
    ring: Option<BrickCoord>,
    revision: u64,
    dirty: Vec<BrickCoord>,
    allocated_bytes: u64,
    present_bricks: usize,
    slots: Vec<Option<usize>>,
}

fn snapshot(owner: &WorldSdfGpuFocus, probes: &[BrickCoord]) -> Snapshot {
    let ring = owner.ring();
    Snapshot {
        focus: owner.focus().center(),
        ring: ring.scheduler().center(),
        revision: ring.scheduler().revision(),
        dirty: ring.scheduler().dirty_bricks().to_vec(),
        allocated_bytes: ring.allocated_buffer_bytes(),
        present_bricks: ring.brick_count(),
        slots: probes
            .iter()
            .map(|&coord| ring.scheduler().slot_index(coord))
            .collect(),
    }
}

fn assert_center(owner: &WorldSdfGpuFocus, expected: BrickCoord) {
    assert_eq!(owner.focus().center(), Some(expected));
    assert_eq!(owner.ring().scheduler().center(), Some(expected));
}

struct Readback {
    buffer: wgpu::Buffer,
    count: usize,
}

// No map or device wait here: each consumer has an independent GPU witness.
fn submit_copy(
    context: &GpuContext,
    owner: &mut WorldSdfGpuFocus,
    queries: &[GpuSdfQuery],
) -> Readback {
    assert!(!queries.is_empty());
    let bytes = u64::try_from(queries.len())
        .unwrap()
        .checked_mul(16)
        .unwrap();
    let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("world-mm SDF focus independent readback"),
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
    assert!(called.get(), "successful query must encode its consumer");
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
            .expect("physical SDF focus GPU readback poll failed");
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("physical SDF focus GPU map callback missing")
            .expect("physical SDF focus GPU map failed");
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
                    _ => panic!("invalid SDF focus GPU record {index}: {status}/{value}"),
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
    owner: &mut WorldSdfGpuFocus,
    queries: &[GpuSdfQuery],
    expected: WorldSdfGpuClipmapError,
) {
    let called = Cell::new(false);
    let result = owner.submit_queries(queries, encoder(context), |_, _| called.set(true));
    assert!(!called.get(), "rejected query must not encode its consumer");
    assert_eq!(result.expect_err("query must fail closed"), expected);
}

#[test]
fn negative_offset_multimip_independent_xz_deadbands_and_exact_y_plane() {
    let context = context();
    // B=20mm; distinct X/Z margins exercise independent half-open intervals.
    let mapping = focus(10, 2, [3, 0, 7]);
    let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
    for (coord, value) in [
        (brick(-1, -1, -1, 0), -17),
        (brick(-1, -1, -1, 1), -71),
        (brick(1, -1, -1, 0), 31),
        (brick(1, -1, 1, 0), 41),
        (brick(0, 0, 0, 0), 53),
        (brick(0, 0, 0, 1), 61),
    ] {
        grid.insert_brick(coord, vec![value; 8]).unwrap();
    }
    let mut owner = WorldSdfGpuFocus::new(&context, mapping, config(2, 1), 8, 4096).unwrap();
    let allocated = owner.ring().allocated_buffer_bytes();
    assert!(allocated > 0);
    let first = owner.update_focus(&grid, position([-1, -1, -1])).unwrap();
    assert_eq!(first.center, brick(-1, -1, -1, 0));
    let first_stats = first.applied.unwrap();
    assert_eq!(
        (first_stats.uploaded_bytes, first_stats.pending_bricks),
        (608, 0)
    );
    assert_eq!(first_stats.revision, owner.ring().scheduler().revision());
    assert_center(&owner, first.center);
    assert_eq!(owner.ring().brick_count(), 2);
    // Negative mip-zero -1 must floor to -1, not truncate to zero at mip one.
    assert!(owner
        .ring()
        .scheduler()
        .slot_index(brick(-1, -1, -1, 1))
        .is_some());
    assert_eq!(
        owner.ring().scheduler().slot_index(brick(-1, 0, -1, 1)),
        None
    );
    let queries = [
        query([-2, -2, -2], 0),
        query([-2, -2, -2], 1),
        query([2, -2, -2], 0),
        query([2, -2, 2], 0),
        query([0, 0, 0], 0),
        query([0, 0, 0], 1),
    ];
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(-17), Value(-71), Unknown, Unknown, Unknown, Unknown]
    );

    let probes = [brick(-1, -1, -1, 0), brick(-1, -1, -1, 1)];
    let held = snapshot(&owner, &probes);
    // Both lower endpoints are inclusive; both upper endpoints are exclusive.
    for relative in [[-23, -1, -27], [2, -1, 6], [-1, -1, -1]] {
        let update = owner.update_focus(&grid, position(relative)).unwrap();
        assert_eq!(update.center, first.center);
        assert_eq!(update.applied, None);
        assert_eq!(snapshot(&owner, &probes), held);
    }

    // X exits at +3 while Z's raw brick is zero but remains held at -1.
    let x_roll = owner.update_focus(&grid, position([3, -1, 6])).unwrap();
    assert_eq!(x_roll.center, brick(0, -1, -1, 0));
    assert_eq!(x_roll.applied.unwrap().uploaded_bytes, 6 * 32 + 16);
    assert_center(&owner, x_roll.center);
    assert_eq!(owner.ring().brick_count(), 3);
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Value(-17), Value(-71), Value(31), Unknown, Unknown, Unknown]
    );

    // Only Z now exits; both negative coarse coordinates cross to zero.
    let z_roll = owner.update_focus(&grid, position([3, -1, 7])).unwrap();
    assert_eq!(z_roll.center, brick(0, -1, 0, 0));
    assert_eq!(z_roll.applied.unwrap().uploaded_bytes, 6 * 32 + 16);
    assert_center(&owner, z_roll.center);
    assert_eq!(owner.ring().brick_count(), 4);
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![
            Value(-17),
            Value(-71),
            Value(31),
            Value(41),
            Unknown,
            Unknown
        ]
    );

    // hy=0 changes Y immediately at zero, replacing all nine slots at EACH mip.
    let y_roll = owner.update_focus(&grid, position([3, 0, 7])).unwrap();
    assert_eq!(y_roll.center, brick(0, 0, 0, 0));
    let stats = y_roll.applied.unwrap();
    assert_eq!(
        (stats.uploaded_bytes, stats.pending_bricks),
        (18 * 32 + 2 * 16, 0)
    );
    assert_center(&owner, y_roll.center);
    assert_eq!(owner.ring().brick_count(), 2);
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    assert_eq!(
        submit_copy(&context, &mut owner, &queries).read(&context),
        vec![Unknown, Unknown, Unknown, Unknown, Value(53), Value(61)]
    );
}

#[test]
fn initial_303_byte_failure_stays_uninitialized_and_same_focus_retries_after_removal() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(x_brick(0), vec![7; 8]).unwrap();
    let mut owner =
        WorldSdfGpuFocus::new(&context, focus(1, 2, [0; 3]), config(1, 1), 1, 303).unwrap();
    let probes = [x_brick(0), x_brick(1)];
    let initial = snapshot(&owner, &probes);
    assert_eq!(
        (initial.focus, initial.ring, initial.revision),
        (None, None, 0)
    );
    assert!(initial.dirty.is_empty());
    assert_eq!(initial.present_bricks, 0);
    assert!(initial.allocated_bytes > 0);
    assert_eq!(
        owner.update_focus(&grid, ORIGIN),
        Err(WorldSdfFocusUpdateError::Apply(
            WorldSdfClipmapUploadError::Backend(GpuSdfError::UploadBudgetExceeded {
                required: 304,
                budget: 303
            })
        ))
    );
    assert_eq!(snapshot(&owner, &probes), initial);
    assert_rejected(
        &context,
        &mut owner,
        &[query([0, 0, 0], 0)],
        WorldSdfClipmapUploadError::Uninitialized,
    );
    assert_eq!(snapshot(&owner, &probes), initial);

    assert!(grid.remove_brick(x_brick(0)).unwrap());
    let retry = owner.update_focus(&grid, ORIGIN).unwrap();
    assert_eq!(retry.center, x_brick(0));
    let stats = retry
        .applied
        .expect("failed focus was not committed; retry must apply");
    assert_eq!(
        (stats.uploaded_bytes, stats.pending_bricks, stats.revision),
        (288, 0, 1)
    );
    assert_center(&owner, x_brick(0));
    assert_eq!(owner.ring().brick_count(), 0);
    assert_eq!(
        owner.ring().allocated_buffer_bytes(),
        initial.allocated_bytes
    );
    assert_eq!(
        submit_copy(&context, &mut owner, &[query([0, 0, 0], 0)]).read(&context),
        vec![Unknown]
    );
}

#[test]
fn successful_roll_retains_in_window_edits_drops_exited_edits_and_uploads_entered_data() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    for (x, value) in [(-1, 11), (0, 21), (1, 31), (2, 41)] {
        grid.insert_brick(x_brick(x), vec![value; 8]).unwrap();
    }
    let mut owner =
        WorldSdfGpuFocus::new(&context, focus(1, 2, [0; 3]), config(1, 1), 4, 4096).unwrap();
    owner.update_focus(&grid, ORIGIN).unwrap();
    let allocated = owner.ring().allocated_buffer_bytes();
    let queries = [
        query([-2, 0, 0], 0),
        query([0, 0, 0], 0),
        query([2, 0, 0], 0),
        query([4, 0, 0], 0),
    ];
    let before_roll = submit_copy(&context, &mut owner, &queries);
    grid.remove_brick(x_brick(-1)).unwrap();
    grid.insert_brick(x_brick(0), vec![27; 8]).unwrap();
    // An off-window edit needs no mark: entering the window uploads it. A
    // mixed invalid mark request is rejected atomically, not silently filtered.
    let unmarked = snapshot(&owner, &[x_brick(-1), x_brick(0), x_brick(2)]);
    assert_eq!(
        owner.mark_dirty(&[x_brick(-1), x_brick(0), x_brick(2)]),
        Err(WorldSdfClipmapUploadError::Scheduler(
            prism_world_runtime::WorldSdfClipmapError::DirtyOutsideWindow { coord: x_brick(2) }
        ))
    );
    assert_eq!(
        snapshot(&owner, &[x_brick(-1), x_brick(0), x_brick(2)]),
        unmarked
    );
    assert_eq!(owner.mark_dirty(&[x_brick(-1), x_brick(0)]).unwrap(), 2);
    let revision = owner.ring().scheduler().revision();
    let rolled = owner.update_focus(&grid, position([2, 0, 0])).unwrap();
    assert_center(&owner, x_brick(1));
    let stats = rolled.applied.unwrap();
    assert_eq!(stats.uploaded_bytes, 3 * 32 + 16);
    assert_eq!(stats.pending_bricks, 1);
    assert_eq!(stats.revision, revision + 1);
    assert_eq!(owner.ring().scheduler().dirty_bricks(), &[x_brick(0)]);
    assert_eq!(owner.ring().scheduler().slot_index(x_brick(-1)), None);
    assert!(owner.ring().scheduler().slot_index(x_brick(0)).is_some());
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );
    let retained = snapshot(&owner, &[x_brick(-1), x_brick(0), x_brick(2)]);
    assert_eq!(
        owner
            .update_focus(&grid, position([2, 0, 0]))
            .unwrap()
            .applied,
        None
    );
    assert_eq!(
        snapshot(&owner, &[x_brick(-1), x_brick(0), x_brick(2)]),
        retained
    );
    let flushed = owner.flush_dirty(&grid, 1, 48).unwrap();
    assert_eq!((flushed.uploaded_bytes, flushed.pending_bricks), (48, 0));
    let after_roll = submit_copy(&context, &mut owner, &queries);
    // Later writes cannot alter an earlier consumer's independent GPU witness.
    assert_eq!(
        before_roll.read(&context),
        vec![Value(11), Value(21), Value(31), Unknown]
    );
    assert_eq!(
        after_roll.read(&context),
        vec![Unknown, Value(27), Value(31), Value(41)]
    );
}

#[test]
fn teleport_336_byte_failure_preserves_pair_dirty_and_old_gpu_witness_then_retries() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(x_brick(0), vec![7; 8]).unwrap();
    let mut owner =
        WorldSdfGpuFocus::new(&context, focus(1, 2, [0; 3]), config(1, 1), 3, 304).unwrap();
    assert_eq!(
        owner
            .update_focus(&grid, ORIGIN)
            .unwrap()
            .applied
            .unwrap()
            .uploaded_bytes,
        304
    );
    let allocated = owner.ring().allocated_buffer_bytes();
    let queries = [
        query([0, 0, 0], 0),
        query([2, 0, 0], 0),
        query([20, 0, 0], 0),
    ];
    let before_failure = submit_copy(&context, &mut owner, &queries);
    for x in 9..=11 {
        grid.insert_brick(x_brick(x), vec![22; 8]).unwrap();
    }
    // x=0 remains CLEAN: flushing x=1 must not repair a partially overwritten
    // x=0 GPU slot and hide a failed transaction's writes.
    grid.insert_brick(x_brick(1), vec![13; 8]).unwrap();
    assert_eq!(owner.mark_dirty(&[x_brick(1)]).unwrap(), 1);
    let probes = [x_brick(0), x_brick(1), x_brick(10)];
    let committed = snapshot(&owner, &probes);
    assert_eq!(committed.dirty, vec![x_brick(1)]);
    assert_eq!(
        owner.update_focus(&grid, position([20, 0, 0])),
        Err(WorldSdfFocusUpdateError::Apply(
            WorldSdfClipmapUploadError::Backend(GpuSdfError::UploadBudgetExceeded {
                required: 336,
                budget: 304
            })
        ))
    );
    assert_eq!(snapshot(&owner, &probes), committed);
    assert_center(&owner, x_brick(0));
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );
    assert_eq!(snapshot(&owner, &probes), committed);
    let flushed = owner.flush_dirty(&grid, 1, 48).unwrap();
    assert_eq!((flushed.uploaded_bytes, flushed.pending_bricks), (48, 0));
    assert_eq!(flushed.revision, committed.revision + 1);
    assert_center(&owner, x_brick(0));
    let after_failure = submit_copy(&context, &mut owner, &queries);

    // A one-brick roll must use the OLD backend planner and preserve x=0.
    let small = owner.update_focus(&grid, position([2, 0, 0])).unwrap();
    assert_eq!(small.applied.unwrap().uploaded_bytes, 96);
    assert_center(&owner, x_brick(1));
    let after_small_roll = submit_copy(&context, &mut owner, &queries);

    assert!(grid.remove_brick(x_brick(9)).unwrap());
    assert!(grid.remove_brick(x_brick(11)).unwrap());
    let retry = owner.update_focus(&grid, position([20, 0, 0])).unwrap();
    assert_eq!(retry.center, x_brick(10));
    let stats = retry
        .applied
        .expect("retry must publish the previously rejected focus");
    assert_eq!((stats.uploaded_bytes, stats.pending_bricks), (304, 0));
    assert_center(&owner, x_brick(10));
    assert_eq!(owner.ring().brick_count(), 1);
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    let after_retry = submit_copy(&context, &mut owner, &queries);

    // No device wait until all later uploads/submissions have been staged.
    assert_eq!(
        before_failure.read(&context),
        vec![Value(7), Unknown, Unknown]
    );
    assert_eq!(
        after_failure.read(&context),
        vec![Value(7), Value(13), Unknown]
    );
    assert_eq!(
        after_small_roll.read(&context),
        vec![Value(7), Value(13), Unknown]
    );
    assert_eq!(
        after_retry.read(&context),
        vec![Unknown, Unknown, Value(22)]
    );
}

#[test]
fn stationary_mismatch_and_pending_edits_fail_closed_flush_and_deletion_recover() {
    let context = context();
    // One i16 sample still occupies a padded four-byte GPU sample word.
    let mut grid = SparseSdfBrickGrid::new(1, 2).unwrap();
    grid.insert_brick(x_brick(0), vec![-13]).unwrap();
    grid.insert_brick(brick(0, 0, 0, 1), vec![29]).unwrap();
    let mut owner =
        WorldSdfGpuFocus::new(&context, focus(10, 1, [2, 0, 4]), config(2, 1), 3, 4096).unwrap();
    let first = owner.update_focus(&grid, ORIGIN).unwrap().applied.unwrap();
    assert_eq!(
        (first.uploaded_bytes, first.pending_bricks),
        (18 * 32 + 2 * 4, 0)
    );
    let allocated = owner.ring().allocated_buffer_bytes();
    grid.insert_brick(x_brick(0), vec![101]).unwrap();
    grid.insert_brick(x_brick(1), vec![55]).unwrap();
    assert_eq!(owner.mark_dirty(&[x_brick(0), x_brick(1)]).unwrap(), 2);
    let probes = [x_brick(0), x_brick(1), brick(0, 0, 0, 1)];
    let pending = snapshot(&owner, &probes);
    let wrong_size = SparseSdfBrickGrid::new(2, 2).unwrap();
    let wrong_mips = SparseSdfBrickGrid::new(1, 1).unwrap();
    for wrong in [&wrong_size, &wrong_mips] {
        for at in [ORIGIN, position([11, 0, 13])] {
            assert_eq!(
                owner.update_focus(wrong, at),
                Err(WorldSdfFocusUpdateError::Apply(
                    WorldSdfClipmapUploadError::GridMismatch
                ))
            );
            assert_eq!(snapshot(&owner, &probes), pending);
        }
        assert_eq!(
            owner.flush_dirty(wrong, 2, 4096),
            Err(WorldSdfClipmapUploadError::GridMismatch)
        );
        assert_eq!(snapshot(&owner, &probes), pending);
    }
    for at in [ORIGIN, position([11, 0, 13])] {
        let held = owner.update_focus(&grid, at).unwrap();
        assert_eq!(held.center, x_brick(0));
        assert_eq!(held.applied, None);
        assert_eq!(snapshot(&owner, &probes), pending);
    }
    let queries = [
        query([0, 0, 0], 0),
        query([0, 0, 0], 1),
        query([1, 0, 0], 0),
    ];
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 2 },
    );
    assert_eq!(snapshot(&owner, &probes), pending);
    assert_eq!(
        owner.flush_dirty(&grid, 1, 35),
        Err(WorldSdfClipmapUploadError::UploadBudgetTooSmall {
            required: 36,
            budget: 35
        })
    );
    assert_eq!(snapshot(&owner, &probes), pending);
    let first_batch = owner.flush_dirty(&grid, 1, 36).unwrap();
    assert_eq!(
        (first_batch.uploaded_bytes, first_batch.pending_bricks),
        (36, 1)
    );
    assert_eq!(owner.ring().scheduler().dirty_bricks(), &[x_brick(0)]);
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );
    let second_batch = owner.flush_dirty(&grid, 1, 36).unwrap();
    assert_eq!(
        (second_batch.uploaded_bytes, second_batch.pending_bricks),
        (36, 0)
    );
    assert_center(&owner, x_brick(0));
    assert_eq!(owner.ring().brick_count(), 3);
    let before_deletion = submit_copy(&context, &mut owner, &queries);

    assert!(grid.remove_brick(x_brick(0)).unwrap());
    owner.mark_dirty(&[x_brick(0)]).unwrap();
    let deleted_pending = snapshot(&owner, &probes);
    assert_eq!(owner.update_focus(&grid, ORIGIN).unwrap().applied, None);
    assert_eq!(snapshot(&owner, &probes), deleted_pending);
    assert_rejected(
        &context,
        &mut owner,
        &queries,
        WorldSdfClipmapUploadError::PendingDirty { bricks: 1 },
    );
    let deleted = owner.flush_dirty(&grid, 1, 32).unwrap();
    assert_eq!((deleted.uploaded_bytes, deleted.pending_bricks), (32, 0));
    assert_center(&owner, x_brick(0));
    assert_eq!(owner.ring().brick_count(), 2);
    assert_eq!(owner.ring().allocated_buffer_bytes(), allocated);
    let after_deletion = submit_copy(&context, &mut owner, &queries);
    assert_eq!(
        before_deletion.read(&context),
        vec![Value(101), Value(29), Value(55)]
    );
    assert_eq!(
        after_deletion.read(&context),
        vec![Unknown, Value(29), Value(55)]
    );
}

#[test]
fn mapping_and_ring_overflow_do_not_advance_focus_or_discard_pending_edits() {
    let context = context();
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(x_brick(0), vec![7; 8]).unwrap();
    let mut owner =
        WorldSdfGpuFocus::new(&context, focus(1, 2, [1, 0, 1]), config(1, 1), 1, 4096).unwrap();
    owner.update_focus(&grid, ORIGIN).unwrap();
    grid.insert_brick(x_brick(1), vec![13; 8]).unwrap();
    owner.mark_dirty(&[x_brick(1)]).unwrap();
    let probes = [x_brick(0), x_brick(1)];
    let pending = snapshot(&owner, &probes);
    let above = (i64::from(i32::MAX) + 1) * 2;
    let below = (i64::from(i32::MIN) - 1) * 2;
    for (relative, axis) in [
        ([above, 0, 0], ClipmapAxis::X),
        ([below, 0, 0], ClipmapAxis::X),
        ([0, above, 0], ClipmapAxis::Y),
        ([0, below, 0], ClipmapAxis::Y),
        ([0, 0, above], ClipmapAxis::Z),
        ([0, 0, below], ClipmapAxis::Z),
    ] {
        assert_eq!(
            owner.update_focus(&grid, position(relative)),
            Err(WorldSdfFocusUpdateError::Focus(
                WorldSdfFocusError::CoordinateOverflow { axis }
            ))
        );
        assert_eq!(snapshot(&owner, &probes), pending);
    }
    // The center itself fits i32, but radius-one X/Z endpoints do not.
    for relative in [
        [i64::from(i32::MAX) * 2, 0, 0],
        [0, 0, i64::from(i32::MIN) * 2],
    ] {
        assert!(matches!(
            owner.update_focus(&grid, position(relative)),
            Err(WorldSdfFocusUpdateError::Apply(
                WorldSdfClipmapUploadError::Scheduler(_)
            ))
        ));
        assert_eq!(snapshot(&owner, &probes), pending);
    }
    owner.flush_dirty(&grid, 1, 48).unwrap();
    assert_eq!(
        submit_copy(&context, &mut owner, &[query([0, 0, 0], 0)]).read(&context),
        vec![Value(7)]
    );

    let extreme = WorldSdfFocusConfig::new([i64::MIN, i64::MAX, 0], 1, 2, [0; 3]).unwrap();
    let mut uninitialized =
        WorldSdfGpuFocus::new(&context, extreme, config(1, 1), 1, 4096).unwrap();
    let initial = snapshot(&uninitialized, &probes);
    for (at, axis) in [
        ([i64::MAX, i64::MAX, 0], ClipmapAxis::X),
        ([i64::MIN, i64::MIN, 0], ClipmapAxis::Y),
    ] {
        assert_eq!(
            uninitialized.update_focus(&grid, at),
            Err(WorldSdfFocusUpdateError::Focus(
                WorldSdfFocusError::CoordinateOverflow { axis }
            ))
        );
        assert_eq!(snapshot(&uninitialized, &probes), initial);
    }
}

#[test]
fn gpu_coverage_rejects_y_deadband_and_radius_zero_xz_but_cpu_driver_allows_xyz() {
    let context = context();
    for (margins, radius, axis) in [
        ([0, 1, 0], 1, ClipmapAxis::Y),
        ([1, 0, 0], 0, ClipmapAxis::X),
        ([0, 0, 1], 0, ClipmapAxis::Z),
        ([1, 1, 1], 1, ClipmapAxis::Y),
    ] {
        let mapping = focus(10, 2, margins);
        let mut cpu = WorldSdfFocusDriver::new(mapping).unwrap();
        let initial = cpu
            .update_with(position([-1, -1, -1]), |_| Ok::<(), ()>(()))
            .unwrap();
        assert_eq!(initial.center, brick(-1, -1, -1, 0));
        // Every enabled CPU axis may retain brick -1 at relative zero.
        let held = cpu.update_with(ORIGIN, |_| Ok::<(), ()>(())).unwrap();
        assert_eq!(
            held.center,
            brick(
                if margins[0] == 0 { 0 } else { -1 },
                if margins[1] == 0 { 0 } else { -1 },
                if margins[2] == 0 { 0 } else { -1 },
                0,
            )
        );
        match WorldSdfGpuFocus::new(&context, mapping, config(1, radius), 1, 4096) {
            Err(error) => assert_eq!(
                error,
                WorldSdfFocusUpdateError::Focus(WorldSdfFocusError::InvalidCoverage { axis })
            ),
            Ok(_) => panic!("GPU focus accepted uncovered margins {margins:?} at radius {radius}"),
        }
    }
    for (margins, radius) in [([0; 3], 0), ([1, 0, 0], 1), ([0, 0, 1], 1), ([3, 0, 7], 1)] {
        let owner =
            WorldSdfGpuFocus::new(&context, focus(10, 2, margins), config(1, radius), 1, 4096)
                .unwrap();
        assert_eq!(owner.focus().center(), None);
        assert_eq!(owner.ring().scheduler().center(), None);
        assert!(owner.ring().allocated_buffer_bytes() > 0);
    }
}
