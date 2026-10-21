//! Cell lifecycle and aggregate admission remain atomic around background SDF work.

use core::convert::Infallible;
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use prism_tasks::{CancelToken, TaskPool};
use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfQueryResult, SdfRebakeError, SdfRebakeLimits, SdfRebakeRegion,
    SdfRebakeTaskError, WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_physics_cell_runtime::{
    SdfCellCodec, SdfCellCodecLimits, SdfCellRebakeBudget, SdfCellRebakeError,
    SdfCellRebakeResource, SdfCellRebakeRuntime, SdfCellSnapshot, SdfPhysicsCellHost,
    SdfPhysicsCellHostError, SdfResidencyBudget,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use prism_world_runtime::SceneHost;
use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};

struct Host;

impl SceneHost for Host {
    type Error = Infallible;

    fn load_cell(&mut self, _: &CellManifest, _: &[u8]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn evict_cell(&mut self, _: SectorId) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn sector(x: i64) -> SectorId {
    SectorId::new(0, x, 0, 0)
}

fn snapshot(sector: SectorId, revision: u64, value: i16) -> SdfCellSnapshot {
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![value; 8])
        .expect("brick");
    SdfCellSnapshot::new(
        sector,
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 100, 10).expect("transform"),
            revision,
            grid,
        ),
    )
}

fn manifest(sector: SectorId, revision: u64) -> CellManifest {
    CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(200, 200, 200))
            .expect("bounds"),
        scene_key: 1,
        content_version: revision,
        derived_hash: u128::from(revision),
        io_bytes: 0,
        data_layers: DataLayerMask::EMPTY,
    }
}

fn codec() -> SdfCellCodec {
    SdfCellCodec::new(SdfCellCodecLimits::new(4096, 2, 16, 160))
}

fn physics<'a>(
    host: &'a mut Host,
    cells: usize,
    bricks: usize,
) -> SdfPhysicsCellHost<'a, Host, SdfCellCodec> {
    SdfPhysicsCellHost::new(
        host,
        codec(),
        SdfPhysicsBridgeBudget::new(2, 16, 160).expect("bridge budget"),
        SdfResidencyBudget::new(cells, bricks, bricks * 8, bricks * 80),
    )
}

fn load(host: &mut SdfPhysicsCellHost<'_, Host, SdfCellCodec>, sector: SectorId, revision: u64) {
    let payload = codec()
        .encode(&snapshot(sector, revision, revision as i16))
        .unwrap();
    host.load_cell(&manifest(sector, revision), &payload)
        .expect("cell load");
}

fn limits() -> SdfRebakeLimits {
    SdfRebakeLimits {
        max_bricks: 2,
        max_samples: 16,
        max_scratch_bytes: 4096,
    }
}

fn region(voxel: i32) -> SdfRebakeRegion {
    SdfRebakeRegion {
        min_voxel: [voxel; 3],
        max_voxel: [voxel; 3],
        mip: 0,
    }
}

/// Successful reload invalidates old work but its charge remains until collection.
#[test]
fn reload_invalidates_gated_work_without_releasing_its_budget_early() {
    let pool = TaskPool::with_threads(1);
    if pool.worker_count() == 0 {
        return; // The single-thread feature has no worker to gate.
    }
    let key = sector(0);
    let mut downstream = Host;
    let mut physics = physics(&mut downstream, 1, 2);
    load(&mut physics, key, 1);
    let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 1,
            max_samples: 8,
            max_scratch_bytes: 4096,
        },
    )
    .expect("runtime");
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let id = runtime
        .start(
            key,
            2,
            region(0),
            limits(),
            &CancelToken::new(),
            move |_| {
                started_tx.send(()).expect("worker starts");
                release_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("release worker");
                Ok(7)
            },
        )
        .expect("start");
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("worker entered source");
    assert_eq!(runtime.physics().resident(key).unwrap().revision(), 1);
    assert!(runtime.try_commit(&id).unwrap().is_none());
    assert_eq!(runtime.pending_usage().0, 1);

    let payload = codec().encode(&snapshot(key, 2, 9)).unwrap();
    runtime
        .load_cell(&manifest(key, 2), &payload)
        .expect("successful reload");
    assert_eq!(runtime.physics().resident(key).unwrap().revision(), 2);
    assert_eq!(
        runtime.pending_usage().0,
        1,
        "invalidation is not collection"
    );
    assert!(matches!(
        runtime.start(key, 3, region(0), limits(), &CancelToken::new(), |_| Ok(1)),
        Err(SdfCellRebakeError::BudgetExceeded {
            resource: SdfCellRebakeResource::Tasks,
            ..
        })
    ));
    release_tx.send(()).expect("release worker");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !runtime.is_finished(&id).unwrap() {
        assert!(Instant::now() < deadline, "worker completion deadline");
        thread::yield_now();
    }
    assert!(matches!(
        runtime.try_commit(&id),
        Err(SdfCellRebakeError::Invalidated { sector }) if sector == key
    ));
    assert_eq!(runtime.pending_usage(), (0, 0, 0));
    assert_eq!(runtime.physics().resident(key).unwrap().revision(), 2);
}

/// Concurrent growth admission is rechecked at commit and rejected without partial writes.
#[test]
fn competing_cells_recheck_aggregate_resident_budget_atomically() {
    let pool = TaskPool::with_threads(0);
    let first = sector(0);
    let second = sector(1);
    let mut downstream = Host;
    let mut physics = physics(&mut downstream, 2, 3);
    load(&mut physics, first, 1);
    load(&mut physics, second, 1);
    let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 2,
            max_samples: 32,
            max_scratch_bytes: 8192,
        },
    )
    .unwrap();
    let first_old = runtime.physics().resident(first).unwrap().clone();
    let second_old = runtime.physics().resident(second).unwrap().clone();
    let a = runtime
        .start(first, 2, region(2), limits(), &CancelToken::new(), |_| {
            Ok(11)
        })
        .unwrap();
    let b = runtime
        .start(second, 2, region(2), limits(), &CancelToken::new(), |_| {
            Ok(12)
        })
        .unwrap();
    assert_eq!(runtime.physics().resident_usage(), (2, 2, 16, 160));
    assert_eq!(runtime.try_commit(&a).unwrap().unwrap().revision, 2);
    assert_eq!(runtime.physics().resident_usage(), (2, 3, 24, 192));
    assert!(matches!(
        runtime.try_commit(&b),
        Err(SdfCellRebakeError::BudgetExceeded {
            resource: SdfCellRebakeResource::ResidentBricks,
            required: 4,
            limit: 3,
        })
    ));
    assert_eq!(runtime.physics().resident(first).unwrap().revision(), 2);
    assert_eq!(runtime.physics().resident(second), Some(&second_old));
    assert_ne!(runtime.physics().resident(first), Some(&first_old));
    assert_eq!(runtime.physics().resident_usage(), (2, 3, 24, 192));
    assert_eq!(runtime.pending_usage(), (0, 0, 0));
}

/// Failed reloads keep both the old query and the in-flight publication eligible.
#[test]
fn failed_reload_preserves_background_work_and_old_query() {
    let pool = TaskPool::with_threads(0);
    let key = sector(0);
    let mut downstream = Host;
    let mut physics = physics(&mut downstream, 1, 2);
    load(&mut physics, key, 1);
    let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 1,
            max_samples: 8,
            max_scratch_bytes: 4096,
        },
    )
    .unwrap();
    let id = runtime
        .start(key, 2, region(0), limits(), &CancelToken::new(), |_| Ok(17))
        .unwrap();
    let mut corrupt = codec().encode(&snapshot(key, 3, 3)).unwrap();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    assert!(matches!(
        runtime.load_cell(&manifest(key, 3), &corrupt),
        Err(SdfPhysicsCellHostError::Decoder(_))
    ));
    assert_eq!(runtime.physics().resident(key).unwrap().revision(), 1);
    assert_eq!(runtime.pending_usage().0, 1);
    assert_eq!(runtime.try_commit(&id).unwrap().unwrap().revision, 2);
    assert_eq!(runtime.physics().resident(key).unwrap().revision(), 2);
}

/// Foreign and already-collected IDs cannot release another task's accounting.
#[test]
fn task_ids_are_runtime_scoped_and_collected_once() {
    let pool = TaskPool::with_threads(0);
    let key = sector(0);
    let mut downstream_a = Host;
    let mut physics_a = physics(&mut downstream_a, 1, 2);
    load(&mut physics_a, key, 1);
    let mut a = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics_a,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 1,
            max_samples: 8,
            max_scratch_bytes: 4096,
        },
    )
    .unwrap();
    let id = a
        .start(key, 2, region(0), limits(), &CancelToken::new(), |_| Ok(2))
        .unwrap();

    let mut downstream_b = Host;
    let mut physics_b = physics(&mut downstream_b, 1, 2);
    load(&mut physics_b, key, 1);
    let mut b = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics_b,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 1,
            max_samples: 8,
            max_scratch_bytes: 4096,
        },
    )
    .unwrap();
    assert!(matches!(
        b.try_commit(&id),
        Err(SdfCellRebakeError::UnknownTask)
    ));
    assert_eq!(a.try_commit(&id).unwrap().unwrap().revision, 2);
    assert!(matches!(
        a.try_commit(&id),
        Err(SdfCellRebakeError::UnknownTask)
    ));
}

/// Cancellation does not release an active task's bounded source capacity early.
#[test]
fn cancellation_remains_charged_until_worker_is_collected() {
    let pool = TaskPool::with_threads(1);
    if pool.worker_count() == 0 {
        return; // Cancellation accounting is covered inline by neighboring tests.
    }
    let key = sector(0);
    let mut downstream = Host;
    let mut physics = physics(&mut downstream, 1, 2);
    load(&mut physics, key, 1);
    let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: 1,
            max_samples: 8,
            max_scratch_bytes: 4096,
        },
    )
    .unwrap();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let scope = CancelToken::new();
    let id = runtime
        .start(key, 2, region(0), limits(), &scope, move |_| {
            started_tx.send(()).expect("worker starts");
            release_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("release worker");
            Ok(7)
        })
        .unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("worker entered source");
    runtime.cancel(&id).unwrap();
    assert_eq!(runtime.try_commit(&id).unwrap(), None);
    assert_eq!(runtime.pending_usage().0, 1);
    release_tx.send(()).expect("release worker");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !runtime.is_finished(&id).unwrap() {
        assert!(Instant::now() < deadline, "worker completion deadline");
        thread::yield_now();
    }
    assert!(matches!(
        runtime.try_commit(&id),
        Err(SdfCellRebakeError::Task(SdfRebakeTaskError::Rebake(
            SdfRebakeError::Cancelled
        )))
    ));
    assert_eq!(runtime.pending_usage(), (0, 0, 0));
    assert_eq!(
        runtime.physics().query_sample(key, [0; 3], 0),
        Ok(SdfQueryResult::Value(1))
    );
}
