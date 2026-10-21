//! Owner polling cost with a gated real worker and a fully occupied task index.

use core::{convert::Infallible, hint::black_box};
use prism_tasks::{CancelToken, TaskPool};
use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfRebakeError, SdfRebakeLimits, SdfRebakeRegion, SdfRebakeTaskError,
    WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_physics_cell_runtime::{
    SdfCellCodec, SdfCellCodecLimits, SdfCellRebakeBudget, SdfCellRebakeError,
    SdfCellRebakeRuntime, SdfCellSnapshot, SdfPhysicsCellHost, SdfResidencyBudget,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use prism_world_runtime::SceneHost;
use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};
use std::{
    io::{self, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

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

fn measure(capacity: usize) {
    let pool = TaskPool::with_threads(1);
    assert_eq!(pool.worker_count(), 1, "benchmark requires a real worker");
    let sector = SectorId::new(0, 0, 0, 0);
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![3; 8])
        .expect("brick");
    let source = SdfCellSnapshot::new(
        sector,
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 100, 10).expect("transform"),
            1,
            grid,
        ),
    );
    let codec = SdfCellCodec::new(SdfCellCodecLimits::new(4096, 1, 8, 80));
    let payload = codec.encode(&source).expect("encode");
    let manifest = CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(200, 200, 200))
            .expect("bounds"),
        scene_key: 1,
        content_version: 1,
        derived_hash: 1,
        io_bytes: 0,
        data_layers: DataLayerMask::EMPTY,
    };
    let mut downstream = Host;
    let mut physics = SdfPhysicsCellHost::new(
        &mut downstream,
        codec,
        SdfPhysicsBridgeBudget::new(1, 8, 80).expect("budget"),
        SdfResidencyBudget::new(1, 1, 8, 80),
    );
    physics.load_cell(&manifest, &payload).expect("load");
    let before = physics.resident(sector).expect("resident").clone();
    let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(
        physics,
        &pool,
        SdfCellRebakeBudget {
            max_tasks: capacity,
            max_samples: capacity * 8,
            max_scratch_bytes: capacity * 4096,
        },
    )
    .expect("runtime");
    let region = SdfRebakeRegion {
        min_voxel: [0; 3],
        max_voxel: [0; 3],
        mip: 0,
    };
    let limits = SdfRebakeLimits {
        max_bricks: 1,
        max_samples: 8,
        max_scratch_bytes: 4096,
    };
    let scope = CancelToken::new();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let mut first = true;
    let first_id = runtime
        .start(sector, 2, region, limits, &scope, move |_| {
            if first {
                first = false;
                started_tx.send(()).expect("started");
                release_rx
                    .recv_timeout(Duration::from_secs(60))
                    .expect("release");
            }
            Ok(7)
        })
        .expect("start");
    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("worker started");
    let mut ids = vec![first_id];
    for _ in 1..capacity {
        ids.push(
            runtime
                .start(sector, 2, region, limits, &scope, |_| Ok(7))
                .expect("queued"),
        );
    }
    let usage = runtime.pending_usage();
    assert_eq!((usage.0, usage.1), (capacity, capacity * 8));
    let id = ids.last().expect("last task");
    let mut timings = Vec::with_capacity(128);
    for trial in 0..136 {
        let start = Instant::now();
        for _ in 0..256 {
            assert!(black_box(runtime.try_commit(black_box(id)).expect("pending")).is_none());
        }
        let elapsed = start.elapsed().as_nanos() / 256;
        if trial >= 8 {
            timings.push(elapsed);
        }
    }
    assert_eq!(runtime.pending_usage(), usage);
    runtime.cancel_all();
    assert_eq!(runtime.pending_usage(), usage);
    release_tx.send(()).expect("release");
    for id in ids {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !runtime.is_finished(&id).expect("known task") {
            assert!(Instant::now() < deadline, "worker completion deadline");
            thread::yield_now();
        }
        assert!(matches!(
            runtime.try_commit(&id),
            Err(SdfCellRebakeError::Task(SdfRebakeTaskError::Rebake(
                SdfRebakeError::Cancelled
            )))
        ));
    }
    assert_eq!(runtime.pending_usage(), (0, 0, 0));
    assert_eq!(runtime.physics().resident(sector), Some(&before));
    timings.sort_unstable();
    writeln!(io::stdout().lock(),
        "pending={capacity} samples={} scratch={} real_workers=1 trials=128 warmups=8 polls_per_trial=256 pending_try_commit P50/P95/P99 ns={}/{}/{}",
        usage.1, usage.2, timings[63], timings[121], timings[126]).expect("benchmark report");
}

fn main() {
    for capacity in [1, 64, 1024] {
        measure(capacity);
    }
}
