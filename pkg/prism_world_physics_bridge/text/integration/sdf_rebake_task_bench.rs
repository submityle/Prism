//! Separate preparation, worker scheduling/evaluation, and owner commit costs.

extern crate alloc;

use alloc::vec::Vec;
use core::{convert::Infallible, hint::black_box};
use prism_tasks::{CancelToken, TaskPool};
use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfRebakeLimits, SdfRebakePoint, SdfRebakeRegion,
    WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use std::{
    io::{self, Write},
    thread,
    time::{Duration, Instant},
};

fn sample(point: SdfRebakePoint) -> Result<i16, Infallible> {
    Ok(
        ((point.world_position_mm[0] / 10 + point.world_position_mm[1] / 5
            - point.world_position_mm[2] / 10)
            .rem_euclid(65536)
            - 32768) as i16,
    )
}

fn measure(pool: &TaskPool, resident: i32, full: bool) {
    let mut grid = SparseSdfBrickGrid::new(8, 3).expect("grid");
    for x in 0..resident {
        grid.insert_brick(BrickCoord::new(x, 0, 0, 0), alloc::vec![0; 512])
            .expect("brick");
    }
    let mut bridge = WorldSdfPhysicsBridge::new(
        SdfPhysicsBridgeBudget::new(65536, 65536 * 512, 70_000_000).expect("budget"),
    );
    bridge
        .publish(WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 10, 2).expect("transform"),
            1,
            grid,
        ))
        .expect("publish");
    let edit = SdfRebakeRegion {
        min_voxel: [0; 3],
        max_voxel: [if full { resident * 8 - 1 } else { 0 }, 0, 0],
        mip: 0,
    };
    let limits = SdfRebakeLimits {
        max_bricks: 65536,
        max_samples: 65536 * 512,
        max_scratch_bytes: 70_000_000,
    };
    // Build all target keys before timing so every measured operation replaces.
    let baseline = bridge
        .rebake_region(edit, 2, limits, sample)
        .expect("initial");
    let oracle = bridge.current().expect("current").source().clone();
    let scope = CancelToken::new();
    let mut times: [Vec<u128>; 3] = core::array::from_fn(|_| Vec::with_capacity(128));
    for trial in 0..136 {
        let start = Instant::now();
        let work = bridge
            .prepare_rebake(black_box(edit), trial + 3, limits)
            .expect("prepare");
        let prepared = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut task = work.spawn(pool, &scope, |point| sample(black_box(point)));
        let deadline = Instant::now() + Duration::from_secs(30);
        while !task.is_finished() {
            assert!(Instant::now() < deadline, "worker deadline");
            thread::yield_now();
        }
        let computed = start.elapsed().as_nanos();
        let start = Instant::now();
        let report = task
            .try_commit(&mut bridge)
            .expect("commit")
            .expect("ready");
        let committed = start.elapsed().as_nanos();
        // Exact grid comparison is deliberately outside every timed phase.
        let current = bridge.current().expect("current");
        assert_eq!(current.grid(), oracle.grid());
        assert_eq!(report.rebaked_bricks, baseline.rebaked_bricks);
        assert_eq!(report.inserted_bricks, 0);
        if trial >= 8 {
            for (values, duration) in times.iter_mut().zip([prepared, computed, committed]) {
                values.push(duration);
            }
        }
    }
    writeln!(
        io::stdout().lock(),
        "workers={} resident={resident} full={full} bricks={} samples={} scratch={}",
        pool.worker_count(),
        baseline.rebaked_bricks,
        baseline.sampled_voxels,
        baseline.scratch_bytes
    )
    .expect("report");
    for (phase, mut values) in ["prepare", "spawn_to_finished", "commit"]
        .into_iter()
        .zip(times)
    {
        values.sort_unstable();
        writeln!(
            io::stdout().lock(),
            "{phase} P50/P95/P99 ns={}/{}/{}",
            values[63],
            values[121],
            values[126]
        )
        .expect("report");
    }
}

fn main() {
    let pool = TaskPool::with_threads(1);
    for resident in [64, 32768] {
        measure(&pool, resident, false);
    }
    measure(&pool, 4096, true);
}
