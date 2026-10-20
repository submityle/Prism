//! Local update scaling with resident size; source setup and checks are untimed.

extern crate alloc;

use alloc::vec::Vec;
use core::{convert::Infallible, hint::black_box};
use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfRebakeLimits, SdfRebakePoint, SdfRebakeRegion,
    WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use std::{
    io::{self, Write},
    time::Instant,
};

fn sample(point: SdfRebakePoint) -> Result<i16, Infallible> {
    // A cheap deterministic field keeps staging/commit costs visible.
    Ok(
        ((point.world_position_mm[0] / 10 + point.world_position_mm[1] / 5
            - point.world_position_mm[2] / 10)
            .rem_euclid(65536)
            - 32768) as i16,
    )
}

fn measure(resident: i32, full: bool) {
    let mut grid = SparseSdfBrickGrid::new(8, 3).expect("grid");
    for x in 0..resident {
        grid.insert_brick(BrickCoord::new(x, 0, 0, 0), vec![0; 512])
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
    let report = bridge
        .rebake_region(edit, 2, limits, sample)
        .expect("initial update");
    // Validate every staged sample against the field before timing.
    let snapshot = bridge.current().expect("current");
    for coord in snapshot.grid().allocated_bricks() {
        if !full && coord != BrickCoord::new(0, 0, 0, coord.mip) {
            continue;
        }
        let values = snapshot
            .grid()
            .brick_samples(coord)
            .expect("coord")
            .expect("resident");
        for (index, value) in values.iter().enumerate() {
            let voxel = [
                i64::from(coord.x) * 8 + (index % 8) as i64,
                ((index / 8) % 8) as i64,
                (index / 64) as i64,
            ];
            let spacing = 10_i128 << coord.mip;
            assert_eq!(
                *value,
                sample(SdfRebakePoint {
                    brick: coord,
                    voxel,
                    world_position_mm: voxel.map(|v| i128::from(v) * spacing),
                    voxel_size_mm: spacing,
                    distance_scale_mm: 2
                })
                .expect("sample")
            );
        }
    }
    let mut times = Vec::with_capacity(128);
    for trial in 0..136 {
        let start = Instant::now();
        let result = bridge
            .rebake_region(black_box(edit), trial + 3, limits, |p| sample(black_box(p)))
            .expect("bounded update");
        let elapsed = start.elapsed().as_nanos();
        if trial >= 8 {
            times.push(elapsed);
        }
        assert_eq!(result.inserted_bricks, 0);
        assert_eq!(result.rebaked_bricks, report.rebaked_bricks);
    }
    times.sort_unstable();
    writeln!(io::stdout().lock(), "resident_base_bricks={resident} full={full} rebaked={} samples={} scratch_bytes={} logical_bytes={}; P50/P95/P99 ns={}/{}/{}", report.rebaked_bricks, report.sampled_voxels, report.scratch_bytes, report.logical_bytes, times[63], times[121], times[126]).expect("report");
}

fn main() {
    for resident in [64, 4096, 32768] {
        measure(resident, false);
    }
    for resident in [64, 4096] {
        measure(resident, true);
    }
}
