//! Fixed-input query throughput; all correctness checks precede timed batches.

use core::hint::black_box;
use std::{
    io::{self, Write},
    time::Instant,
};

use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfPhysicsSnapshot, SdfQueryResult, WorldSdfPhysicsBridge,
    WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

fn snapshot(populated: bool) -> SdfPhysicsSnapshot {
    let mut grid = SparseSdfBrickGrid::new(8, 1).expect("grid");
    if populated {
        for bz in 0..4 {
            for by in 0..4 {
                for bx in 0..4 {
                    let mut samples = Vec::with_capacity(512);
                    for z in 0..8 {
                        for y in 0..8 {
                            for x in 0..8 {
                                samples.push(
                                    (bx * 8 + x + 2 * (by * 8 + y) + 3 * (bz * 8 + z)) as i16,
                                );
                            }
                        }
                    }
                    grid.insert_brick(BrickCoord::new(bx, by, bz, 0), samples)
                        .expect("brick");
                }
            }
        }
    }
    WorldSdfPhysicsBridge::new(SdfPhysicsBridgeBudget::new(64, 32768, 66608).expect("budget"))
        .stage(WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 1000, 10).expect("transform"),
            1,
            grid,
        ))
        .expect("snapshot")
}

fn measure(label: &str, snapshot: &SdfPhysicsSnapshot, positions: &[[i64; 3]]) {
    let mut times = Vec::with_capacity(128);
    for _ in 0..128 {
        let start = Instant::now();
        for position in positions {
            black_box(
                snapshot
                    .query_world_distance_gradient_mm(black_box(*position), 0)
                    .expect("validated query"),
            );
        }
        times.push(start.elapsed().as_nanos() as f64 / positions.len() as f64);
    }
    times.sort_by(f64::total_cmp);
    writeln!(
        io::stdout().lock(),
        "{label}: {}x{} queries; P50/P95/P99 ns/query = {:.3}/{:.3}/{:.3}; logical bytes = {}",
        times.len(),
        positions.len(),
        times[63],
        times[121],
        times[126],
        snapshot.logical_bytes()
    )
    .expect("write benchmark report");
}

fn main() {
    let positions: Vec<[i64; 3]> = (0..4096)
        .map(|i| [(i * 17) % 31000, (i * 31) % 31000, (i * 47) % 31000])
        .collect();
    let resident = snapshot(true);
    let missing = snapshot(false);
    // Mixed interior/boundary positions have an independently known affine field.
    for position in &positions {
        let SdfQueryResult::Value(sample) = resident
            .query_world_distance_gradient_mm(*position, 0)
            .expect("query")
        else {
            panic!("resident field missing")
        };
        let expected = (position[0] + 2 * position[1] + 3 * position[2]) as f64 * 0.01;
        assert!((sample.distance_mm - expected).abs() < 1e-9);
        assert_eq!(sample.gradient, [0.01, 0.02, 0.03]);
        assert_eq!(
            missing.query_world_distance_gradient_mm(*position, 0),
            Ok(SdfQueryResult::Unknown)
        );
    }
    measure("resident 64 bricks", &resident, &positions);
    measure("missing first corner", &missing, &positions);
}
