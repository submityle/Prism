//! Hierarchy world-transform propagation throughput (roadmap M3 "基准即规格").
//!
//! The design spec (`docs/prism_transform_design_zh.md` §22) names *parallel
//! hierarchy propagation throughput* as an acceptance metric: spreading the
//! parent-before-child world sweep across depth levels onto a
//! [`prism_tasks::TaskPool`] via [`prism_transform::parallel`] should beat the
//! single-threaded [`prism_transform::propagation::propagate_in_order`] pass on
//! a wide forest. This benchmark builds a large forest, times both paths with a
//! cached [`LevelPlan`]/traversal order so only the propagation math is
//! measured, and reports the achieved scaling, with a guard that the parallel
//! world transforms are bit-identical to the serial reference so a "fast" run
//! that skipped or reordered work fails loudly.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_transform --bench hierarchy_propagation
//! ```
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains
//! **no Unreal Engine source or derived code**.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use std::hint::black_box;
use std::time::Instant;

use prism_math::{Quat, Vec3};
use prism_transform::hierarchy::Hierarchy;
use prism_transform::parallel::{LevelPlan, propagate_parallel_with_plan};
use prism_transform::propagation::propagate_in_order;
use prism_transform::{GlobalTransform, Transform};
use prism_tasks::TaskPool;

/// Roots in the forest (the widest level).
const ROOTS: usize = 2048;
/// Children spawned per node at each interior level.
const FANOUT: usize = 3;
/// Interior depth levels below the roots.
const DEPTH: usize = 5;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 48;

/// Build a wide forest and the matching local-transform buffer. Node ids are
/// handed out sequentially, so `locals[node.index()]` lines up by construction.
fn build_forest() -> (Hierarchy, Vec<Transform>) {
    let mut hierarchy = Hierarchy::new();
    let mut locals: Vec<Transform> = Vec::new();

    // A small ring of precomputed rotations cycled by index: avoids a million
    // trig calls at setup while still exercising non-identity rotation blends.
    let rotations: [Quat; 4] = [
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::X, 0.6),
        Quat::from_axis_angle(Vec3::new(0.3, 0.7, 0.2).normalize(), 0.9),
        Quat::from_axis_angle(Vec3::Y, 1.3),
    ];

    let push_local = |locals: &mut Vec<Transform>, i: usize| {
        let f = i as f32;
        // Trig-free deterministic spread so each node differs in every lane.
        let t = Vec3::new((f * 0.013) % 7.0 - 3.5, (f * 0.021) % 5.0 - 2.5, (f * 0.017) % 9.0 - 4.5);
        let s = 0.8 + ((i % 5) as f32) * 0.1;
        locals.push(
            Transform::from_translation(t)
                .with_rotation(rotations[i % rotations.len()])
                .with_scale(Vec3::splat(s)),
        );
    };

    let mut level: Vec<_> = Vec::with_capacity(ROOTS);
    for _ in 0..ROOTS {
        let id = hierarchy.spawn_root();
        push_local(&mut locals, id.index());
        level.push(id);
    }

    for _ in 0..DEPTH {
        let mut next = Vec::with_capacity(level.len() * FANOUT);
        for &parent in &level {
            for _ in 0..FANOUT {
                let id = hierarchy.spawn_child(parent);
                push_local(&mut locals, id.index());
                next.push(id);
            }
        }
        level = next;
    }

    (hierarchy, locals)
}

/// Best-of-`PASSES` wall time (seconds) for the serial in-order sweep.
fn bench_serial(
    hierarchy: &Hierarchy,
    order: &[prism_transform::hierarchy::NodeId],
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        propagate_in_order(hierarchy, black_box(order), black_box(locals), globals);
        black_box(&globals[globals.len() - 1]);
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

/// Best-of-`PASSES` wall time (seconds) for the parallel level sweep.
fn bench_parallel(
    pool: &TaskPool,
    plan: &LevelPlan,
    hierarchy: &Hierarchy,
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        propagate_parallel_with_plan(pool, plan, hierarchy, black_box(locals), globals);
        black_box(&globals[globals.len() - 1]);
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let (hierarchy, locals) = build_forest();
    let nodes = hierarchy.len();

    let order = hierarchy.compute_order().expect("forest has a valid order");
    let plan = LevelPlan::build(&hierarchy).expect("forest has a valid level plan");
    let pool = TaskPool::with_threads(cores);

    let mut serial = vec![GlobalTransform::IDENTITY; nodes];
    let mut parallel = vec![GlobalTransform::IDENTITY; nodes];

    // Warm caches / spawn worker threads before timing.
    propagate_in_order(&hierarchy, &order, &locals, &mut serial);
    propagate_parallel_with_plan(&pool, &plan, &hierarchy, &locals, &mut parallel);

    // Correctness guard: the parallel sweep must reproduce the serial world
    // transforms bit-for-bit (identical `Affine3` operand order), otherwise the
    // throughput number would be meaningless.
    for (i, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(
            a.0, b.0,
            "parallel world transform diverged from serial reference at node {i}"
        );
    }

    let serial_s = bench_serial(&hierarchy, &order, &locals, &mut serial);
    let parallel_s = bench_parallel(&pool, &plan, &hierarchy, &locals, &mut parallel);

    // Re-check after timing (the timed passes overwrite the buffers).
    for (i, (a, b)) in serial.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(a.0, b.0, "post-timing divergence at node {i}");
    }

    let serial_mps = nodes as f64 / serial_s / 1e6;
    let parallel_mps = nodes as f64 / parallel_s / 1e6;
    let speedup = serial_s / parallel_s;
    let efficiency = speedup / cores as f64 * 100.0;

    println!("prism_transform hierarchy_propagation (M3 parallel scaling)");
    println!("  nodes/pass    : {nodes}  (levels {})", plan.level_count());
    println!("  worker threads: {cores}");
    println!("  serial        : {:.3} ms  ({serial_mps:.1} Mnodes/s)", serial_s * 1e3);
    println!("  parallel      : {:.3} ms  ({parallel_mps:.1} Mnodes/s)", parallel_s * 1e3);
    println!("  speedup       : {speedup:.2}x  ({efficiency:.0}% parallel efficiency)");
}
