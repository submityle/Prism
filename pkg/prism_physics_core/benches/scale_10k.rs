//! Offline throughput benchmark for a ten-thousand-body scene (M3 scale goal).
//!
//! This benchmark is intentionally dependency-free: it uses a
//! `harness = false` plain `main` and [`std::time::Instant`] rather than an
//! external harness, so it builds and runs fully offline with no extra crates.
//! Run it with:
//!
//! ```text
//! cargo bench -p prism_physics_core --bench scale_10k
//! # with the parallel island solver:
//! cargo bench -p prism_physics_core --bench scale_10k --features parallel
//! ```
//!
//! It builds a 100 x 100 field of boxes on a ground plane (10,000 dynamic
//! bodies), then reports the average per-step time over two windows: an early
//! "all awake" window where every body is being solved, and a later "settled"
//! window where sleeping has collapsed the active set. The gap between the two
//! is the payoff of M3's island + sleeping machinery.
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains **no Unreal
//! Engine source or derived code**.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use glam::Vec3;
use prism_physics_core::{
    BodyDesc, ColliderShape, PhysicsMaterial, PhysicsWorld, Solver, XpbdSolver,
};
use std::time::Instant;

/// Field side length in boxes; the field holds `GRID * GRID` bodies.
const GRID: usize = 100;
/// Half-extent of every box.
const HALF_EXTENT: f32 = 0.5;
/// Horizontal spacing so boxes never touch (each is its own island).
const SPACING: f32 = 2.0;
/// Fixed simulation timestep.
const DT: f32 = 1.0 / 60.0;
/// Sub-steps per full step.
const SUBSTEPS: u32 = 4;

/// Builds the ground plane plus the box field and returns the world.
fn build_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    let plane = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.0,
    });
    world.spawn(
        BodyDesc::static_at(Vec3::ZERO)
            .with_collider(plane)
            .with_material(PhysicsMaterial::DEFAULT),
    );

    let box_shape = ColliderShape::Cuboid {
        half_extents: Vec3::splat(HALF_EXTENT),
    };
    let box_handle = world.shapes.insert(box_shape);
    let mass_properties = box_shape.mass_properties(1.0);

    let origin = -(GRID as f32 - 1.0) * 0.5 * SPACING;
    for ix in 0..GRID {
        for iz in 0..GRID {
            let x = origin + ix as f32 * SPACING;
            let z = origin + iz as f32 * SPACING;
            world.spawn(
                BodyDesc::dynamic_at(Vec3::new(x, HALF_EXTENT + 0.05, z))
                    .with_collider(box_handle)
                    .with_mass_properties(mass_properties)
                    .with_material(PhysicsMaterial::DEFAULT),
            );
        }
    }

    world
}

/// Steps the world `count` times and returns the average per-step time in
/// milliseconds.
fn time_steps(world: &mut PhysicsWorld, solver: &mut XpbdSolver, count: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..count {
        solver.step(world, DT, SUBSTEPS);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(count)
}

fn main() {
    let bodies = GRID * GRID;
    let mut world = build_world();
    let mut solver = XpbdSolver::new();

    // Early window: the field is still settling, so nearly every body is awake
    // and solved every step.
    let awake_ms = time_steps(&mut world, &mut solver, 30);

    // Let the field settle and sleep.
    for _ in 0..120 {
        solver.step(&mut world, DT, SUBSTEPS);
    }

    // Settled window: sleeping has collapsed the active set.
    let settled_ms = time_steps(&mut world, &mut solver, 30);

    println!("prism_physics_core scale_10k benchmark");
    println!("  bodies:            {bodies}");
    println!("  substeps/step:     {SUBSTEPS}");
    println!("  awake avg:         {awake_ms:.3} ms/step");
    println!("  settled avg:       {settled_ms:.3} ms/step");
    if settled_ms > 0.0 {
        println!("  sleeping speedup:  {:.2}x", awake_ms / settled_ms);
    }
}
