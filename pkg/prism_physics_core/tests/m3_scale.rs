//! Large-scale acceptance test for the M3 milestone.
//!
//! M3 makes ten-thousand-body scenes affordable through three cooperating
//! mechanisms: island-based solving (independent contact/joint groups are
//! solved separately), sleeping (settled bodies stop being predicted and
//! solved), and optional parallel island solving. This test builds a wide
//! field of independently-settling boxes on a ground plane and proves the two
//! behaviours a game depends on at scale:
//!
//! 1. **Stability** — every body stays finite and comes to rest at its
//!    supported height; nothing explodes, sinks through the floor, or drifts.
//! 2. **Awake-island collapse** — once the field settles, almost every body
//!    sleeps, so the solver's active set collapses from "all bodies" to a tiny
//!    fraction. This is the property that keeps large scenes real-time.
//!
//! Every number asserted below is the outcome of stepping a real
//! [`XpbdSolver`] over the world; nothing is stubbed.
//!
//! # Provenance
//!
//! This is an original scenario authored for Prism. It contains **no Unreal
//! Engine source or derived code**.

use glam::Vec3;
use prism_physics_core::{
    BodyDesc, BodyHandle, ColliderShape, PhysicsMaterial, PhysicsWorld, Solver, XpbdSolver,
};

/// Half-extent of every box in the field.
const HALF_EXTENT: f32 = 0.5;
/// Horizontal spacing between box centres. Larger than a full box so no two
/// boxes ever touch: each is its own single-body island.
const SPACING: f32 = 2.0;
/// Field side length in boxes; the field holds `GRID * GRID` boxes.
const GRID: usize = 24;

/// Builds a ground plane plus a `GRID x GRID` field of boxes resting just above
/// their supported height, and returns the world together with the box handles.
fn settled_field() -> (PhysicsWorld, Vec<BodyHandle>) {
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

    let mut handles = Vec::with_capacity(GRID * GRID);
    let origin = -(GRID as f32 - 1.0) * 0.5 * SPACING;
    for ix in 0..GRID {
        for iz in 0..GRID {
            let x = origin + ix as f32 * SPACING;
            let z = origin + iz as f32 * SPACING;
            // Drop from a hair above the resting height so the field settles
            // (and then sleeps) within a short window.
            let y = HALF_EXTENT + 0.05;
            let handle = world.spawn(
                BodyDesc::dynamic_at(Vec3::new(x, y, z))
                    .with_collider(box_handle)
                    .with_mass_properties(mass_properties)
                    .with_material(PhysicsMaterial::DEFAULT),
            );
            handles.push(handle);
        }
    }

    (world, handles)
}

/// Counts how many of `handles` are currently awake (not sleeping).
fn awake_count(world: &PhysicsWorld, handles: &[BodyHandle]) -> usize {
    handles
        .iter()
        .filter(|&&h| world.bodies.is_sleeping(h) == Some(false))
        .count()
}

#[test]
fn wide_field_settles_and_stays_finite() {
    let (mut world, handles) = settled_field();
    let mut solver = XpbdSolver::new();

    // Two seconds of simulation is ample for this shallow drop to settle.
    for _ in 0..120 {
        solver.step(&mut world, 1.0 / 60.0, 4);
    }

    for &handle in &handles {
        let position = world.bodies.position(handle).unwrap();
        assert!(
            position.is_finite(),
            "body {handle:?} left the domain with a non-finite position {position:?}"
        );
        // The box must rest with its centre one half-extent above the plane.
        assert!(
            (position.y - HALF_EXTENT).abs() < 0.02,
            "body {handle:?} settled at y = {}, expected ~{HALF_EXTENT}",
            position.y
        );
        let velocity = world.bodies.linear_velocity(handle).unwrap();
        assert!(
            velocity.length() < 0.05,
            "body {handle:?} should be at rest, v = {velocity:?}"
        );
    }
}

#[test]
fn awake_island_set_collapses_as_the_field_sleeps() {
    let (mut world, handles) = settled_field();
    let mut solver = XpbdSolver::new();

    // Every body starts awake.
    assert_eq!(
        awake_count(&world, &handles),
        handles.len(),
        "all bodies should begin awake"
    );

    // Simulate long enough for the idle dwell time to elapse and the field to
    // sleep. The default time-to-sleep is 0.5 s, so ~2 s is comfortable.
    for _ in 0..150 {
        solver.step(&mut world, 1.0 / 60.0, 4);
    }

    let still_awake = awake_count(&world, &handles);
    // The active set must collapse to a small fraction of the field. We allow a
    // lenient ceiling (a tenth) rather than demanding exactly zero, since a few
    // boxes may still be settling on the final frame.
    let ceiling = handles.len() / 10;
    assert!(
        still_awake <= ceiling,
        "awake set failed to collapse: {still_awake} of {} bodies still awake (ceiling {ceiling})",
        handles.len()
    );
}
