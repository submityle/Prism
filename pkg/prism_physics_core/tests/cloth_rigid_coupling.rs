//! CPU golden integration test for the cloth↔rigid two-way coupling bridge.
//!
//! These tests exercise the full pipeline wired in
//! [`prism_physics_core::soft::rigid_coupling`]: a soft particle sheet and a
//! rigid prop body coupled through [`PhysicsWorld::resolve_cloth_coupling`].
//! They prove **both directions** of the coupling as required:
//!
//! * (a) a light rigid prop resting on a pinned cloth sheet is pushed back up
//!   (its reaction impulse is non-zero and points `+Y`, and its velocity gains
//!   an upward component so it does not fall through);
//! * (b) the cloth visibly dents under the prop (an interior particle is pushed
//!   below the rest plane);
//! * (c) determinism (array-in / array-out, identical result run twice);
//! * (d) the rigid-only path is untouched when the coupling flag is off.
//!
//! # Provenance
//!
//! This test contains **no Unreal Engine source or derived code**.

use glam::Vec3;
use prism_physics_core::soft::rigid_coupling::ClothRigidCouplingConfig;
use prism_physics_core::state::body::{BodyDesc, BodyKind, MassProperties};
use prism_physics_core::{CouplingReport, PhysicsWorld, WorldConfig};

/// Builds a flat `n x n` cloth sheet on the `y = 0` plane spanning
/// `[-half, half]` in X and Z. Border particles are pinned (`inverse_mass = 0`),
/// interior particles are free (`inverse_mass = 1`).
fn build_cloth(n: usize, half: f32) -> (Vec<Vec3>, Vec<f32>) {
    let mut positions = Vec::with_capacity(n * n);
    let mut inverse_masses = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let fx = (c as f32) / ((n - 1) as f32); // 0..1
            let fz = (r as f32) / ((n - 1) as f32);
            let x = -half + fx * (2.0 * half);
            let z = -half + fz * (2.0 * half);
            positions.push(Vec3::new(x, 0.0, z));
            let border = r == 0 || c == 0 || r == n - 1 || c == n - 1;
            inverse_masses.push(if border { 0.0 } else { 1.0 });
        }
    }
    (positions, inverse_masses)
}

/// Spawns a light dynamic sphere prop that penetrates the `y = 0` sheet: its
/// center sits at `y = h` with `h < radius`, so the sheet is inside the sphere.
/// A large inverse mass makes the prop light so it moves far more than the
/// (effectively heavy) particles under the mass-weighted split.
fn spawn_prop(
    world: &mut PhysicsWorld,
    center_y: f32,
    radius: f32,
    inv_mass: f32,
) -> prism_physics_core::BodyHandle {
    let shape = world
        .shapes
        .insert(prism_physics_core::ColliderShape::Sphere { radius });
    let desc = BodyDesc {
        kind: BodyKind::Dynamic,
        position: Vec3::new(0.0, center_y, 0.0),
        collider: Some(shape),
        mass_properties: MassProperties {
            inv_mass,
            inv_inertia: Vec3::ZERO,
        },
        ..BodyDesc::default()
    };
    world.spawn(desc)
}

#[test]
fn prop_is_pushed_up_and_cloth_dents() {
    let mut world = PhysicsWorld::new(WorldConfig::default());
    world.cloth_coupling = ClothRigidCouplingConfig::active();
    // Light prop (inv_mass = 20) sitting low enough to penetrate the sheet:
    // center at y = 0.1, radius 0.4, so the sheet plane y = 0 is 0.1 inside.
    let handle = spawn_prop(&mut world, 0.1, 0.4, 20.0);

    let (mut positions, inverse_masses) = build_cloth(7, 1.0);
    let dt = 1.0 / 60.0;

    let report: CouplingReport =
        world.resolve_cloth_coupling(&mut positions, &inverse_masses, dt);

    // (a) The prop received a reaction impulse pointing up (+Y) and nothing was
    // culled: exactly one movable proxy was written back.
    assert_eq!(report.proxy_count, 1);
    assert_eq!(report.applied_count, 1);
    assert!(
        report.applied_impulse.y > 1e-4,
        "expected upward reaction impulse, got {:?}",
        report.applied_impulse
    );

    // The prop's velocity gained an upward component (so it is being pushed back
    // out of the sheet, not falling through).
    let v = world.bodies.linear_velocity(handle).expect("velocity");
    assert!(v.y > 1e-4, "expected upward velocity, got {v:?}");

    // The prop's center was lifted (translated by the pass displacement).
    let pos = world.bodies.position(handle).expect("position");
    assert!(
        pos.y > 0.1 + 1e-5,
        "expected prop lifted above its start, got {pos:?}"
    );

    // (b) The cloth dented: at least one interior particle was pushed below the
    // rest plane (down onto the lower hemisphere of the sphere).
    let min_y = positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    assert!(
        min_y < -1e-4,
        "expected cloth to dent below y=0, min_y = {min_y}"
    );

    // Lateral symmetry sanity: the net impulse is essentially vertical.
    assert!(report.applied_impulse.x.abs() < 1e-3);
    assert!(report.applied_impulse.z.abs() < 1e-3);
}

#[test]
fn coupling_is_deterministic() {
    let run = || {
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active();
        let handle = spawn_prop(&mut world, 0.1, 0.4, 20.0);
        let (mut positions, inverse_masses) = build_cloth(7, 1.0);
        let report = world.resolve_cloth_coupling(&mut positions, &inverse_masses, 1.0 / 60.0);
        let v = world.bodies.linear_velocity(handle).unwrap();
        let pos = world.bodies.position(handle).unwrap();
        (positions, report, v, pos)
    };
    let (pos_a, rep_a, v_a, p_a) = run();
    let (pos_b, rep_b, v_b, p_b) = run();
    assert_eq!(pos_a, pos_b);
    assert_eq!(rep_a, rep_b);
    assert_eq!(v_a, v_b);
    assert_eq!(p_a, p_b);
}

#[test]
fn flag_off_leaves_rigid_path_untouched() {
    let mut world = PhysicsWorld::new(WorldConfig::default());
    // Flag left at default (disabled).
    assert!(!world.cloth_coupling.is_enabled());
    let handle = spawn_prop(&mut world, 0.1, 0.4, 20.0);
    let start_pos = world.bodies.position(handle).unwrap();
    let start_vel = world.bodies.linear_velocity(handle).unwrap();

    let (mut positions, inverse_masses) = build_cloth(7, 1.0);
    let positions_before = positions.clone();

    let report = world.resolve_cloth_coupling(&mut positions, &inverse_masses, 1.0 / 60.0);

    // Empty report, particles untouched, rigid state untouched.
    assert_eq!(report, CouplingReport::default());
    assert_eq!(positions, positions_before);
    assert_eq!(world.bodies.position(handle).unwrap(), start_pos);
    assert_eq!(world.bodies.linear_velocity(handle).unwrap(), start_vel);
}

#[test]
fn kinematic_prop_records_reaction_but_does_not_move() {
    let mut world = PhysicsWorld::new(WorldConfig::default());
    world.cloth_coupling = ClothRigidCouplingConfig::active();
    let shape = world
        .shapes
        .insert(prism_physics_core::ColliderShape::Sphere { radius: 0.4 });
    let desc = BodyDesc {
        kind: BodyKind::Kinematic,
        position: Vec3::new(0.0, 0.1, 0.0),
        collider: Some(shape),
        ..BodyDesc::default()
    };
    let handle = world.spawn(desc);

    let (mut positions, inverse_masses) = build_cloth(7, 1.0);
    let report = world.resolve_cloth_coupling(&mut positions, &inverse_masses, 1.0 / 60.0);

    // A kinematic prop is a proxy (proxy_count == 1) but has zero inverse mass,
    // so it is never written back (applied_count == 0) and never moves.
    assert_eq!(report.proxy_count, 1);
    assert_eq!(report.applied_count, 0);
    assert_eq!(
        world.bodies.position(handle).unwrap(),
        Vec3::new(0.0, 0.1, 0.0)
    );
    // The cloth still dents against the immovable kinematic prop.
    let min_y = positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    assert!(min_y < -1e-4, "expected dent against kinematic prop");
}
