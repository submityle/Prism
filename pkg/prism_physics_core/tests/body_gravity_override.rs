//! Persistent gravity policy across prediction, rollback and body replacement.

use glam::Vec3;
use prism_physics_core::dynamics::Integrator;
use prism_physics_core::joint::{DistanceJoint, JointKind};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationJoint, SpatialInertia, SpatialTransform,
};
use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::state::storage::BodyStorage;
use prism_physics_core::world::AssemblyWorldError;
use prism_physics_core::{BodyDesc, BodyHandle, BodyKind, PhysicsWorld};

fn close(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).abs().max_element() < 2.0e-4,
        "actual {actual:?}, expected {expected:?}"
    );
}

fn world(gravity: Vec3) -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(gravity);
    world.config.sleep.enabled = false;
    world
}

#[test]
fn storage_policy_is_optional_finite_and_atomic_on_invalid_inputs() {
    let mut world = world(Vec3::NEG_Y);
    world.bodies = BodyStorage::with_capacity(4);
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let stale = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    assert!(world.bodies.remove(stale));
    let replacement = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    assert_eq!(stale.index(), replacement.index());
    assert_ne!(stale.generation(), replacement.generation());
    assert_eq!(world.bodies.gravity_override(body), Some(None));
    // Finite means every finite component, not an arbitrary magnitude cutoff.
    let extreme = Vec3::new(f32::MAX, -f32::MAX, f32::MIN_POSITIVE);
    assert!(world.bodies.set_gravity_override(body, Some(extreme)));
    assert_eq!(world.bodies.gravity_override(body), Some(Some(extreme)));
    {
        let view = world.bodies.solver_view_mut();
        view.sleeping[body.index() as usize] = true;
        view.sleep_timers[body.index() as usize] = 3.0;
    }
    let before = hash_world(&world).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for axis in 0..3 {
            let mut value = Vec3::ZERO;
            value[axis] = bad;
            assert!(!world.bodies.set_gravity_override(body, Some(value)));
            assert_eq!(hash_world(&world).unwrap(), before);
        }
    }
    for invalid in [BodyHandle::INVALID, stale] {
        assert_eq!(world.bodies.gravity_override(invalid), None);
        assert!(!world.bodies.set_gravity_override(invalid, None));
        assert!(!world.bodies.set_gravity_override(invalid, Some(Vec3::X)));
        assert_eq!(hash_world(&world).unwrap(), before);
    }
    assert!(world.bodies.set_gravity_override(body, Some(extreme)));
    assert_eq!(
        hash_world(&world).unwrap(),
        before,
        "same policy is a no-op"
    );
    assert!(world.bodies.set_gravity_override(body, None));
    assert_eq!(world.bodies.gravity_override(body), Some(None));
    assert_eq!(world.bodies.is_sleeping(body), Some(false));
    assert_eq!(world.bodies.sleep_timer(body), Some(0.0));
    assert!(world.bodies.columns_consistent());
}

#[test]
fn changed_policy_wakes_and_recycled_generation_never_inherits_override() {
    let mut world = world(Vec3::ZERO);
    let old = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    {
        let view = world.bodies.solver_view_mut();
        view.sleeping[old.index() as usize] = true;
        view.sleep_timers[old.index() as usize] = 4.0;
    }
    assert!(world.bodies.set_gravity_override(old, Some(Vec3::Y)));
    assert_eq!(world.bodies.is_sleeping(old), Some(false));
    assert_eq!(world.bodies.sleep_timer(old), Some(0.0));
    assert!(world.bodies.remove(old));
    assert_eq!(
        world.bodies.solver_view_mut().gravity_overrides[old.index() as usize],
        None
    );
    let new = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    assert_eq!(old.index(), new.index());
    assert_ne!(old.generation(), new.generation());
    assert_eq!(world.bodies.gravity_override(new), Some(None));
    let before = hash_world(&world).unwrap();
    assert!(!world.bodies.set_gravity_override(old, Some(Vec3::X)));
    assert!(!world.bodies.set_gravity_override(old, None));
    assert_eq!(hash_world(&world).unwrap(), before);
    assert!(world.bodies.columns_consistent());
}

#[test]
fn integrator_consumes_persistent_override_and_clear_resumes_world_gravity() {
    let mut bodies = BodyStorage::new();
    let follow = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
    let pinned = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
    let zero = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
    assert!(bodies.set_gravity_override(pinned, Some(Vec3::X * 4.0)));
    assert!(bodies.set_gravity_override(zero, Some(Vec3::ZERO)));
    for _ in 0..4 {
        Integrator::integrate(&mut bodies, Vec3::NEG_Y * 8.0, 0.125);
    }
    close(bodies.linear_velocity(follow).unwrap(), Vec3::NEG_Y * 4.0);
    close(bodies.linear_velocity(pinned).unwrap(), Vec3::X * 2.0);
    close(bodies.position(pinned).unwrap(), Vec3::X * 0.625);
    close(bodies.position(zero).unwrap(), Vec3::ZERO);
    assert!(bodies.set_gravity_override(pinned, None));
    Integrator::integrate(&mut bodies, Vec3::Z * 8.0, 0.125);
    close(
        bodies.linear_velocity(pinned).unwrap(),
        Vec3::new(2.0, 0.0, 1.0),
    );
    close(
        bodies.linear_velocity(follow).unwrap(),
        Vec3::new(0.0, -4.0, 1.0),
    );
    close(bodies.position(zero).unwrap(), Vec3::ZERO);
}

#[test]
fn both_predictors_ignore_static_and_kinematic_overrides() {
    for xpbd in [false, true] {
        let mut world = world(Vec3::NEG_Y * 10.0);
        let mut kinematic = BodyDesc::static_at(Vec3::Z);
        kinematic.kind = BodyKind::Kinematic;
        let handles = [
            world.spawn(BodyDesc::static_at(Vec3::X)),
            world.spawn(kinematic),
        ];
        for body in handles {
            assert!(world
                .bodies
                .set_gravity_override(body, Some(Vec3::Y * 20.0)));
        }
        if xpbd {
            XpbdSolver::default().step(&mut world, 0.25, 4);
        } else {
            Integrator::integrate(&mut world.bodies, world.config.gravity, 0.25);
        }
        for (body, position) in handles.into_iter().zip([Vec3::X, Vec3::Z]) {
            close(world.bodies.position(body).unwrap(), position);
            close(world.bodies.linear_velocity(body).unwrap(), Vec3::ZERO);
        }
    }
}

#[test]
fn xpbd_applies_override_every_substep_without_force_queue_compensation() {
    for substeps in [1, 4] {
        let mut world = world(Vec3::NEG_Y * 8.0);
        let follow = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let pinned = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let zero = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        assert!(world
            .bodies
            .set_gravity_override(pinned, Some(Vec3::X * 4.0)));
        assert!(world.bodies.set_gravity_override(zero, Some(Vec3::ZERO)));
        assert!(world.external_forces.pending(pinned).is_none());
        let mut solver = XpbdSolver::default();
        solver.step(&mut world, 0.25, substeps);
        close(
            world.bodies.linear_velocity(follow).unwrap(),
            Vec3::NEG_Y * 2.0,
        );
        close(world.bodies.linear_velocity(pinned).unwrap(), Vec3::X);
        world.config.gravity = Vec3::Z * 8.0;
        solver.step(&mut world, 0.25, substeps);
        close(
            world.bodies.linear_velocity(follow).unwrap(),
            Vec3::new(0.0, -2.0, 2.0),
        );
        close(world.bodies.linear_velocity(pinned).unwrap(), Vec3::X * 2.0);
        close(world.bodies.position(zero).unwrap(), Vec3::ZERO);
        assert!(world.external_forces.pending(pinned).is_none());
        assert!(world.bodies.set_gravity_override(pinned, None));
        solver.step(&mut world, 0.25, substeps);
        close(
            world.bodies.linear_velocity(pinned).unwrap(),
            Vec3::new(2.0, 0.0, 2.0),
        );
    }
}

#[test]
fn clone_full_snapshot_restore_hash_and_future_replay_preserve_policy() {
    let mut world = world(Vec3::NEG_Y);
    let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let following_hash = hash_world(&world).unwrap();
    assert!(world
        .bodies
        .set_gravity_override(body, Some(world.config.gravity)));
    assert_ne!(
        hash_world(&world).unwrap(),
        following_hash,
        "policy is simulation state"
    );
    let snapshot = FullStateSnapshot::capture(7, &world);
    let mut fork = snapshot.fork_world();
    assert_eq!(hash_world(&fork).unwrap(), snapshot.state_hash().unwrap());
    assert_eq!(
        world.clone().bodies.gravity_override(body),
        Some(Some(Vec3::NEG_Y))
    );
    assert!(world.bodies.set_gravity_override(body, None));
    snapshot.restore_into(&mut world);
    assert_eq!(hash_world(&world).unwrap(), snapshot.state_hash().unwrap());
    for _ in 0..4 {
        world.config.gravity = Vec3::X;
        fork.config.gravity = Vec3::X;
        XpbdSolver::default().step(&mut world, 0.125, 2);
        XpbdSolver::default().step(&mut fork, 0.125, 2);
        assert_eq!(hash_world(&world).unwrap(), hash_world(&fork).unwrap());
    }
    close(
        world.bodies.linear_velocity(body).unwrap(),
        Vec3::NEG_Y * 0.5,
    );
}

#[test]
fn handoff_fixed_and_floating_pin_source_gravity_even_if_initially_equal() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for source_gravity in [Vec3::ZERO, Vec3::new(2.0, -4.0, 1.0)] {
            for initially_equal in [false, true] {
                let mut world = world(if initially_equal {
                    source_gravity
                } else {
                    Vec3::NEG_Y * 9.0
                });
                let world_gravity = world.config.gravity;
                let ordinary = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
                let inertia = SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::splat(2.0));
                let mut builder =
                    ArticulationBuilder::new(base, inertia).with_gravity(source_gravity);
                builder
                    .add_link(
                        None,
                        SpatialTransform::IDENTITY,
                        inertia,
                        ArticulationJoint::fixed(),
                    )
                    .unwrap();
                let mut tree = builder.build().unwrap();
                tree.break_joint(0);
                let articulation = world.spawn_articulation(tree);
                let report = world.handoff_articulation(articulation).unwrap();
                assert_eq!(world.config.gravity, world_gravity);
                assert_eq!(world.bodies.gravity_override(ordinary), Some(None));
                assert!(!world.articulations.contains(articulation));
                let handoff_bodies = [report.base.body, report.links[0].body];
                for body in handoff_bodies {
                    assert_eq!(
                        world.bodies.gravity_override(body),
                        Some(Some(source_gravity))
                    );
                }
                world.config.gravity = Vec3::Z * 8.0;
                let mut solver = XpbdSolver::default();
                for _ in 0..4 {
                    solver.step(&mut world, 0.0625, 2);
                }
                close(
                    world.bodies.linear_velocity(ordinary).unwrap(),
                    Vec3::Z * 2.0,
                );
                close(
                    world.bodies.linear_velocity(report.links[0].body).unwrap(),
                    source_gravity * 0.25,
                );
                close(
                    world.bodies.linear_velocity(report.base.body).unwrap(),
                    if base == ArticulationBase::Fixed {
                        Vec3::ZERO
                    } else {
                        source_gravity * 0.25
                    },
                );
            }
        }
    }
}

fn assembly_pair() -> (PhysicsWorld, BodyHandle, BodyHandle) {
    let mut world = world(Vec3::NEG_Y * 8.0);
    let a = world.spawn(BodyDesc::dynamic_at(-Vec3::X));
    let b = world.spawn(BodyDesc::dynamic_at(Vec3::X));
    world
        .connect_assembly(
            a,
            b,
            None,
            JointKind::Distance(DistanceJoint::rigid(2.0)),
            None,
        )
        .unwrap();
    (world, a, b)
}

#[test]
fn assembly_rejects_distinct_policies_atomically_including_world_following() {
    for policy_b in [None, Some(Vec3::X * 3.0)] {
        let (mut world, a, b) = assembly_pair();
        assert!(world
            .bodies
            .set_gravity_override(a, Some(world.config.gravity)));
        assert!(world.bodies.set_gravity_override(b, policy_b));
        assert!(world.apply_force(a, Vec3::Z, -Vec3::X));
        let before = hash_world(&world).unwrap();
        assert_eq!(
            world.merge_assembly(&[a, b]),
            Err(AssemblyWorldError::GravityPolicyConflict)
        );
        assert_eq!(hash_world(&world).unwrap(), before);
        assert!(world.bodies.contains(a) && world.bodies.contains(b));
    }
}

#[test]
fn assembly_proxy_and_split_keep_live_uniform_policy_and_sleep_state() {
    for policy in [None, Some(Vec3::X * 4.0)] {
        for replacement in [None, Some(Vec3::Z * 2.0)] {
            let (mut world, a, b) = assembly_pair();
            for body in [a, b] {
                assert!(world.bodies.set_gravity_override(body, policy));
            }
            let proxy = world.merge_assembly(&[a, b]).unwrap();
            assert_eq!(world.bodies.gravity_override(proxy), Some(policy));
            assert!(world.bodies.set_gravity_override(proxy, replacement));
            {
                let view = world.bodies.solver_view_mut();
                view.sleeping[proxy.index() as usize] = true;
                view.sleep_timers[proxy.index() as usize] = 2.5;
            }
            let snapshot = FullStateSnapshot::capture(0, &world);
            let mut fork = snapshot.fork_world();
            let children = world.split_assembly(proxy).unwrap();
            assert_eq!(fork.split_assembly(proxy).unwrap(), children);
            assert_eq!(hash_world(&world).unwrap(), hash_world(&fork).unwrap());
            assert!(!world.bodies.contains(proxy));
            for &child in &children {
                assert_eq!(world.bodies.gravity_override(child), Some(replacement));
                assert_eq!(world.bodies.is_sleeping(child), Some(true));
                assert_eq!(world.bodies.sleep_timer(child), Some(2.5));
                assert!(world.bodies.wake(child));
            }
            XpbdSolver::default().step(&mut world, 0.125, 2);
            for child in children {
                close(
                    world.bodies.linear_velocity(child).unwrap(),
                    replacement.unwrap_or(world.config.gravity) * 0.125,
                );
            }
        }
    }
}

#[test]
fn aero_added_mass_uses_the_same_gravity_as_prediction() {
    use prism_physics_core::aero::{AddedMass, AeroModel, FluidKind, FluidPlane};
    use prism_physics_core::AeroBodyConfig;
    for xpbd in [false, true] {
        for policy in [None, Some(Vec3::ZERO), Some(Vec3::X * 4.0)] {
            let mut world = world(Vec3::NEG_Y * 8.0);
            let body = world.spawn(BodyDesc::dynamic_at(Vec3::NEG_Y));
            assert!(world.bodies.set_gravity_override(body, policy));
            let model = AeroModel {
                water_density: 1.0,
                added_mass: Some(AddedMass {
                    fluid: FluidKind::Water,
                    volume: 1.0,
                    linear_coefficients: Vec3::ONE,
                    ..AddedMass::default()
                }),
                water_plane: Some(FluidPlane::horizontal(0.0, 1.0)),
                ..AeroModel::default()
            };
            world
                .register_aero_body(body, AeroBodyConfig::new(model))
                .unwrap();
            if xpbd {
                XpbdSolver::default().step(&mut world, 0.125, 2);
            } else {
                world.step_external_dynamics(0.125);
            }
            // Unit body + unit added fluid mass halves the gravity response.
            close(
                world.bodies.linear_velocity(body).unwrap(),
                policy.unwrap_or(world.config.gravity) * 0.0625,
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn island_and_colored_scratch_preserve_overrides_and_match_serial_contacts() {
    use prism_physics_core::{ColliderShape, XpbdConfig};
    let mut source = world(Vec3::NEG_Y * 8.0);
    let shape = source.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut handles = Vec::new();
    for island in 0..2 {
        for offset in [0.0, 0.9] {
            let body = source.spawn(
                BodyDesc::dynamic_at(Vec3::X * (island as f32 * 5.0 + offset)).with_collider(shape),
            );
            assert!(source
                .bodies
                .set_gravity_override(body, Some(Vec3::Y * (island as f32 + 1.0))));
            handles.push(body);
        }
    }
    let mut serial = source.clone();
    let mut serial_solver = XpbdSolver::with_config(XpbdConfig {
        parallel_islands: false,
        ..XpbdConfig::default()
    });
    serial_solver.step(&mut serial, 0.125, 2);
    assert!(
        (serial.bodies.position(handles[0]).unwrap().x
            - serial.bodies.position(handles[1]).unwrap().x)
            .abs()
            > 0.99,
        "overlapping pairs must actually exercise contact scratch"
    );
    for colored in [false, true] {
        let mut parallel = source.clone();
        XpbdSolver::with_config(XpbdConfig {
            parallel_islands: true,
            parallel_within_island: colored,
            ..XpbdConfig::default()
        })
        .step(&mut parallel, 0.125, 2);
        for &body in &handles {
            assert_eq!(
                parallel.bodies.gravity_override(body),
                source.bodies.gravity_override(body)
            );
            close(
                parallel.bodies.position(body).unwrap(),
                serial.bodies.position(body).unwrap(),
            );
            close(
                parallel.bodies.linear_velocity(body).unwrap(),
                serial.bodies.linear_velocity(body).unwrap(),
            );
        }
    }
}
