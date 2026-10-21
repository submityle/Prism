use super::*;
use crate::aero::{AeroBodyConfig, WindField};
use crate::collide::ContactManifold;
use crate::collider::ColliderShape;
use crate::dynamics::assembly::{BreakLimit, JointLoad};
use crate::dynamics::field::{Field, FieldId};
use crate::joint::{DistanceJoint, JointDesc};
use crate::reduced::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationJoint, LinkContactImpulse,
    SpatialInertia, SpatialTransform,
};
use crate::snapshot::hash::hash_state;
use crate::snapshot::{FullStateSnapshot, StateSnapshot};
use crate::soft::{ClothGrid, ParticleHandle, SoftBody};
use crate::solver::{Solver, XpbdSolver};
use crate::state::body::BodyDesc;
use glam::Vec3;

fn articulation() -> Articulation {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
    builder
        .add_link(
            None,
            SpatialTransform::IDENTITY,
            SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE),
            ArticulationJoint::revolute(Vec3::Z),
        )
        .unwrap();
    builder.build().unwrap()
}

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let a = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let b = world.spawn(BodyDesc::dynamic_at(Vec3::X));
    let joint = world.spawn_joint(JointDesc::distance(
        a,
        Vec3::ZERO,
        b,
        Vec3::ZERO,
        DistanceJoint::rigid(1.0),
    ));
    world
        .breakable_joints
        .register(joint, BreakLimit::new(100.0, 100.0).with_fatigue(0.01))
        .unwrap();
    world.spawn_soft_body(
        ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default()
        .body,
    );
    world.spawn_articulation(articulation());
    world.fields.insert(Field::constant(FieldId(0), Vec3::X));
    world
}

fn changed_without_render_change(world: &PhysicsWorld, change: impl FnOnce(&mut PhysicsWorld)) {
    let mut changed = world.clone();
    change(&mut changed);
    assert_eq!(
        hash_state(&StateSnapshot::capture(world)),
        hash_state(&StateSnapshot::capture(&changed))
    );
    assert_ne!(hash_world(world).unwrap(), hash_world(&changed).unwrap());
}

#[test]
fn includes_non_render_rigid_state_shapes_configuration_and_breakage() {
    let world = world();
    let body = world.bodies.handle_at_slot(0).unwrap();
    let joint = world.joints.active_entries().next().unwrap().0;
    changed_without_render_change(&world, |w| {
        w.bodies.set_prev_position(body, Vec3::Y);
    });
    changed_without_render_change(&world, |w| {
        w.bodies.set_ccd(body, true);
    });
    changed_without_render_change(&world, |w| {
        w.bodies.set_sensor(body, true);
    });
    changed_without_render_change(&world, |w| {
        w.bodies.solver_view_mut().sleep_timers[body.index() as usize] = 0.3;
    });
    changed_without_render_change(&world, |w| {
        w.config.gravity = Vec3::Y;
    });
    changed_without_render_change(&world, |w| {
        w.shapes.insert(ColliderShape::Sphere { radius: 2.0 });
    });
    changed_without_render_change(&world, |w| {
        w.joints.get_mut(joint).unwrap().enabled = false;
    });
    changed_without_render_change(&world, |w| {
        assert!(w.breakable_joints.record_load(
            joint,
            JointLoad {
                linear_impulse: Vec3::X,
                angular_impulse: Vec3::ZERO,
                substep_dt: 0.1,
            }
        ));
    });
    changed_without_render_change(&world, |w| {
        w.assemblies.detach_component(&[body]);
    });
}

#[test]
fn includes_soft_history_constraints_topology_and_allocator_history() {
    let world = world();
    let soft = world.soft_bodies.handles().next().unwrap();
    changed_without_render_change(&world, |w| {
        w.soft_bodies
            .get_mut(soft)
            .unwrap()
            .particles
            .set_velocity(ParticleHandle::from_index(0), Vec3::Y);
    });
    changed_without_render_change(&world, |w| {
        w.soft_bodies.get_mut(soft).unwrap().config.iterations += 1;
    });
    changed_without_render_change(&world, |w| {
        let body = w.soft_bodies.get_mut(soft).unwrap();
        let events = body
            .puncture_topology(Vec3::new(0.05, 0.0, 0.05), Vec3::Y, 1.0)
            .unwrap();
        assert!(!events.is_empty());
        assert!(
            body.particles.len() > 9,
            "puncture must really split particles"
        );
    });
    changed_without_render_change(&world, |w| {
        let transient = w.spawn_soft_body(SoftBody::with_default_config());
        assert!(w.remove_soft_body(transient));
    });

    // Identical live and render state, but free-list order determines which
    // handle the next insertion receives and must therefore affect the hash.
    let mut a = PhysicsWorld::default();
    let first = a.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let second = a.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let mut b = a.clone();
    assert!(a.remove_body(first));
    assert!(a.remove_body(second));
    assert!(b.remove_body(second));
    assert!(b.remove_body(first));
    assert_eq!(
        hash_state(&StateSnapshot::capture(&a)),
        hash_state(&StateSnapshot::capture(&b))
    );
    assert_ne!(hash_world(&a).unwrap(), hash_world(&b).unwrap());
}

#[test]
fn includes_field_id_allocator_pending_force_articulation_queues_and_aero() {
    let world = world();
    let body = world.bodies.handle_at_slot(0).unwrap();
    let art = world.articulations.handles().next().unwrap();
    changed_without_render_change(&world, |w| {
        let id = w.fields.insert(Field::constant(FieldId(0), Vec3::Y));
        w.fields.remove(id).unwrap();
        assert_eq!(w.fields.fields(), world.fields.fields());
    });
    changed_without_render_change(&world, |w| {
        w.external_forces.accumulate_force(body, Vec3::Z).unwrap();
    });
    changed_without_render_change(&world, |w| {
        assert!(w.articulations.get_mut(art).unwrap().set_position(0, 0.1));
    });
    changed_without_render_change(&world, |w| {
        assert!(w.articulations.get_mut(art).unwrap().set_velocity(0, 0.1));
    });
    changed_without_render_change(&world, |w| {
        assert!(w.queue_articulation_contact_impulse(
            art,
            LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y
            }
        ));
    });
    changed_without_render_change(&world, |w| {
        assert!(w.request_articulation_break(art, 0));
    });
    changed_without_render_change(&world, |w| {
        let h = w.spawn_articulation(articulation());
        assert!(w.remove_articulation(h));
    });
    changed_without_render_change(&world, |w| {
        w.aero_time = 0.5;
    });
    changed_without_render_change(&world, |w| {
        w.set_aero_field(WindField::new(Vec3::Y));
    });
    changed_without_render_change(&world, |w| {
        w.register_aero_body(body, AeroBodyConfig::default())
            .unwrap();
    });
}

#[test]
fn contact_hash_is_independent_of_hash_seeds_and_insertion_order_but_not_history() {
    let mut a = PhysicsWorld::default();
    let handles: Vec<_> = (0..9)
        .map(|i| a.spawn(BodyDesc::dynamic_at(Vec3::X * i as f32)))
        .collect();
    let mut b = a.clone();
    let contacts: Vec<_> = handles
        .windows(2)
        .map(|p| ContactManifold::new(p[0], p[1], Vec3::X))
        .collect();
    let _ = a.contact_events.record(&contacts, |h| h.index() % 3 == 0);
    let reversed: Vec<_> = contacts.iter().rev().copied().collect();
    let _ = b.contact_events.record(&reversed, |h| h.index() % 3 == 0);
    assert!(a.contact_events.active_collision_count() > 1);
    assert!(a.contact_events.active_trigger_count() > 1);
    let expected = hash_world(&a).unwrap();
    for _ in 0..16 {
        // record constructs freshly seeded maps each time.
        let _ = b.contact_events.record(&reversed, |h| h.index() % 3 == 0);
        assert_eq!(hash_world(&b).unwrap(), expected);
    }
    b.contact_events.reset();
    assert_ne!(hash_world(&b).unwrap(), expected);
}

#[test]
fn snapshot_hash_survives_restore_resimulation_and_merged_assembly_split() {
    let mut world = world();
    let bodies: Vec<_> = (0..world.bodies.slot_count())
        .filter_map(|slot| world.bodies.handle_at_slot(slot))
        .collect();
    let proxy = world.merge_assembly(&bodies).unwrap();
    let before = FullStateSnapshot::capture(17, &world);
    assert_eq!(before.state_hash().unwrap(), hash_world(&world).unwrap());
    let mut restored = PhysicsWorld::default();
    before.restore_into(&mut restored);
    assert_eq!(hash_world(&restored).unwrap(), before.state_hash().unwrap());
    // This exercises private merge-recovery state, not merely proxy poses.
    world.split_assembly(proxy).unwrap();
    restored.split_assembly(proxy).unwrap();
    assert_eq!(hash_world(&restored).unwrap(), hash_world(&world).unwrap());
    XpbdSolver::new().step(&mut world, 1.0 / 60.0, 2);
    XpbdSolver::new().step(&mut restored, 1.0 / 60.0, 2);
    assert_eq!(hash_world(&restored).unwrap(), hash_world(&world).unwrap());
    assert_ne!(hash_world(&restored).unwrap(), before.state_hash().unwrap());
}

#[test]
fn float_fingerprint_retains_signed_zero_and_rejects_nan_in_hidden_state() {
    let mut a = world();
    let mut b = a.clone();
    a.aero_time = 0.0;
    b.aero_time = -0.0;
    assert_ne!(hash_world(&a).unwrap(), hash_world(&b).unwrap());
    let body = b.bodies.handle_at_slot(0).unwrap();
    b.bodies
        .set_prev_position(body, Vec3::new(f32::from_bits(0x7fc0_0001), 0.0, 0.0));
    assert_eq!(hash_world(&b), Err(WorldHashError::Nan));
    assert_eq!(
        FullStateSnapshot::capture(0, &b).state_hash(),
        Err(WorldHashError::Nan)
    );
}

#[test]
fn canonicalizer_sorts_only_unordered_containers_and_handles_escaping() {
    let canonical = |input: &str| {
        CanonicalDebug {
            input: input.as_bytes(),
            cursor: 0,
        }
        .group(None, 0)
    };
    assert_eq!(
        canonical("X { b: {3, 1}, a: [2, 1] }"),
        canonical("X{a:[2,1],b:{1,3}}")
    );
    assert_ne!(canonical("[1,2]"), canonical("[2,1]"));
    assert_eq!(
        canonical(r#"{"a,}\"b": [1,2]}"#).unwrap(),
        br#"{"a,}\"b":[1,2]}"#
    );
    assert_eq!(
        canonical("X{a:[1,2}"),
        Err(WorldHashError::InvalidRepresentation)
    );
    assert_eq!(canonical("[NaN]"), Err(WorldHashError::Nan));
}
