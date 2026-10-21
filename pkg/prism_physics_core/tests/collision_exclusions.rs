//! World-owned group policy contracts, exercised through public collision and
//! lifecycle APIs. Handoff installs one group for the entire original tree;
//! membership must not depend on the remaining joint graph.

use glam::Vec3;
use prism_physics_core::ccd::resolve_ccd;
use prism_physics_core::joint::{DistanceJoint, JointKind};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationColliderDesc, ArticulationJoint,
    SpatialInertia, SpatialTransform,
};
use prism_physics_core::snapshot::{FullStateSnapshot, StateSnapshot};
use prism_physics_core::snapshot::hash::{hash_state, hash_world};
use prism_physics_core::world::AssemblyWorldError;
use prism_physics_core::{
    detect_contacts, BodyDesc, BodyHandle, ColliderShape, CollisionExclusionError, Isometry,
    PhysicsMaterial, PhysicsWorld, Solver, XpbdSolver,
};

const DT: f32 = 1.0 / 120.0;
const FLAGS: [[bool; 2]; 4] = [[false, false], [false, true], [true, false], [true, true]];

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    world
}

fn sphere(world: &mut PhysicsWorld, position: Vec3) -> BodyHandle {
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    world.spawn(
        BodyDesc::dynamic_at(position)
            .with_collider(shape)
            .with_material(PhysicsMaterial::FRICTIONLESS),
    )
}

fn has_contact(world: &PhysicsWorld, a: BodyHandle, b: BodyHandle) -> bool {
    detect_contacts(world).iter().any(|contact| {
        (contact.body_a == a && contact.body_b == b)
            || (contact.body_a == b && contact.body_b == a)
    })
}

#[test]
fn finite_contacts_require_both_flags_but_preserve_other_groups_and_outsiders() {
    for flags in FLAGS {
        let mut world = world();
        let a = sphere(&mut world, Vec3::ZERO);
        let b = sphere(&mut world, Vec3::X * 0.8);
        let outsider = sphere(&mut world, Vec3::Y * 0.6);
        let other_group = sphere(&mut world, -Vec3::Y * 0.6);
        assert!(world.collision_pair_allowed(a, b));
        assert!(!world.collision_pair_allowed(a, a));
        assert!(has_contact(&world, a, b));
        world.install_collision_exclusion_group(&[(a, flags[0]), (b, flags[1])]).unwrap();
        world.install_collision_exclusion_group(&[(other_group, false)]).unwrap();
        assert_eq!(has_contact(&world, a, b), flags[0] && flags[1]);
        assert_eq!(world.collision_pair_allowed(b, a), flags[0] && flags[1]);
        assert!(has_contact(&world, a, outsider));
        assert!(has_contact(&world, a, other_group));
        assert!(world.collision_pair_allowed(other_group, outsider));
    }
}

#[test]
fn plane_contacts_use_the_same_group_gate() {
    for flags in FLAGS {
        let mut world = world();
        let plane_shape = world.shapes.insert(ColliderShape::Plane { normal: Vec3::Y, offset: 0.0 });
        let plane = world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(plane_shape));
        let a = sphere(&mut world, Vec3::Y * 0.4);
        let outsider = sphere(&mut world, Vec3::new(3.0, 0.4, 0.0));
        world.install_collision_exclusion_group(&[(a, flags[0]), (plane, flags[1])]).unwrap();
        assert_eq!(has_contact(&world, a, plane), flags[0] && flags[1]);
        assert!(has_contact(&world, outsider, plane));
    }
}

#[test]
fn disconnected_members_keep_one_linear_group_until_each_body_is_removed() {
    let mut world = world();
    let members: Vec<_> = (0..256)
        .map(|i| (world.spawn(BodyDesc::dynamic_at(Vec3::X * i as f32)), false))
        .collect();
    let group = world.install_collision_exclusion_group(&members).unwrap();
    // No assembly edges exist: all original branches are already disconnected.
    assert_eq!(world.assemblies.connected_components().len(), members.len());
    assert_eq!(world.collision_exclusions().len(), members.len());
    assert!(!world.collision_pair_allowed(members[0].0, members[255].0));
    assert!(world.remove_body(members[128].0));
    assert_eq!(world.collision_exclusions().len(), members.len() - 1);
    assert_eq!(world.collision_exclusions().membership(members[255].0).unwrap().group, group);
    assert!(!world.collision_pair_allowed(members[0].0, members[255].0));
    for &(body, _) in &members {
        let _ = world.remove_body(body);
    }
    assert!(world.collision_exclusions().is_empty());
}

#[test]
fn removal_and_generation_reuse_do_not_inherit_or_erase_membership() {
    let mut world = world();
    let a = sphere(&mut world, Vec3::ZERO);
    let old = sphere(&mut world, Vec3::X * 0.8);
    world.install_collision_exclusion_group(&[(a, false), (old, false)]).unwrap();
    assert!(world.remove_body(old));
    assert_eq!(world.collision_exclusions().len(), 1);
    assert!(world.collision_exclusions().membership(old).is_none());
    let new = sphere(&mut world, Vec3::X * 0.8);
    assert_eq!(new.index(), old.index());
    assert_ne!(new.generation(), old.generation());
    assert!(has_contact(&world, a, new));
    world.install_collision_exclusion_group(&[(new, false)]).unwrap();
    let before = hash_world(&world).unwrap();
    assert!(!world.remove_body(old));
    assert_eq!(hash_world(&world).unwrap(), before);
    assert!(world.collision_exclusions().membership(new).is_some());
    assert!(world.collision_pair_allowed(a, new));
}

#[test]
fn low_level_storage_removal_is_generation_safe_and_reclaimed_at_substep() {
    let mut world = world();
    let a = sphere(&mut world, Vec3::ZERO);
    let old = sphere(&mut world, Vec3::X * 0.8);
    world.install_collision_exclusion_group(&[(a, false), (old, false)]).unwrap();
    assert!(world.bodies.remove(old));
    let new = world.bodies.insert(BodyDesc::dynamic_at(Vec3::X * 0.8));
    assert_eq!(new.index(), old.index());
    assert!(world.collision_exclusions().membership(new).is_none());
    assert!(world.collision_pair_allowed(a, new));
    world.prepare_external_substep(DT);
    assert_eq!(world.collision_exclusions().len(), 1);
    world.install_collision_exclusion_group(&[(new, false)]).unwrap();
    assert!(world.collision_pair_allowed(a, new));
}

#[test]
fn invalid_group_requests_leave_members_and_allocator_unchanged() {
    let mut world = world();
    let a = sphere(&mut world, Vec3::ZERO);
    let b = sphere(&mut world, Vec3::X);
    let c = sphere(&mut world, Vec3::Y);
    world.install_collision_exclusion_group(&[(a, false)]).unwrap();
    let before = hash_world(&world).unwrap();
    for (request, error) in [
        (vec![], CollisionExclusionError::EmptyGroup),
        (vec![(b, true), (BodyHandle::INVALID, false)], CollisionExclusionError::InvalidBody(BodyHandle::INVALID)),
        (vec![(b, true), (b, false)], CollisionExclusionError::DuplicateBody(b)),
        (vec![(b, true), (a, true)], CollisionExclusionError::AlreadyGrouped(a)),
    ] {
        assert_eq!(world.install_collision_exclusion_group(&request), Err(error));
        assert_eq!(hash_world(&world).unwrap(), before);
    }
    assert!(world.collision_exclusions().membership(b).is_none());
    let mut control = world.clone();
    let actual = world.install_collision_exclusion_group(&[(b, true), (c, false)]).unwrap();
    let expected = control.install_collision_exclusion_group(&[(b, true), (c, false)]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(hash_world(&world).unwrap(), hash_world(&control).unwrap());
}

#[test]
fn clone_full_snapshot_restore_and_hash_include_flags_and_group_identity() {
    let mut base = world();
    let a = sphere(&mut base, Vec3::ZERO);
    let b = sphere(&mut base, Vec3::X * 0.8);
    let mut filtered = base.clone();
    filtered.install_collision_exclusion_group(&[(a, true), (b, false)]).unwrap();
    let mut both_true = base.clone();
    both_true.install_collision_exclusion_group(&[(a, true), (b, true)]).unwrap();
    let mut separate = base.clone();
    separate.install_collision_exclusion_group(&[(a, true)]).unwrap();
    separate.install_collision_exclusion_group(&[(b, false)]).unwrap();
    assert_eq!(hash_state(&StateSnapshot::capture(&base)), hash_state(&StateSnapshot::capture(&filtered)));
    for other in [&base, &both_true, &separate] {
        assert_ne!(hash_world(&filtered).unwrap(), hash_world(other).unwrap());
    }
    let snapshot = FullStateSnapshot::capture(7, &filtered);
    let clone = filtered.clone();
    assert!(!has_contact(&clone, a, b));
    assert_eq!(snapshot.state_hash().unwrap(), hash_world(&clone).unwrap());
    assert!(filtered.remove_body(b));
    snapshot.restore_into(&mut filtered);
    assert!(!has_contact(&filtered, a, b));
    assert_eq!(hash_world(&filtered).unwrap(), snapshot.state_hash().unwrap());
    let mut fork = snapshot.fork_world();
    let c = sphere(&mut filtered, Vec3::Y * 4.0);
    let fork_c = sphere(&mut fork, Vec3::Y * 4.0);
    assert_eq!(c, fork_c);
    assert_eq!(filtered.install_collision_exclusion_group(&[(c, false)]).unwrap(), fork.install_collision_exclusion_group(&[(fork_c, false)]).unwrap());
    assert_eq!(hash_world(&filtered).unwrap(), hash_world(&fork).unwrap());
    XpbdSolver::new().step(&mut filtered, DT, 1);
    XpbdSolver::new().step(&mut fork, DT, 1);
    assert_eq!(hash_world(&filtered).unwrap(), hash_world(&fork).unwrap());
}

fn wall(world: &mut PhysicsWorld, x: f32) -> BodyHandle {
    let shape = world.shapes.insert(ColliderShape::Cuboid { half_extents: Vec3::new(0.05, 3.0, 3.0) });
    world.spawn(BodyDesc::static_at(Vec3::X * x).with_collider(shape))
}

#[test]
fn ccd_skips_excluded_wall_but_still_hits_external_wall() {
    for flags in FLAGS {
        for external in [false, true] {
            let mut world = world();
            let target = wall(&mut world, 0.0);
            if external { wall(&mut world, 1.5); }
            let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.1 });
            let mover = world.spawn(BodyDesc::dynamic_at(-Vec3::X * 2.0).with_collider(shape).with_ccd(true));
            world.install_collision_exclusion_group(&[(mover, flags[0]), (target, flags[1])]).unwrap();
            assert!(world.bodies.set_position(mover, Vec3::X * 2.0));
            resolve_ccd(&mut world, DT);
            let x = world.bodies.position(mover).unwrap().x;
            if flags[0] && flags[1] {
                assert!((-0.2..-0.1).contains(&x), "first wall: {x}");
            } else if external {
                assert!((1.3..1.4).contains(&x), "external wall: {x}");
            } else {
                assert_eq!(x, 2.0);
            }
        }
    }
}

#[test]
fn ccd_plane_candidates_obey_flags_without_suppressing_external_planes() {
    for flags in FLAGS {
        let mut world = world();
        let shape = world.shapes.insert(ColliderShape::Plane { normal: Vec3::Y, offset: 0.0 });
        let plane = world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(shape));
        world.spawn(BodyDesc::static_at(-Vec3::Y).with_collider(shape));
        let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.1 });
        let mover = world.spawn(BodyDesc::dynamic_at(Vec3::Y * 2.0).with_collider(shape).with_ccd(true));
        world.install_collision_exclusion_group(&[(mover, flags[0]), (plane, flags[1])]).unwrap();
        world.bodies.set_position(mover, -Vec3::Y * 2.0);
        resolve_ccd(&mut world, DT);
        let y = world.bodies.position(mover).unwrap().y;
        let expected = if flags[0] && flags[1] { 0.1 } else { -0.9 };
        assert!((y - expected).abs() < 0.01, "flags {flags:?}: {y}");
    }
}

fn contacting_tree(world: &mut PhysicsWorld) {
    let inertia = SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.1));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, inertia)
        .with_base_pose(Isometry::from_translation(-Vec3::X * 0.8))
        .with_gravity(Vec3::ZERO);
    builder.add_link(None, SpatialTransform::IDENTITY, inertia, ArticulationJoint::fixed()).unwrap();
    let owner = world.spawn_articulation(builder.build().unwrap());
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let mut desc = ArticulationColliderDesc::new(owner, Some(0), shape);
    desc.material = PhysicsMaterial::FRICTIONLESS;
    world.attach_articulation_collider(desc).unwrap();
}

#[test]
fn mixed_pass_does_not_reintroduce_excluded_rigid_pairs_in_tree_coupled_islands() {
    for flags in FLAGS {
        let mut world = world();
        let a = sphere(&mut world, Vec3::ZERO);
        let b = sphere(&mut world, Vec3::X * 0.8);
        world.install_collision_exclusion_group(&[(a, flags[0]), (b, flags[1])]).unwrap();
        contacting_tree(&mut world);
        XpbdSolver::new().step(&mut world, DT, 1);
        let b_x = world.bodies.position(b).unwrap().x;
        if flags[0] && flags[1] {
            assert!(b_x > 0.81, "opted-in pair did not respond: {b_x}");
        } else {
            assert_eq!(b_x, 0.8, "mixed pass reintroduced excluded pair: {flags:?}");
            assert!(world.bodies.position(a).unwrap().x > 0.01, "external tree contact was lost");
        }
    }
}

#[test]
fn mixed_pass_preserves_rigid_contacts_between_different_false_groups() {
    let mut world = world();
    let a = sphere(&mut world, Vec3::ZERO);
    let b = sphere(&mut world, Vec3::X * 0.8);
    world.install_collision_exclusion_group(&[(a, false)]).unwrap();
    world.install_collision_exclusion_group(&[(b, false)]).unwrap();
    contacting_tree(&mut world);
    XpbdSolver::new().step(&mut world, DT, 1);
    assert!(world.bodies.position(b).unwrap().x > 0.81);
}

fn assembly_pair(world: &mut PhysicsWorld) -> [BodyHandle; 2] {
    let a = sphere(world, -Vec3::X * 2.0);
    let b = sphere(world, Vec3::X * 2.0);
    world.connect_assembly(a, b, None, JointKind::Distance(DistanceJoint::rigid(4.0)), None).unwrap();
    [a, b]
}

#[test]
fn merging_registered_members_fails_before_mutation_instead_of_losing_policy() {
    let mut world = world();
    let [a, b] = assembly_pair(&mut world);
    let branch = sphere(&mut world, Vec3::Y * 5.0);
    world.install_collision_exclusion_group(&[(a, true), (b, false), (branch, false)]).unwrap();
    let before = hash_world(&world).unwrap();
    assert_eq!(world.merge_assembly(&[a, b]), Err(AssemblyWorldError::CollisionExclusionConflict));
    assert_eq!(hash_world(&world).unwrap(), before);
    assert!(!world.collision_pair_allowed(a, branch));
    assert!(!world.collision_pair_allowed(b, branch));
}

#[test]
fn proxy_registration_is_rejected_atomically_and_unrelated_policy_survives_split() {
    let mut world = world();
    let pair = assembly_pair(&mut world);
    let a = sphere(&mut world, Vec3::Y * 5.0);
    let b = sphere(&mut world, Vec3::Y * 5.8);
    let group = world.install_collision_exclusion_group(&[(a, false), (b, true)]).unwrap();
    let proxy = world.merge_assembly(&pair).unwrap();
    let before = hash_world(&world).unwrap();
    assert_eq!(world.install_collision_exclusion_group(&[(proxy, false)]), Err(CollisionExclusionError::MergedBody(proxy)));
    assert_eq!(hash_world(&world).unwrap(), before);
    let restored = world.split_assembly(proxy).unwrap();
    assert_eq!(restored.len(), 2);
    assert_eq!(world.collision_exclusions().len(), 2);
    assert_eq!(world.collision_exclusions().membership(a).unwrap().group, group);
    assert!(!world.collision_pair_allowed(a, b));
    for body in restored {
        assert!(world.collision_exclusions().membership(body).is_none());
        assert!(world.collision_pair_allowed(body, a));
    }
}
