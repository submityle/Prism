//! Multiple reduced attachments share one adopted ECS body, not one entity per leaf.
use std::time::Duration;

use glam::{Mat3, Quat, Vec3};
use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_core::collider::compound::CompoundLeafPolicy;
use prism_physics_core::collider::{CompoundChild, PhysicsMaterial};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::query::QueryFilter;
use prism_physics_core::reduced::articulation::handoff::{
    ArticulationHandoffError, UnsupportedJointPolicy,
};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationColliderDesc, ArticulationColliderId, ArticulationHandle,
    ArticulationJoint, ArticulationLink, SpatialInertia, SpatialTransform,
};
use prism_physics_core::{BodyDesc, ColliderHandle, ColliderShape};
use prism_physics_ecs::articulation::{
    ArticulationBinding, ArticulationBindings, ArticulationDiagnostics, ArticulationHandoff,
    ArticulationIssue, ArticulationSnapshot, PrismArticulationPlugin,
};
use prism_physics_ecs::{
    Collider, CollisionShape, PhysicsBindings, PhysicsWorldResource, PrismPhysicsPlugin, RigidBody,
};

struct Fixture {
    app: App,
    owner: Entity,
    source: ArticulationHandle,
    attachments: Vec<(ArticulationColliderId, ArticulationColliderDesc)>,
    primitive: ColliderHandle,
    nested: ColliderHandle,
    children: [CompoundChild; 2],
    source_poses: Vec<Isometry>,
    config: ArticulationBinding,
}

fn fixture(base: ArticulationBase, unsupported: bool) -> Fixture {
    let mut app = App::new();
    let step = Duration::from_millis(10);
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(step))
        .set_fixed_timestep(step)
        .add_plugins((PrismPhysicsPlugin, PrismArticulationPlugin));
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.config.gravity = Vec3::ZERO;
        physics.config.default_substeps = 1;
    }
    let axes = Mat3::from_quat(Quat::from_rotation_y(0.4));
    let tensor = axes * Mat3::from_diagonal(Vec3::new(0.8, 1.3, 1.9)) * axes.transpose();
    let inertia = SpatialInertia::new(
        2.0,
        Vec3::new(0.2, -0.1, 0.3),
        tensor.transpose().to_cols_array_2d(),
    );
    let mut config = ArticulationBinding::new(base, inertia);
    config.base_pose = Isometry::new(Vec3::new(2.0, 3.0, -1.0), Quat::from_rotation_x(0.3));
    for index in 0..2_usize {
        config.links.push(ArticulationLink {
            parent: index.checked_sub(1),
            x_tree: SpatialTransform::new(Quat::from_rotation_z(0.2), Vec3::Y * 4.0),
            inertia,
            joint: ArticulationJoint::revolute(Vec3::Z),
        });
    }
    if unsupported {
        // Link 0 survives the break at link 1: its extra policy rejects handoff.
        config.links[0].joint.damping = 0.5;
    }
    let owner = app.world_mut().spawn((config.clone(),));
    app.update();
    let source = app
        .world()
        .resource::<ArticulationBindings>()
        .articulation(owner)
        .unwrap();
    let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
    let primitive = physics.shapes.insert(ColliderShape::Cuboid {
        half_extents: Vec3::new(0.04, 0.06, 0.08),
    });
    let children = [
        CompoundChild {
            collider: primitive,
            local_pose: Isometry::new(Vec3::X * 0.4, Quat::from_rotation_y(0.2)),
        },
        CompoundChild {
            collider: primitive,
            local_pose: Isometry::new(-Vec3::X * 0.4, Quat::from_rotation_x(-0.3)),
        },
    ];
    let nested = physics
        .shapes
        .insert_compound_with_policies(
            &children,
            &[CompoundLeafPolicy {
                // The reduced attachment, not this deliberately conflicting policy,
                // must be authoritative for every descendant after conversion.
                material: Some(PhysicsMaterial::new(9.0, 1.0)),
                self_collision: Some(true),
            }; 2],
        )
        .unwrap();
    let mut attachments = Vec::new();
    for link in [None, Some(0), Some(1)] {
        for index in 0..2 {
            let mut desc = ArticulationColliderDesc::new(source, link, nested);
            desc.local_pose = Isometry::new(
                Vec3::Z * (index as f32 * 2.0),
                Quat::from_rotation_y(0.2 + index as f32 * 0.3),
            );
            desc.material = PhysicsMaterial::new(0.2 + index as f32, index as f32 * 0.4);
            desc.self_collision = index == 1;
            let id = physics.attach_articulation_collider(desc.clone()).unwrap();
            attachments.push((id, desc));
        }
    }
    let source_poses = physics
        .articulations
        .get(source)
        .unwrap()
        .maximal_body_states()
        .into_iter()
        .map(|state| state.pose)
        .collect();
    Fixture {
        app,
        owner,
        source,
        attachments,
        primitive,
        nested,
        children,
        source_poses,
        config,
    }
}

fn degrade(f: &mut Fixture) {
    assert!(f
        .app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .articulations
        .get_mut(f.source)
        .unwrap()
        .break_joint(1)
        .is_some());
    f.app.update();
}

fn view(f: &Fixture) -> ArticulationHandoff {
    f.app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(f.owner)
        .unwrap()
        .clone()
}

fn close_pose(actual: Isometry, expected: Isometry) {
    for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        assert!(
            actual
                .transform_point(point)
                .distance(expected.transform_point(point))
                < 2.0e-4,
            "actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_adopted(f: &Fixture, converted: &ArticulationHandoff) {
    let world = f.app.world();
    let physics = world.resource::<PhysicsWorldResource>();
    let bindings = world.resource::<PhysicsBindings>();
    assert_eq!(converted.report.articulation, f.source);
    assert_eq!(converted.report.broken_link, 1);
    assert_eq!(converted.bodies.len(), 3);
    assert_eq!(converted.report.joints.len(), 1);
    assert_eq!(converted.report.colliders.len(), 6);
    assert_eq!(
        physics.bodies.len(),
        3,
        "never allocate a body per attachment/leaf"
    );
    assert_eq!(physics.joints.len(), 1);
    assert!(!physics.articulations.contains(f.source));
    assert!(world.get::<ArticulationSnapshot>(f.owner).is_none());
    assert_eq!(
        world
            .resource::<ArticulationBindings>()
            .articulation(f.owner),
        None
    );
    assert_eq!(bindings.body(f.owner), None);
    let leaves = physics.collider_leaves(&QueryFilter::ALL);
    assert_eq!(leaves.len(), 12);
    for (body_index, member) in converted.bodies.iter().enumerate() {
        let mappings: Vec<_> = converted
            .report
            .colliders
            .iter()
            .filter(|mapping| mapping.body == member.body)
            .collect();
        assert_eq!(mappings.len(), 2);
        assert_eq!(bindings.body(member.entity), Some(member.body));
        assert_eq!(bindings.entity(member.body), Some(member.entity));
        assert_ne!(member.entity, f.owner);
        assert_eq!(bindings.body(member.authored_entity), None);
        assert!(world.get::<RigidBody>(member.authored_entity).is_none());
        assert!(world.get::<Collider>(member.authored_entity).is_none());
        assert_eq!(
            world.relation_targets::<ChildOf>(member.authored_entity),
            [member.entity]
        );
        let collider = world.get::<Collider>(member.entity).unwrap();
        assert_eq!(
            collider.shape,
            CollisionShape::Registered(mappings[0].body_shape)
        );
        assert_eq!(
            physics.bodies.collider(member.body),
            Some(mappings[0].body_shape)
        );
        assert_eq!(
            collider.material,
            physics.bodies.material(member.body).unwrap()
        );
        assert_eq!(mappings[0].body_shape, mappings[1].body_shape);
        assert_eq!(mappings[0].body_child_range, Some(0..2));
        assert_eq!(mappings[1].body_child_range, Some(2..4));
        assert_eq!(
            leaves
                .iter()
                .filter(|leaf| leaf.body == member.body)
                .count(),
            4
        );
        for (attachment_index, mapping) in mappings.into_iter().enumerate() {
            let (id, desc) = &f.attachments[body_index * 2 + attachment_index];
            assert_eq!(mapping.source, *id);
            assert_eq!(mapping.source_shape, f.nested);
            assert_eq!(mapping.link, member.link);
            assert_eq!(mapping.local_pose, desc.local_pose);
            assert_eq!(mapping.material, desc.material);
            assert_eq!(mapping.self_collision, desc.self_collision);
            assert!(physics.articulation_colliders.get(*id).is_none());
            let range = mapping.body_child_range.clone().unwrap();
            for (index, child) in f.children.iter().enumerate() {
                let leaf = leaves
                    .iter()
                    .find(|leaf| {
                        leaf.body == member.body
                            && leaf.child_index == Some(range.start + index as u32)
                    })
                    .unwrap();
                assert_eq!(bindings.entity(leaf.body), Some(member.entity));
                assert_eq!(leaf.body_collider, mapping.body_shape);
                assert_eq!(leaf.collider, f.primitive);
                assert_eq!(leaf.material, desc.material);
                assert_eq!(leaf.self_collision, desc.self_collision);
                close_pose(
                    leaf.pose,
                    f.source_poses[body_index]
                        .mul(&desc.local_pose)
                        .mul(&child.local_pose),
                );
            }
        }
    }
    let a = converted.bodies[0].body;
    let b = converted.bodies[2].body;
    let exclusions = physics.collision_exclusions();
    assert_eq!(exclusions.len(), 3);
    assert_eq!(
        exclusions.membership(a).unwrap().group,
        exclusions.membership(b).unwrap().group
    );
    assert!(!exclusions.allows_leaf_pair(a, a, Some(true), Some(true)));
    // Use the effective queried flags, not the conservative body-level OR.
    for la in leaves.iter().filter(|leaf| leaf.body == a) {
        for lb in leaves.iter().filter(|leaf| leaf.body == b) {
            assert_eq!(
                exclusions.allows_leaf_pair(a, b, Some(la.self_collision), Some(lb.self_collision)),
                la.self_collision && lb.self_collision
            );
        }
    }
}

#[test]
fn repeated_nested_attachments_keep_one_ecs_owner_per_body_and_exact_leaf_policies() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let mut f = fixture(base, false);
        degrade(&mut f);
        let converted = view(&f);
        assert_adopted(&f, &converted);
        let shape_count = f
            .app
            .world()
            .resource::<PhysicsWorldResource>()
            .shapes
            .len();
        for _ in 0..3 {
            f.app.update();
            assert_eq!(view(&f), converted);
            assert_adopted(&f, &converted);
            assert_eq!(
                f.app
                    .world()
                    .resource::<PhysicsWorldResource>()
                    .shapes
                    .len(),
                shape_count
            );
        }
    }
}

#[test]
fn collider_edits_removal_and_reinsertion_keep_body_identity_then_root_readd_retires_group() {
    let mut f = fixture(ArticulationBase::Fixed, false);
    degrade(&mut f);
    let converted = view(&f);
    let member = &converted.bodies[2];
    let original = *f.app.world().get::<Collider>(member.entity).unwrap();
    let edited_material = PhysicsMaterial::new(0.7, 0.3);
    let mut edited = original;
    edited.material = edited_material;
    f.app.world_mut().insert(member.entity, (edited,));
    // After conversion, root authoring edits are not a reduced rebuild request.
    let mut edited_config = f.config.clone();
    edited_config.positions = vec![f32::NAN];
    f.app.world_mut().insert(f.owner, (edited_config,));
    f.app.update();
    assert_eq!(view(&f), converted);
    assert_eq!(
        f.app
            .world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .material(member.body),
        Some(edited_material)
    );
    assert_adopted(&f, &converted); // explicit attachment policies still win

    // Replacing the compound is ordinary ECS input, not a source-map rewrite.
    let mut replacement = Collider::sphere(0.05);
    replacement.material = edited_material;
    f.app.world_mut().insert(member.entity, (replacement,));
    f.app.update();
    assert_eq!(view(&f), converted);
    let physics = f.app.world().resource::<PhysicsWorldResource>();
    let leaves = physics.collider_leaves(&QueryFilter::ALL);
    let own: Vec<_> = leaves
        .iter()
        .filter(|leaf| leaf.body == member.body)
        .collect();
    assert_eq!(own.len(), 1);
    assert_eq!(own[0].child_index, None);
    assert_eq!(own[0].material, edited_material);
    assert_eq!(leaves.len(), 9);

    f.app.world_mut().remove::<Collider>(member.entity);
    f.app.update();
    assert_eq!(view(&f), converted);
    assert_eq!(
        f.app
            .world()
            .resource::<PhysicsBindings>()
            .body(member.entity),
        Some(member.body)
    );
    let physics = f.app.world().resource::<PhysicsWorldResource>();
    assert_eq!(physics.bodies.collider(member.body), None);
    assert_eq!(physics.collider_leaves(&QueryFilter::ALL).len(), 8);
    f.app.world_mut().insert(member.entity, (original,));
    f.app.update();
    assert_adopted(&f, &converted);

    // Shapes are append-only shared allocations, not group-owned entities.
    let (shape_count, foreign) = {
        let physics = f.app.world_mut().resource_mut::<PhysicsWorldResource>();
        let foreign =
            physics.spawn(BodyDesc::static_at(Vec3::splat(50.0)).with_collider(f.primitive));
        (physics.shapes.len(), foreign)
    };
    f.app.world_mut().remove::<ArticulationBinding>(f.owner);
    f.app.world_mut().insert(f.owner, (f.config.clone(),));
    f.app.update();
    let physics = f.app.world().resource::<PhysicsWorldResource>();
    let bindings = f.app.world().resource::<PhysicsBindings>();
    assert_eq!(physics.bodies.len(), 1);
    assert!(physics.bodies.contains(foreign));
    assert!(physics.joints.is_empty());
    assert!(physics.collision_exclusions().is_empty());
    assert_eq!(physics.shapes.len(), shape_count);
    for mapping in &converted.report.colliders {
        assert!(physics.shapes.get(mapping.body_shape).is_some());
    }
    for member in &converted.bodies {
        assert!(!physics.bodies.contains(member.body));
        assert!(!f.app.world().contains(member.entity));
        assert!(!f.app.world().contains(member.authored_entity));
        assert_eq!(bindings.body(member.entity), None);
        assert_eq!(bindings.entity(member.body), None);
    }
    let articulation = f.app.world().resource::<ArticulationBindings>();
    assert!(articulation.handoff(f.owner).is_none());
    let next = articulation.articulation(f.owner).unwrap();
    assert_ne!(next, f.source);
    assert!(f.app.world().get::<ArticulationSnapshot>(f.owner).is_some());
    assert!(physics
        .articulation_colliders
        .iter()
        .all(|(_, desc)| desc.articulation != f.source));
}

#[test]
fn failed_multi_attachment_handoff_retains_all_sources_and_retries_without_partial_adoption() {
    let mut f = fixture(ArticulationBase::Fixed, true);
    let entity_count = f.app.world().entity_count();
    let shape_count = f
        .app
        .world()
        .resource::<PhysicsWorldResource>()
        .shapes
        .len();
    degrade(&mut f);
    for _ in 0..3 {
        let world = f.app.world();
        let physics = world.resource::<PhysicsWorldResource>();
        let bindings = world.resource::<ArticulationBindings>();
        assert_eq!(bindings.articulation(f.owner), Some(f.source));
        assert!(bindings.handoff(f.owner).is_none());
        assert_eq!(
            world
                .get::<ArticulationSnapshot>(f.owner)
                .unwrap()
                .broken_joint,
            Some(1)
        );
        assert_eq!(world.entity_count(), entity_count);
        assert!(physics.bodies.is_empty());
        assert!(physics.joints.is_empty());
        assert!(physics.collision_exclusions().is_empty());
        assert_eq!(physics.shapes.len(), shape_count);
        assert_eq!(
            physics.articulation_colliders.iter().count(),
            f.attachments.len()
        );
        for (id, desc) in &f.attachments {
            assert_eq!(physics.articulation_colliders.get(*id), Some(desc));
        }
        assert!(world
            .resource::<ArticulationDiagnostics>()
            .issues
            .iter()
            .any(|issue| issue.entity == Some(f.owner)
                && issue.issue
                    == ArticulationIssue::HandoffFailed(
                        ArticulationHandoffError::UnsupportedJointPolicy(
                            UnsupportedJointPolicy::InertOrExtraParameters { link: 0 }
                        )
                    )));
        f.app.update();
    }
    // Match the existing handoff retry API: repair the stopped core runtime
    // in-place; preserve its generational identity and every attachment ID.
    let mut config = f.config.clone();
    config.links[0].joint.damping = 0.0;
    let mut repaired = config.build().unwrap();
    repaired.break_joint(1).unwrap();
    *f.app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .articulations
        .get_mut(f.source)
        .unwrap() = repaired;
    f.app.update();
    let converted = view(&f);
    assert_adopted(&f, &converted);
    let converted_shape_count = f
        .app
        .world()
        .resource::<PhysicsWorldResource>()
        .shapes
        .len();
    let converted_entity_count = f.app.world().entity_count();
    for _ in 0..3 {
        f.app.update();
        assert_eq!(view(&f), converted);
        assert_adopted(&f, &converted);
        assert_eq!(f.app.world().entity_count(), converted_entity_count);
        assert_eq!(
            f.app
                .world()
                .resource::<PhysicsWorldResource>()
                .shapes
                .len(),
            converted_shape_count
        );
    }
}

#[test]
fn removing_one_generated_member_retires_all_multi_leaf_owners_without_resurrection() {
    let mut f = fixture(ArticulationBase::Fixed, false);
    degrade(&mut f);
    let converted = view(&f);
    assert_adopted(&f, &converted);
    let shape_count = f
        .app
        .world()
        .resource::<PhysicsWorldResource>()
        .shapes
        .len();
    f.app
        .world_mut()
        .remove::<RigidBody>(converted.bodies[2].entity);
    f.app.update();
    assert!(f
        .app
        .world()
        .resource::<ArticulationDiagnostics>()
        .issues
        .iter()
        .any(|issue| issue.entity == Some(f.owner)
            && issue.issue == ArticulationIssue::HandoffInvalidated));
    for _ in 0..3 {
        let world = f.app.world();
        let physics = world.resource::<PhysicsWorldResource>();
        let bindings = world.resource::<PhysicsBindings>();
        assert!(world
            .resource::<ArticulationBindings>()
            .handoff(f.owner)
            .is_none());
        assert_eq!(
            world
                .resource::<ArticulationBindings>()
                .articulation(f.owner),
            None
        );
        assert!(physics.bodies.is_empty());
        assert!(physics.joints.is_empty());
        assert!(physics.articulations.is_empty());
        assert!(physics.collision_exclusions().is_empty());
        assert!(physics.collider_leaves(&QueryFilter::ALL).is_empty());
        assert_eq!(physics.shapes.len(), shape_count);
        for member in &converted.bodies {
            assert!(!world.contains(member.entity));
            assert!(!world.contains(member.authored_entity));
            assert_eq!(bindings.body(member.entity), None);
            assert_eq!(bindings.entity(member.body), None);
        }
        f.app.update();
    }
}
