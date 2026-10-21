//! Adoption and retirement of core-generated bodies; never spawn a second body
//! through the generic synchronizer. Shape registry entries are shared immutable
//! world allocations (append-only); retiring a group removes references, not shapes.

use super::*;
use crate::{
    BodyBinding, Collider, CollisionShape, PhysicsBindings, RemovalSignals, RigidBody, Velocity,
};
use prism_physics_core::reduced::articulation::handoff::{
    ArticulationHandoffError, ArticulationHandoffReport,
};
use prism_physics_core::BodyHandle;
use prism_transform::Transform;

/// One complete ECS/body identity. None denotes the articulation base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArticulationBodyEntity {
    /// Authored link index, or the base.
    pub link: Option<usize>,
    /// Independent unparented ordinary rigid-body entity, owned by the root.
    /// Transform/Velocity inputs use core's COM/principal-inertia body frame.
    pub entity: Entity,
    /// Owned attachment/output frame for the original authored link (or base).
    /// Its immutable local Transform is inverse(link_from_body), parented to
    /// `entity`. Attach visuals here; apply physical inputs to `entity`, not here.
    /// Its GlobalTransform is body_world * inverse(link_from_body).
    pub authored_entity: Entity,
    /// Full generational identity adopted from core.
    pub body: BodyHandle,
}

/// Last-boundary ownership view. Collider/joint mappings retain exact core
/// identities; source articulation/collider identities are historical only.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationHandoff {
    /// Atomic core conversion report, including owned surviving joint handles.
    pub report: ArticulationHandoffReport,
    /// Base first, followed by authored link order.
    pub bodies: Vec<ArticulationBodyEntity>,
}

pub(super) struct OwnedHandoff {
    pub(super) view: ArticulationHandoff,
    members: Vec<Member>,
    materialized: bool,
}

#[derive(Clone)]
struct Member {
    entity: Entity,
    authored_entity: Entity,
    authored_local: Transform,
    binding: BodyBinding,
    collider: Option<Collider>,
}

pub(super) fn convert(
    world: &mut World,
    physics: &mut PhysicsWorldResource,
    owner: Entity,
    handle: ArticulationHandle,
) -> Result<OwnedHandoff, ArticulationHandoffError> {
    // No ECS allocation, callback or shared identity mutation precedes this
    // atomic core operation. Failure leaves the entire source tree untouched.
    let report = physics.handoff_articulation(handle)?;
    let mut members = Vec::with_capacity(report.links.len() + 1);
    let mut bodies = Vec::with_capacity(report.links.len() + 1);
    for mapping in std::iter::once(&report.base).chain(&report.links) {
        let body = mapping.body;
        let position = physics.bodies.position(body).expect("handoff body");
        let rotation = physics.bodies.orientation(body).expect("handoff body");
        let pose = Transform {
            translation: prism_math::Vec3::new(position.x, position.y, position.z),
            rotation: prism_math::Quat::from_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
            scale: prism_math::Vec3::ONE,
        };
        let velocity = Velocity {
            linear: physics.bodies.linear_velocity(body).expect("handoff body"),
            angular: physics.bodies.angular_velocity(body).expect("handoff body"),
        };
        let authored = RigidBody {
            kind: physics.bodies.kind(body).expect("handoff body"),
            mass: physics.bodies.mass_properties(body).expect("handoff body"),
            linear_damping: 0.0,
            angular_damping: 0.0,
            ccd: false,
        };
        let collider = physics.bodies.collider(body).map(|shape| Collider {
            shape: CollisionShape::Registered(shape),
            material: physics.bodies.material(body).expect("handoff body"),
            sensor: false,
        });
        // Empty entities have no user component callbacks. Restore all shared
        // resources and ownership views before publishing the component bundle.
        let entity = world.spawn(());
        let authored_entity = world.spawn(());
        let body_from_link = mapping.link_from_body.inverse();
        let authored_local = Transform {
            translation: prism_math::Vec3::new(
                body_from_link.translation.x,
                body_from_link.translation.y,
                body_from_link.translation.z,
            ),
            rotation: prism_math::Quat::from_xyzw(
                body_from_link.rotation.x,
                body_from_link.rotation.y,
                body_from_link.rotation.z,
                body_from_link.rotation.w,
            ),
            scale: prism_math::Vec3::ONE,
        };
        debug_assert_ne!(entity, owner);
        bodies.push(ArticulationBodyEntity {
            link: mapping.link,
            entity,
            authored_entity,
            body,
        });
        members.push(Member {
            entity,
            authored_entity,
            authored_local,
            binding: BodyBinding {
                handle: body,
                authored,
                pose,
                velocity: Some(velocity),
                active: true,
            },
            collider,
        });
    }
    Ok(OwnedHandoff {
        view: ArticulationHandoff { report, bodies },
        members,
        materialized: false,
    })
}

fn adopt(world: &mut World, member: &Member) {
    world.init_resource::<PhysicsBindings>();
    let bindings = world.resource_mut::<PhysicsBindings>();
    bindings.entities.insert(
        (
            member.binding.handle.index(),
            member.binding.handle.generation(),
        ),
        member.entity,
    );
    bindings
        .bodies
        .insert(member.entity, member.binding.clone());
}

pub(super) fn materialize(world: &mut World) {
    let pending: Vec<_> = world
        .resource::<ArticulationRuntime>()
        .bindings
        .iter()
        .filter_map(|(&owner, binding)| {
            binding
                .handoff
                .as_ref()
                .filter(|owned| !owned.materialized)
                .map(|owned| (owner, owned.members.clone()))
        })
        .collect();
    let changed = !pending.is_empty();
    for (owner, members) in pending {
        // Mark first: callbacks must never cause a second materialization.
        if let Some(owned) = world
            .resource_mut::<ArticulationRuntime>()
            .bindings
            .get_mut(&owner)
            .and_then(|binding| binding.handoff.as_mut())
        {
            owned.materialized = true;
        }
        // Publish all body identities before the first component callback.
        for member in &members {
            adopt(world, member);
        }
        for member in members {
            if !can_materialize(world, owner, &member) {
                break;
            }
            // Keep physical inputs in body coordinates. A separate immutable
            // child carries the authored frame through ordinary propagation,
            // including after generic sync accepts a body pose override.
            let before = observer_resources(world);
            world.add_relation::<ChildOf>(member.authored_entity, member.entity);
            world.insert(member.authored_entity, (member.authored_local,));
            after_snapshot_observers(world, before);
            if !can_materialize(world, owner, &member) {
                break;
            }
            let before = observer_resources(world);
            if let Some(collider) = member.collider {
                world.insert(
                    member.entity,
                    (
                        member.binding.authored,
                        member.binding.pose,
                        member.binding.velocity.unwrap(),
                        collider,
                    ),
                );
            } else {
                world.insert(
                    member.entity,
                    (
                        member.binding.authored,
                        member.binding.pose,
                        member.binding.velocity.unwrap(),
                    ),
                );
            }
            after_snapshot_observers(world, before);
        }
    }
    if changed {
        let before = observer_resources(world);
        world.flush_changed_observers();
        if prism_transform::ecs::propagate_transforms(world).is_err() {
            if let Some(diagnostics) = world.get_resource_mut::<PhysicsDiagnostics>() {
                diagnostics.issues.push(crate::PhysicsDiagnostic {
                    entity: None,
                    issue: crate::PhysicsIssue::TransformPropagation,
                });
            }
        }
        world.flush_changed_observers();
        after_snapshot_observers(world, before);
    }
}

fn can_materialize(world: &World, owner: Entity, member: &Member) -> bool {
    world.get::<ArticulationBinding>(owner).is_some()
        && !world.resource::<RemovedAuthoring>().0.contains(&owner)
        && world.contains_resource::<PhysicsWorldResource>()
        && world.contains(member.entity)
        && world.contains(member.authored_entity)
}

fn valid_member(world: &World, physics: &PhysicsWorldResource, member: &Member) -> bool {
    if !physics.bodies.contains(member.binding.handle)
        || world.get::<RigidBody>(member.entity) != Some(&member.binding.authored)
        || world
            .get_resource::<RemovalSignals>()
            .is_some_and(|s| s.rigid.contains(&member.entity))
        || !world.relation_targets::<ChildOf>(member.entity).is_empty()
        || world.relation_targets::<ChildOf>(member.authored_entity) != [member.entity]
        || world.get::<Transform>(member.authored_entity) != Some(&member.authored_local)
        || world.has::<RigidBody>(member.authored_entity)
        || world.has::<Collider>(member.authored_entity)
    {
        return false;
    }
    let Some(pose) = world.get::<Transform>(member.entity) else {
        return false;
    };
    if pose.scale != prism_math::Vec3::ONE
        || !pose.translation.is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1.0e-4
        || world
            .get::<Velocity>(member.entity)
            .is_some_and(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return false;
    }
    if let Some(collider) = world.get::<Collider>(member.entity) {
        if !collider.material.friction.is_finite()
            || collider.material.friction < 0.0
            || !collider.material.restitution.is_finite()
            || !(0.0..=1.0).contains(&collider.material.restitution)
        {
            return false;
        }
        use prism_physics_core::ColliderShape;
        let valid = match collider.shape {
            CollisionShape::Registered(shape) => physics.shapes.get(shape).is_some(),
            CollisionShape::Analytic(ColliderShape::Sphere { radius }) => {
                radius.is_finite() && radius > 0.0
            }
            CollisionShape::Analytic(ColliderShape::Cuboid { half_extents }) => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
            CollisionShape::Analytic(ColliderShape::Capsule {
                radius,
                half_height,
            }) => {
                radius.is_finite() && radius > 0.0 && half_height.is_finite() && half_height >= 0.0
            }
            CollisionShape::Analytic(ColliderShape::Plane { normal, offset }) => {
                normal.is_finite()
                    && offset.is_finite()
                    && (normal.length_squared() - 1.0).abs() <= 1.0e-4
            }
            _ => false,
        };
        if !valid {
            return false;
        }
    }
    world
        .get_resource::<PhysicsBindings>()
        .and_then(|b| b.bodies.get(&member.entity))
        .is_none_or(|b| b.handle == member.binding.handle && b.active)
}

fn remove_bodies_atomically(
    physics: &mut PhysicsWorldResource,
    bodies: impl IntoIterator<Item = BodyHandle>,
) -> bool {
    let mut bodies = bodies.into_iter();
    let Some(first) = bodies.find(|&body| physics.bodies.contains(body)) else {
        // An empty or already-retired group has no write set. In particular,
        // repeated cleanup must not copy the world's body/assembly tables.
        return true;
    };
    // Keep this write set aligned with PhysicsWorld::remove_body: it only
    // changes body storage, assembly graph and aero registrations. Merge
    // bookkeeping is read-only. In particular, do not clone shared geometry,
    // deformables, other articulations, force queues or event history here.
    let saved_bodies = physics.bodies.clone();
    let saved_assemblies = physics.assemblies.clone();
    let saved_aero = physics.aero_bodies.clone();
    // The shared world is borrowed exclusively and no ECS callbacks occur
    // inside removal. A rejection restores allocator/free-list state as well
    // as the graph and registrations before anyone can observe this boundary.
    for body in std::iter::once(first).chain(bodies) {
        if physics.bodies.contains(body) && !physics.remove_body(body) {
            physics.bodies = saved_bodies;
            physics.assemblies = saved_assemblies;
            physics.aero_bodies = saved_aero;
            return false;
        }
    }
    true
}

pub(super) fn reconcile(
    world: &mut World,
    physics: &mut PhysicsWorldResource,
    runtime: &mut ArticulationRuntime,
) {
    for (&owner, binding) in &mut runtime.bindings {
        let Some(owned) = &mut binding.handoff else {
            continue;
        };
        if !owned.materialized {
            continue;
        }
        if owned
            .members
            .iter()
            .any(|member| !valid_member(world, physics, member))
        {
            runtime.handoff_issues.push(ArticulationDiagnostic {
                entity: Some(owner),
                issue: ArticulationIssue::HandoffInvalidated,
            });
            runtime
                .retired_handoffs
                .push(binding.handoff.take().unwrap());
            continue;
        }
        // Preserve the last ordinary-sync caches privately, including optional
        // velocity removal. Deleting a public map must neither spawn a second
        // body nor swallow a simultaneous authored pose/velocity override.
        for member in &mut owned.members {
            if let Some(current) = world
                .get_resource::<PhysicsBindings>()
                .and_then(|b| b.bodies.get(&member.entity))
            {
                member.binding = current.clone();
                world.resource_mut::<PhysicsBindings>().entities.insert(
                    (
                        member.binding.handle.index(),
                        member.binding.handle.generation(),
                    ),
                    member.entity,
                );
            } else {
                adopt(world, member);
            }
        }
    }
    let mut retry = Vec::new();
    for owned in std::mem::take(&mut runtime.retired_handoffs) {
        let handles: BTreeSet<_> = owned
            .view
            .bodies
            .iter()
            .map(|m| (m.body.index(), m.body.generation()))
            .collect();
        let owns = |h: BodyHandle| handles.contains(&(h.index(), h.generation()));
        let mut joints: Vec<_> = owned.view.report.joints.iter().map(|m| m.joint).collect();
        joints.extend(
            physics
                .assemblies
                .connections()
                .filter(|c| owns(c.a) || owns(c.b))
                .filter_map(|c| c.joint),
        );
        joints.extend(
            physics
                .joints
                .active_entries()
                .filter(|(_, j)| owns(j.anchor_a.body) || owns(j.anchor_b.body))
                .map(|(h, _)| h),
        );
        // Direct core assembly edits may temporarily refuse removal. Keep the
        // complete private cleanup record and retry, without half-deleting joints.
        let removed = remove_bodies_atomically(physics, owned.view.bodies.iter().map(|m| m.body));
        if removed {
            for joint in joints {
                physics.breakable_joints.unregister(joint);
                physics.joints.remove(joint);
            }
        } else {
            runtime.handoff_issues.push(ArticulationDiagnostic {
                entity: None,
                issue: ArticulationIssue::LifecycleDeferred,
            });
        }
        for member in &owned.members {
            if world.contains(member.entity) {
                runtime.despawn_pending.insert(member.entity);
            }
            // Queue explicitly as well as relying on ChildOf's cascade: a
            // caller may have detached the authored frame before invalidation.
            if world.contains(member.authored_entity) {
                runtime.despawn_pending.insert(member.authored_entity);
            }
            if let Some(bindings) = world.get_resource_mut::<PhysicsBindings>() {
                let handle = member.binding.handle;
                if bindings
                    .bodies
                    .get(&member.entity)
                    .is_some_and(|b| b.handle == handle)
                {
                    bindings.bodies.remove(&member.entity);
                }
                let key = (handle.index(), handle.generation());
                if bindings.entities.get(&key) == Some(&member.entity) {
                    bindings.entities.remove(&key);
                    bindings.retired.insert(key, member.entity);
                }
                if bindings
                    .aero
                    .get(&member.entity)
                    .is_some_and(|(h, _)| *h == handle)
                {
                    bindings.aero.remove(&member.entity);
                }
            }
        }
        if !removed {
            retry.push(owned);
        }
    }
    runtime.retired_handoffs = retry;
}

pub(super) fn despawn_retired(world: &mut World) {
    let pending = std::mem::take(&mut world.resource_mut::<ArticulationRuntime>().despawn_pending);
    for entity in pending {
        let before = observer_resources(world);
        // Entity includes its generation. A foreign reused slot is never despawned.
        world.despawn(entity);
        after_snapshot_observers(world, before);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::aero::AeroBodyConfig;
    use prism_physics_core::joint::{DistanceJoint, JointDesc};
    use prism_physics_core::snapshot::world_hash::hash_world;
    use prism_physics_core::{BodyDesc, ColliderShape, PhysicsWorld};

    fn retirement_world() -> (PhysicsWorldResource, BodyHandle, BodyHandle, BodyHandle) {
        let mut physics = PhysicsWorld::with_gravity(Vec3::ZERO);
        let removable = physics.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let foreign = physics.spawn(BodyDesc::dynamic_at(Vec3::X));
        physics.spawn_joint(JointDesc::distance(
            removable,
            Vec3::ZERO,
            foreign,
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
        physics
            .register_aero_body(removable, AeroBodyConfig::default())
            .unwrap();
        let a = physics.spawn(BodyDesc::dynamic_at(Vec3::Y * 3.0));
        let b = physics.spawn(BodyDesc::dynamic_at(Vec3::Y * 4.0));
        physics.spawn_joint(JointDesc::distance(
            a,
            Vec3::ZERO,
            b,
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
        let proxy = physics.merge_assembly(&[a, b]).unwrap();
        (PhysicsWorldResource(physics), removable, foreign, proxy)
    }

    #[test]
    fn retirement_rejection_restores_complete_world_before_retry() {
        let (mut physics, removable, foreign, proxy) = retirement_world();
        let before = hash_world(&physics).unwrap();
        let mut allocation_reference = physics.0.clone();
        // The first removal succeeds and mutates all three backed-up fields;
        // the second is a live merged proxy and must refuse removal. Repeat to
        // check that a retry does not accumulate partial allocator/graph edits.
        for _ in 0..2 {
            assert!(!remove_bodies_atomically(&mut physics, [removable, proxy]));
            assert_eq!(hash_world(&physics).unwrap(), before);
            assert!(physics.bodies.contains(removable));
            assert!(physics.bodies.contains(foreign));
            assert!(physics.aero_body(removable).is_some());
        }
        assert_eq!(
            physics.spawn(BodyDesc::dynamic_at(Vec3::Z)),
            allocation_reference.spawn(BodyDesc::dynamic_at(Vec3::Z)),
            "failed retirement must restore generation/free-list state"
        );
        // Private merge recovery must be untouched as well.
        assert_eq!(
            physics.split_assembly(proxy).unwrap(),
            allocation_reference.split_assembly(proxy).unwrap()
        );
        assert_eq!(
            hash_world(&physics).unwrap(),
            hash_world(&allocation_reference).unwrap()
        );
        assert!(remove_bodies_atomically(&mut physics, [removable]));
    }

    #[test]
    fn retirement_with_no_live_members_preserves_world_and_allocator() {
        let (mut physics, removable, foreign, proxy) = retirement_world();
        let stale = physics.spawn(BodyDesc::dynamic_at(Vec3::Z));
        assert!(physics.remove_body(stale));
        let replacement = physics.spawn(BodyDesc::dynamic_at(Vec3::Z * 2.0));
        assert_eq!(stale.index(), replacement.index());
        assert_ne!(stale, replacement);
        let before = hash_world(&physics).unwrap();
        let mut reference = physics.0.clone();
        assert!(remove_bodies_atomically(&mut physics, []));
        for _ in 0..2 {
            assert!(remove_bodies_atomically(&mut physics, [stale, stale]));
            assert_eq!(hash_world(&physics).unwrap(), before);
        }
        for body in [removable, foreign, proxy, replacement] {
            assert!(physics.bodies.contains(body));
        }
        assert!(physics.aero_body(removable).is_some());
        assert_eq!(
            physics.spawn(BodyDesc::dynamic_at(Vec3::Z * 3.0)),
            reference.spawn(BodyDesc::dynamic_at(Vec3::Z * 3.0))
        );
        assert_eq!(
            physics.split_assembly(proxy).unwrap(),
            reference.split_assembly(proxy).unwrap()
        );
        assert_eq!(
            hash_world(&physics).unwrap(),
            hash_world(&reference).unwrap()
        );
    }

    #[test]
    fn retirement_success_matches_full_clone_without_replacing_shared_shapes() {
        let (mut physics, removable, foreign, proxy) = retirement_world();
        let shape = physics.shapes.insert(ColliderShape::Sphere { radius: 0.3 });
        let shape_address = physics.shapes.get(shape).unwrap() as *const _;
        let stale = physics.spawn(BodyDesc::dynamic_at(Vec3::Z));
        assert!(physics.remove_body(stale));
        let replacement = physics.spawn(BodyDesc::dynamic_at(Vec3::Z * 2.0));
        assert_eq!(stale.index(), replacement.index());
        assert_ne!(stale, replacement);
        let mut reference = physics.0.clone();
        assert!(reference.remove_body(removable));
        assert!(remove_bodies_atomically(
            &mut physics,
            [stale, removable, removable]
        ));
        assert_eq!(
            hash_world(&physics).unwrap(),
            hash_world(&reference).unwrap()
        );
        assert_eq!(
            physics.shapes.get(shape).unwrap() as *const _,
            shape_address
        );
        assert!(!physics.bodies.contains(removable));
        assert!(physics.bodies.contains(foreign));
        assert!(physics.bodies.contains(proxy));
        assert!(physics.bodies.contains(replacement));
        assert!(physics.aero_body(removable).is_none());
    }
}
