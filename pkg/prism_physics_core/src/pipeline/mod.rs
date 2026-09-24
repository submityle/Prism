//! Collision pipeline: broad phase plus narrow phase over a whole world.
//!
//! [`detect_contacts`] walks every live body in a
//! [`PhysicsWorld`](crate::world::PhysicsWorld), finds candidate overlapping
//! pairs with the geometry crate's [`DynamicBvh`], and turns each candidate
//! into a [`ContactManifold`] via the narrow-phase
//! [`generate_contact`](crate::collide::generate_contact) dispatch. The
//! resulting manifolds carry stable [`BodyHandle`]s so the solver can address
//! bodies by storage slot.
//!
//! # Planes are special
//!
//! A [`ColliderShape::Plane`](crate::collider::ColliderShape::Plane) is an
//! infinite half-space; its local AABB spans the full float range and would
//! poison the BVH (every query would match, and the fat-box maths would
//! overflow to non-finite values). Planes are therefore kept out of the BVH and
//! paired explicitly against every finite body instead.
//!
//! # Pair filtering
//!
//! A pair is only reported when at least one of its bodies is
//! [`BodyKind::Dynamic`](crate::state::body::BodyKind::Dynamic); two
//! static/kinematic bodies can never need contact resolution.
//!
//! # Provenance
//!
//! The broad-phase/narrow-phase split, AABB sweep, and plane special-casing are
//! standard collision-detection engineering (see Ericson, *Real-Time Collision
//! Detection*). This file contains no Unreal Engine source or derived code.

use crate::collide::{generate_contact, ContactManifold};
use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::Vec3;
use prism_physics_geometry::{generate_pairs, Aabb, DynamicBvh};

/// A live body ready for collision: its slot, handle, shape, and world pose.
struct PosedBody<'a> {
    /// Stable handle used to stamp the resulting manifolds.
    handle: BodyHandle,
    /// The body's collision shape.
    shape: &'a ColliderShape,
    /// The body's world-space transform.
    pose: Isometry,
    /// The body's simulation category.
    kind: BodyKind,
}

impl PosedBody<'_> {
    /// Returns `true` if this body is a fully simulated dynamic body.
    fn is_dynamic(&self) -> bool {
        self.kind == BodyKind::Dynamic
    }
}

/// Detects all contact manifolds in `world` for the current body poses.
///
/// The returned manifolds have their body handles stamped and follow the frozen
/// [`ContactManifold`] conventions. Bodies without a collider, freed slots, and
/// pairs of two non-dynamic bodies are skipped.
#[must_use]
pub fn detect_contacts(world: &PhysicsWorld) -> Vec<ContactManifold> {
    let bodies = collect_bodies(world);

    // Partition into finite bodies (BVH) and infinite planes (paired directly).
    let mut finite: Vec<usize> = Vec::with_capacity(bodies.len());
    let mut planes: Vec<usize> = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        if matches!(body.shape, ColliderShape::Plane { .. }) {
            planes.push(i);
        } else {
            finite.push(i);
        }
    }

    let mut manifolds = Vec::new();
    broad_phase_finite(&bodies, &finite, &mut manifolds);
    pair_finite_with_planes(&bodies, &finite, &planes, &mut manifolds);
    manifolds
}

/// Gathers every live, collidable body in the world into [`PosedBody`] records.
fn collect_bodies(world: &PhysicsWorld) -> Vec<PosedBody<'_>> {
    let mut out = Vec::new();
    for slot in 0..world.bodies.slot_count() {
        let Some(handle) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        let Some(collider) = world.bodies.collider(handle) else {
            continue;
        };
        let Some(shape) = world.shapes.get(collider) else {
            continue;
        };
        let position = world.bodies.position(handle).unwrap_or(Vec3::ZERO);
        let orientation = world.bodies.orientation(handle).unwrap_or_default();
        let kind = world.bodies.kind(handle).unwrap_or(BodyKind::Static);
        out.push(PosedBody {
            handle,
            shape,
            pose: Isometry::new(position, orientation),
            kind,
        });
    }
    out
}

/// Runs the BVH broad phase over the finite bodies and appends their manifolds.
fn broad_phase_finite(
    bodies: &[PosedBody<'_>],
    finite: &[usize],
    manifolds: &mut Vec<ContactManifold>,
) {
    if finite.len() < 2 {
        return;
    }
    let mut bvh = DynamicBvh::with_capacity(finite.len());
    for &i in finite {
        bvh.insert(world_aabb(&bodies[i]), i as u64);
    }
    for pair in generate_pairs(&bvh) {
        let a = pair.a as usize;
        let b = pair.b as usize;
        if !bodies[a].is_dynamic() && !bodies[b].is_dynamic() {
            continue;
        }
        push_contact(&bodies[a], &bodies[b], manifolds);
    }
}

/// Pairs every finite body against every plane and appends their manifolds.
fn pair_finite_with_planes(
    bodies: &[PosedBody<'_>],
    finite: &[usize],
    planes: &[usize],
    manifolds: &mut Vec<ContactManifold>,
) {
    for &p in planes {
        for &f in finite {
            if !bodies[p].is_dynamic() && !bodies[f].is_dynamic() {
                continue;
            }
            push_contact(&bodies[p], &bodies[f], manifolds);
        }
    }
}

/// Runs the narrow phase for one ordered body pair and appends the manifold.
fn push_contact(a: &PosedBody<'_>, b: &PosedBody<'_>, manifolds: &mut Vec<ContactManifold>) {
    if let Some(manifold) = generate_contact(a.shape, &a.pose, b.shape, &b.pose) {
        manifolds.push(manifold.with_bodies(a.handle, b.handle));
    }
}

/// Computes the world-space AABB of a posed finite body from its local AABB.
fn world_aabb(body: &PosedBody<'_>) -> Aabb {
    let (min, max) = body.shape.local_aabb();
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let world: [Vec3; 8] = core::array::from_fn(|i| body.pose.transform_point(corners[i]));
    Aabb::from_points(&world)
        .unwrap_or_else(|| Aabb::new(body.pose.translation, body.pose.translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::PhysicsMaterial;
    use crate::state::body::{BodyDesc, MassProperties};
    use glam::Quat;

    fn world_with_ground() -> (PhysicsWorld, crate::collider::ColliderHandle) {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let plane = world.shapes.insert(ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        });
        (world, plane)
    }

    #[test]
    fn box_resting_on_plane_produces_a_manifold() {
        let (mut world, plane) = world_with_ground();
        let cuboid = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(plane)
                .with_material(PhysicsMaterial::DEFAULT),
        );
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(0.0, 0.45, 0.0))
                .with_collider(cuboid)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        );
        let manifolds = detect_contacts(&world);
        assert_eq!(manifolds.len(), 1);
        assert!(!manifolds[0].is_empty());
    }

    #[test]
    fn two_static_bodies_are_not_reported() {
        let (mut world, plane) = world_with_ground();
        let cuboid = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        });
        world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(plane));
        world.spawn(BodyDesc::static_at(Vec3::new(0.0, 0.45, 0.0)).with_collider(cuboid));
        assert!(detect_contacts(&world).is_empty());
    }

    #[test]
    fn separated_boxes_produce_no_contact() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let cuboid = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        });
        world.spawn(
            BodyDesc::dynamic_at(Vec3::ZERO)
                .with_collider(cuboid)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        );
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(5.0, 0.0, 0.0))
                .with_collider(cuboid)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        );
        assert!(detect_contacts(&world).is_empty());
    }

    #[test]
    fn rotation_is_accounted_for_in_world_aabb() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let rod = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(2.0, 0.1, 0.1),
        });
        let cube = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.15),
        });
        // A long rod aligned with X, then rotated 90 degrees about Z so it
        // instead spans Y in [-2, 2]. Without rotating the AABB, the broad
        // phase would think it only reaches y = 0.1 and miss the overlap.
        let rod_body = world.spawn(
            BodyDesc::dynamic_at(Vec3::ZERO)
                .with_collider(rod)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        );
        world.bodies.set_orientation(
            rod_body,
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        );
        // A small cube well above the rod's un-rotated extent but inside the
        // rotated one.
        world.spawn(
            BodyDesc::dynamic_at(Vec3::new(0.0, 1.5, 0.0))
                .with_collider(cube)
                .with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
        );
        assert_eq!(detect_contacts(&world).len(), 1);
    }
}
