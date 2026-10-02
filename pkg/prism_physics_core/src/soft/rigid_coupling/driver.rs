//! Step driver for the cloth↔rigid two-way coupling bridge.
//!
//! This is the pipeline glue that closes the loop between a [`PhysicsWorld`]'s
//! rigid bodies and a soft body's particle arrays, using the already-tested
//! [`resolve_two_way_coupling`] kernel as the math core. One substep of the
//! bridge is:
//!
//! 1. **Gather** ([`gather_rigid_proxies`]): walk the rigid bodies, build a
//!    [`RigidProxy`] (a [`CouplingBody`] + its [`BodyHandle`]) for every body
//!    whose [`ColliderShape`] has an analytic soft proxy, and cull the ones
//!    whose world AABB does not overlap the soft body's AABB.
//! 2. **Resolve** ([`couple_cloth_to_rigid`]): run the Jacobi two-way kernel so
//!    the soft particles and the proxies exchange a mass-weighted push, and each
//!    proxy accumulates the reaction impulse the particles exerted on it.
//! 3. **Write back**: for every *movable* proxy, apply its reaction impulse to
//!    the rigid body as a linear velocity delta (`impulse * inverse_mass`) and
//!    translate the rigid body by exactly the displacement the pass applied to
//!    its proxy collider, then clear the proxy's reaction accumulator.
//!
//! The write-back of *this* driver is **linear only**. The kernel reduces every
//! particle contact on a body down to one net linear impulse without retaining
//! the per-contact lever arms, so there is no honest torque to apply at this
//! site; nothing here fakes an angular response. The opt-in per-contact angular
//! bridge in [`super::angular`] ([`super::couple_cloth_to_rigid_angular`]) adds
//! a real `Σ arm × impulse` torque on top of this linear pass.
//!
//! The whole stage is a no-op unless [`PhysicsWorld::cloth_coupling`] is
//! enabled, so a world that never opts in keeps a bit-identical rigid-only
//! path.
//!
//! # Determinism
//!
//! Proxies are gathered in slot order, the kernel is array-in / array-out and
//! Jacobi, and write-back visits proxies in the same order, so the whole bridge
//! is deterministic: identical inputs produce identical particle arrays, rigid
//! state, and [`CouplingReport`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is
//! pipeline wiring over the textbook position-based-dynamics coupling kernel in
//! [`crate::soft::collision::coupling`].

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::resolve_two_way_coupling;
use crate::world::PhysicsWorld;

use super::proxy::{
    body_collider_from_shape, collider_anchor, collider_overlaps, proxy_inverse_mass, Aabb,
    RigidProxy,
};

pub use crate::soft::CouplingBody;

/// A summary of what one [`couple_cloth_to_rigid`] pass did, for tests,
/// debugging, and force-feedback readouts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CouplingReport {
    /// Number of rigid proxies that participated in the pass (after culling).
    pub proxy_count: usize,
    /// Number of *movable* proxies whose reaction impulse was written back onto
    /// a rigid body.
    pub applied_count: usize,
    /// Sum of the reaction impulses written back onto the rigid bodies.
    pub applied_impulse: Vec3,
}

/// Gathers a [`RigidProxy`] for every rigid body that has an analytic coupling
/// proxy and whose world AABB overlaps `soft_aabb`.
///
/// Bodies are visited in storage slot order for determinism. A body is skipped
/// when it has no collider, its shape has no analytic proxy
/// ([`body_collider_from_shape`] returns [`None`]), or its proxy does not
/// overlap the soft body's AABB. Static and kinematic bodies are kept
/// (they record reaction impulse for force feedback) but given a zero inverse
/// mass so they never move.
#[must_use]
pub fn gather_rigid_proxies(world: &PhysicsWorld, soft_aabb: &Aabb) -> Vec<RigidProxy> {
    let mut proxies = Vec::new();
    for slot in 0..world.bodies.slot_count() {
        let Some(handle) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        let Some(collider_handle) = world.bodies.collider(handle) else {
            continue;
        };
        let Some(shape) = world.shapes.get(collider_handle) else {
            continue;
        };
        let position = world.bodies.position(handle).unwrap_or_default();
        let orientation = world.bodies.orientation(handle).unwrap_or_default();
        let Some(collider) = body_collider_from_shape(shape, position, orientation) else {
            continue;
        };
        if !collider_overlaps(collider, soft_aabb) {
            continue;
        }
        let kind = world.bodies.kind(handle).unwrap_or_default();
        let inv_mass = world
            .bodies
            .mass_properties(handle)
            .map_or(0.0, |m| m.inv_mass);
        let proxy_inv_mass = proxy_inverse_mass(kind, inv_mass);
        proxies.push(RigidProxy {
            handle,
            body: CouplingBody::new(collider, proxy_inv_mass),
        });
    }
    proxies
}

/// Runs one substep of two-way cloth↔rigid coupling and writes the rigid
/// reaction back onto the bodies.
///
/// `positions` is the soft body's particle position column (corrected in place)
/// and `inverse_masses` the index-aligned inverse-mass column; `dt` is the
/// substep. The function:
///
/// 1. Returns an empty [`CouplingReport`] immediately when the coupling stage
///    is disabled, `dt <= 0`, the particle arrays are empty, or no rigid proxy
///    overlaps the soft body (nothing to do, rigid-only path untouched).
/// 2. Gathers overlapping proxies, snapshots each proxy collider's anchor,
///    and runs [`resolve_two_way_coupling`].
/// 3. For every movable proxy writes the reaction impulse back as a linear
///    velocity delta and translates the body by the pass's proxy displacement,
///    then clears the proxy's reaction accumulator.
pub fn couple_cloth_to_rigid(
    world: &mut PhysicsWorld,
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    dt: Real,
) -> CouplingReport {
    if !world.cloth_coupling.is_enabled() || dt <= 0.0 || positions.is_empty() {
        return CouplingReport::default();
    }
    let Some(soft_aabb) = Aabb::of_points(positions) else {
        return CouplingReport::default();
    };
    let proxies = gather_rigid_proxies(world, &soft_aabb);
    if proxies.is_empty() {
        return CouplingReport::default();
    }

    // Snapshot the pre-pass anchors so the write-back can translate each body
    // by exactly the displacement the Jacobi pass applied to its proxy.
    let mut coupling_bodies: Vec<CouplingBody> = proxies.iter().map(|p| p.body).collect();
    let anchors_before: Vec<Vec3> = coupling_bodies
        .iter()
        .map(|b| collider_anchor(b.collider))
        .collect();

    resolve_two_way_coupling(positions, inverse_masses, &mut coupling_bodies, dt);

    let mut report = CouplingReport {
        proxy_count: proxies.len(),
        ..CouplingReport::default()
    };

    for (idx, (proxy, body)) in proxies.iter().zip(coupling_bodies.iter_mut()).enumerate() {
        if body.inverse_mass > 0.0 {
            // Linear reaction: a Newton impulse maps to a velocity delta by the
            // inverse mass. This matches the kernel, where the proxy's rigid
            // translation equals `reaction_impulse * inverse_mass * dt`.
            let velocity_delta = body.reaction_impulse * body.inverse_mass;
            if let Some(v0) = world.bodies.linear_velocity(proxy.handle) {
                world
                    .bodies
                    .set_linear_velocity(proxy.handle, v0 + velocity_delta);
            }
            // Translate the body by exactly the displacement the pass applied to
            // the proxy collider (its anchor delta), keeping the rigid body and
            // its proxy consistent.
            let translation = collider_anchor(body.collider) - anchors_before[idx];
            if let Some(p0) = world.bodies.position(proxy.handle) {
                world.bodies.set_position(proxy.handle, p0 + translation);
            }
            report.applied_count += 1;
            report.applied_impulse += body.reaction_impulse;

            // Angular response is intentionally not applied here: this
            // linear driver reduces every particle contact on a body to a
            // single net linear impulse and discards the per-contact arms, so
            // there is no honest torque at this site. The opt-in per-contact
            // angular bridge in `super::angular`
            // ([`super::couple_cloth_to_rigid_angular`]) re-derives the arms
            // from the pre-pass contacts and writes `Σ arm × impulse` onto the
            // body's angular velocity; it layers on top of this linear pass so
            // the linear path here stays bit-identical.
        }
        body.clear_reaction();
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::soft::rigid_coupling::ClothRigidCouplingConfig;
    use crate::soft::BodyCollider;
    use crate::state::body::{BodyDesc, BodyKind, MassProperties};
    use crate::world::PhysicsWorld;
    use crate::WorldConfig;

    fn spawn_sphere(
        world: &mut PhysicsWorld,
        kind: BodyKind,
        inv_mass: Real,
        position: Vec3,
        radius: f32,
    ) -> crate::state::handle::BodyHandle {
        let shape = world.shapes.insert(ColliderShape::Sphere { radius });
        let desc = BodyDesc {
            kind,
            position,
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
    fn disabled_stage_is_a_noop() {
        let mut world = PhysicsWorld::new(WorldConfig::default());
        let handle = spawn_sphere(
            &mut world,
            BodyKind::Dynamic,
            1.0,
            Vec3::new(0.0, 0.1, 0.0),
            0.5,
        );
        let mut positions = vec![Vec3::ZERO];
        let inverse_masses = vec![1.0];
        let report = couple_cloth_to_rigid(&mut world, &mut positions, &inverse_masses, 1.0 / 60.0);
        assert_eq!(report, CouplingReport::default());
        assert_eq!(positions[0], Vec3::ZERO);
        assert_eq!(
            world.bodies.position(handle).unwrap(),
            Vec3::new(0.0, 0.1, 0.0)
        );
    }

    #[test]
    fn cuboid_proxy_is_gathered_as_obb() {
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active();
        let shape = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        });
        let desc = BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::ZERO,
            collider: Some(shape),
            mass_properties: MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ZERO,
            },
            ..BodyDesc::default()
        };
        world.spawn(desc);
        let aabb = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        // A cuboid now yields exactly one movable OBB proxy that overlaps the
        // soft AABB (it used to be skipped as unsupported).
        let proxies = gather_rigid_proxies(&world, &aabb);
        assert_eq!(proxies.len(), 1);
        assert!(matches!(proxies[0].body.collider, BodyCollider::Obb { .. }));
        assert!(proxies[0].body.inverse_mass > 0.0);
    }

    #[test]
    fn far_body_is_culled() {
        let mut world = PhysicsWorld::new(WorldConfig::default());
        spawn_sphere(
            &mut world,
            BodyKind::Dynamic,
            1.0,
            Vec3::new(100.0, 0.0, 0.0),
            0.5,
        );
        let aabb = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        assert!(gather_rigid_proxies(&world, &aabb).is_empty());
    }

    #[test]
    fn static_body_gets_zero_inverse_mass_proxy() {
        let mut world = PhysicsWorld::new(WorldConfig::default());
        spawn_sphere(&mut world, BodyKind::Static, 0.0, Vec3::ZERO, 0.5);
        let aabb = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let proxies = gather_rigid_proxies(&world, &aabb);
        assert_eq!(proxies.len(), 1);
        assert_eq!(proxies[0].body.inverse_mass, 0.0);
    }
}
