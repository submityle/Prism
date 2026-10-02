//! Per-contact *angular* driver for the cloth↔rigid two-way coupling bridge.
//!
//! The shared linear kernel ([`crate::soft::resolve_two_way_coupling`]) reduces
//! every particle contact on a body down to a single net *linear* reaction
//! impulse and discards the per-contact lever arms, so the linear driver
//! ([`super::couple_cloth_to_rigid`]) can only spin a body if it fakes a torque
//! — which it never does. This module closes that honest gap with a real,
//! analytic angular path layered *on top of* the linear bridge:
//!
//! 1. Snapshot the pre-pass particle positions (the lever arms must be measured
//!    against where each particle sat *before* the linear kernel corrected it).
//! 2. Run the untouched linear bridge ([`super::couple_cloth_to_rigid`]) so all
//!    linear effects — particle corrections, body translation, linear velocity
//!    write-back — stay **bit-identical** to the linear-only path.
//! 3. For every movable proxy, re-evaluate each particle against the proxy's
//!    **pre-pass** collider with the shared per-particle kernel
//!    ([`crate::soft::couple_particle_against_body`]) to recover the contact
//!    point and reaction impulse the net reduction threw away, and accumulate
//!    the net angular impulse `Σ arm_i × impulse_i`, where
//!    `arm_i = contact_point_i − center_of_mass`.
//! 4. Map the net angular impulse through the body's **world-space** inverse
//!    inertia `I⁻¹_world = R · diag(inv_inertia) · Rᵀ` and add the resulting
//!    `Δω` to the body's angular velocity. Bodies with a zero inverse inertia
//!    (infinite rotational inertia / non-rotatable) are skipped.
//!
//! This is a strict superset of the linear bridge, so it is **opt-in** behind
//! [`ClothRigidCouplingConfig::angular`](super::ClothRigidCouplingConfig): the
//! driver is a complete no-op unless
//! [`is_angular`](super::ClothRigidCouplingConfig::is_angular) is set, keeping
//! the linear goldens and the rigid-only path bit-identical.
//!
//! # Determinism
//!
//! Proxies are gathered in storage slot order, the pre-pass positions are
//! snapshotted once, the per-particle kernel is pure, and the arms are summed
//! in index order, so identical inputs produce an identical [`PhysicsWorld`]
//! state and [`AngularCouplingReport`]. Only `sqrt` is used (via `glam`), no
//! branch panics, and no path can produce a `NaN`.
//!
//! # Limitations (honest)
//!
//! The angular impulse is derived from the same single-shot Jacobi net impulse
//! the linear kernel computes, not from an iterative contact solver, and there
//! is no angular friction / rolling resistance and no coupled linear+angular
//! solve (the linear write-back and the angular write-back are sequential).
//! Central/radial contacts (e.g. a particle against a sphere proxy) produce
//! zero torque by construction, which is physically correct.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The lever
//! arm `arm × impulse` torque accumulation and the `R · diag(I⁻¹) · Rᵀ`
//! body-to-world inverse-inertia rotation are textbook rigid-body mechanics.

use glam::{Mat3, Quat, Vec3};

use crate::math::scalar::Real;
use crate::soft::{couple_particle_against_body, CouplingContribution};
use crate::world::PhysicsWorld;

use super::driver::{couple_cloth_to_rigid, gather_rigid_proxies};
use super::proxy::{collider_anchor, Aabb};

/// Below this squared magnitude a net angular impulse is treated as zero and no
/// `Δω` is written, so numerical dust from a near-symmetric load does not jitter
/// a body's spin.
const ANGULAR_IMPULSE_EPS_SQ: Real = 1e-18;

/// A summary of what one [`couple_cloth_to_rigid_angular`] pass did, for tests,
/// debugging, and force-feedback readouts.
///
/// It extends the linear [`super::CouplingReport`] with the angular totals: the
/// number of bodies that received a `Δω` and the net angular impulse applied.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AngularCouplingReport {
    /// Number of rigid proxies that participated in the pass (after culling).
    pub proxy_count: usize,
    /// Number of bodies whose angular velocity was written back (non-zero net
    /// torque and a non-zero inverse inertia).
    pub applied_count: usize,
    /// Sum of the linear reaction impulses written back by the linear bridge
    /// (carried through so a caller gets both halves from one report).
    pub applied_linear_impulse: Vec3,
    /// Sum of the net angular impulses (`Σ arm × impulse`) written back onto the
    /// bodies' angular velocities.
    pub applied_angular_impulse: Vec3,
}

/// Applies a body's world-space inverse inertia to an angular impulse,
/// returning the resulting angular-velocity change `Δω`.
///
/// `inv_inertia` is the body-local principal inverse-inertia diagonal; the
/// world-space inverse inertia is `I⁻¹_world = R · diag(inv_inertia) · Rᵀ` for
/// the body orientation `R`. Rotating the impulse into the body frame
/// (`Rᵀ·L`), scaling by the diagonal, and rotating back (`R·…`) yields
/// `Δω = I⁻¹_world · L` without materialising the full matrix product.
#[must_use]
pub fn world_inverse_inertia_apply(
    orientation: Quat,
    inv_inertia: Vec3,
    angular_impulse: Vec3,
) -> Vec3 {
    let r = Mat3::from_quat(orientation);
    // Rotate the impulse into the body-local principal frame, scale by the
    // diagonal inverse inertia (component-wise), then rotate back to world.
    let local = r.transpose() * angular_impulse;
    let local_omega = local * inv_inertia;
    r * local_omega
}

/// Runs one substep of two-way cloth↔rigid coupling with the angular bridge
/// layered on the linear one, writing both the linear reaction and the net
/// contact torque back onto the rigid bodies.
///
/// `positions` is the soft body's particle position column (corrected in place
/// by the linear bridge) and `inverse_masses` the index-aligned inverse-mass
/// column; `dt` is the substep. The function:
///
/// 1. Returns an empty [`AngularCouplingReport`] immediately when the angular
///    bridge is not armed
///    ([`is_angular`](super::ClothRigidCouplingConfig::is_angular) is `false`),
///    `dt <= 0`, the particle arrays are empty, or no proxy overlaps the soft
///    body — a complete no-op that leaves the linear path and rigid-only path
///    bit-identical.
/// 2. Snapshots the pre-pass positions, runs the untouched linear bridge
///    ([`couple_cloth_to_rigid`]), then re-derives each movable proxy's net
///    contact torque from the pre-pass contacts and applies the resulting `Δω`.
pub fn couple_cloth_to_rigid_angular(
    world: &mut PhysicsWorld,
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    dt: Real,
) -> AngularCouplingReport {
    if !world.cloth_coupling.is_angular() || dt <= 0.0 || positions.is_empty() {
        return AngularCouplingReport::default();
    }
    let Some(soft_aabb) = Aabb::of_points(positions) else {
        return AngularCouplingReport::default();
    };
    // Snapshot the proxies (pre-pass colliders + handles) and the pre-pass
    // particle positions: the lever arms must be measured against the contacts
    // as they were *before* the linear kernel corrected the particles and
    // translated the bodies.
    let proxies = gather_rigid_proxies(world, &soft_aabb);
    if proxies.is_empty() {
        return AngularCouplingReport::default();
    }
    let originals: Vec<Vec3> = positions.to_vec();

    // Run the untouched linear bridge: particle corrections, body translation,
    // and linear velocity write-back all stay bit-identical to the linear path.
    let linear = couple_cloth_to_rigid(world, positions, inverse_masses, dt);

    let mut report = AngularCouplingReport {
        proxy_count: proxies.len(),
        applied_linear_impulse: linear.applied_impulse,
        ..AngularCouplingReport::default()
    };

    let count = originals.len().min(inverse_masses.len());
    for proxy in &proxies {
        let w_body = proxy.body.inverse_mass;
        if w_body <= 0.0 {
            // Immovable proxy: the linear bridge never translated it and it has
            // no finite inertia to spin, so there is no angular write-back.
            continue;
        }
        let Some(mass) = world.bodies.mass_properties(proxy.handle) else {
            continue;
        };
        if mass.inv_inertia == Vec3::ZERO {
            // Infinite rotational inertia (or an explicitly non-rotatable body):
            // no torque can change its spin.
            continue;
        }

        // Center of mass is the body position, recovered from the proxy's
        // pre-pass collider anchor so the arms use the pre-pass pose.
        let com = collider_anchor(proxy.body.collider);
        let mut angular_impulse = Vec3::ZERO;
        for i in 0..count {
            // Re-evaluate the pre-pass contact to recover the contact point and
            // the reaction impulse the net linear reduction discarded.
            let contribution = couple_particle_against_body(
                originals[i],
                inverse_masses[i],
                proxy.body.collider,
                w_body,
                dt,
            );
            if contribution == CouplingContribution::ZERO {
                continue;
            }
            // Contact point on the proxy surface and its lever arm about the COM.
            let contact = proxy.body.collider.project(originals[i]);
            let arm = contact - com;
            angular_impulse += arm.cross(contribution.impulse);
        }

        if angular_impulse.length_squared() <= ANGULAR_IMPULSE_EPS_SQ {
            // Near-symmetric load: the per-contact torques cancel, so leave the
            // spin untouched instead of writing numerical dust.
            continue;
        }

        let orientation = world.bodies.orientation(proxy.handle).unwrap_or_default();
        let delta_omega =
            world_inverse_inertia_apply(orientation, mass.inv_inertia, angular_impulse);
        if let Some(w0) = world.bodies.angular_velocity(proxy.handle) {
            world
                .bodies
                .set_angular_velocity(proxy.handle, w0 + delta_omega);
        }
        report.applied_count += 1;
        report.applied_angular_impulse += angular_impulse;
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ColliderShape;
    use crate::soft::rigid_coupling::ClothRigidCouplingConfig;
    use crate::state::body::{BodyDesc, BodyKind, MassProperties};
    use crate::state::handle::BodyHandle;
    use crate::world::PhysicsWorld;
    use crate::WorldConfig;

    const DT: Real = 1.0 / 60.0;

    /// Spawns a dynamic cuboid prop at `position` with the given inverse mass
    /// and (body-local principal) inverse inertia and identity orientation.
    fn spawn_box(
        world: &mut PhysicsWorld,
        position: Vec3,
        half_extents: Vec3,
        inv_mass: Real,
        inv_inertia: Vec3,
    ) -> BodyHandle {
        let shape = world.shapes.insert(ColliderShape::Cuboid { half_extents });
        let desc = BodyDesc {
            kind: BodyKind::Dynamic,
            position,
            collider: Some(shape),
            mass_properties: MassProperties {
                inv_mass,
                inv_inertia,
            },
            ..BodyDesc::default()
        };
        world.spawn(desc)
    }

    #[test]
    fn off_center_load_spins_about_the_right_axis() {
        // A light box prop centered at (0, 0.2, 0) with half-extents
        // (1.0, 0.3, 1.0) spans y in [-0.1, 0.5]. A single free particle at
        // (0.5, 0, 0) sits inside the box; its least-penetration face is the
        // bottom (-Y) face, so the particle is pushed down to y = -0.1 and the
        // reaction impulse on the body points +Y. The lever arm from the COM is
        // (0.5, -0.3, 0), so arm × impulse points +Z: the box should spin about
        // +Z and not about X or Y.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active_angular();
        let handle = spawn_box(
            &mut world,
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(1.0, 0.3, 1.0),
            50.0,
            Vec3::ONE,
        );
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];

        let report = couple_cloth_to_rigid_angular(&mut world, &mut positions, &inverse_masses, DT);

        // Angular response: spins about +Z only.
        let omega = world.bodies.angular_velocity(handle).expect("body exists");
        assert!(omega.z > 1e-6, "expected +Z spin, got {omega:?}");
        assert!(omega.x.abs() < 1e-6, "no X spin, got {}", omega.x);
        assert!(omega.y.abs() < 1e-6, "no Y spin, got {}", omega.y);
        assert!(report.applied_count == 1, "one body spun");
        assert!(report.applied_angular_impulse.z > 1e-6);

        // Linear superset still holds: the particle is dented down (y < 0) and
        // the light body is lifted up (y > its start 0.2).
        assert!(
            positions[0].y < 0.0,
            "particle dented, got {}",
            positions[0].y
        );
        let body_y = world.bodies.position(handle).expect("body exists").y;
        assert!(body_y > 0.2, "body lifted, got {body_y}");
    }

    #[test]
    fn symmetric_load_produces_no_net_spin() {
        // Two free particles symmetric about the box's YZ plane at (±0.5, 0, 0)
        // each push the bottom face up; their +Z and -Z torques cancel, so the
        // net angular impulse — and therefore the spin — is ~zero.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active_angular();
        let handle = spawn_box(
            &mut world,
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(1.0, 0.3, 1.0),
            50.0,
            Vec3::ONE,
        );
        let mut positions = [Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];

        couple_cloth_to_rigid_angular(&mut world, &mut positions, &inverse_masses, DT);

        let omega = world.bodies.angular_velocity(handle).expect("body exists");
        assert!(omega.length() < 1e-6, "expected ~zero spin, got {omega:?}");
    }

    #[test]
    fn is_deterministic() {
        let run = || {
            let mut world = PhysicsWorld::new(WorldConfig::default());
            world.cloth_coupling = ClothRigidCouplingConfig::active_angular();
            let handle = spawn_box(
                &mut world,
                Vec3::new(0.0, 0.2, 0.0),
                Vec3::new(1.0, 0.3, 1.0),
                50.0,
                Vec3::ONE,
            );
            let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
            let inverse_masses = [1.0];
            let report =
                couple_cloth_to_rigid_angular(&mut world, &mut positions, &inverse_masses, DT);
            let omega = world.bodies.angular_velocity(handle).expect("body exists");
            (omega, report)
        };
        let (omega_a, report_a) = run();
        let (omega_b, report_b) = run();
        assert_eq!(omega_a, omega_b, "angular velocity must be deterministic");
        assert_eq!(report_a, report_b, "report must be deterministic");
    }

    #[test]
    fn flag_off_is_a_noop() {
        // `active` enables the linear bridge but leaves the angular flag off, so
        // the angular driver must be a complete no-op: no spin, default report.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active();
        let handle = spawn_box(
            &mut world,
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(1.0, 0.3, 1.0),
            50.0,
            Vec3::ONE,
        );
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let before = positions;
        let inverse_masses = [1.0];

        let report = couple_cloth_to_rigid_angular(&mut world, &mut positions, &inverse_masses, DT);

        assert_eq!(report, AngularCouplingReport::default());
        assert_eq!(
            world.bodies.angular_velocity(handle),
            Some(Vec3::ZERO),
            "no spin when angular flag off"
        );
        assert_eq!(positions, before, "positions untouched when angular off");
    }

    #[test]
    fn zero_inverse_inertia_body_is_not_spun() {
        // A box with zero inverse inertia (infinite rotational inertia) still
        // couples linearly but can never be spun.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active_angular();
        let handle = spawn_box(
            &mut world,
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(1.0, 0.3, 1.0),
            50.0,
            Vec3::ZERO,
        );
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];

        let report = couple_cloth_to_rigid_angular(&mut world, &mut positions, &inverse_masses, DT);

        assert_eq!(
            world.bodies.angular_velocity(handle),
            Some(Vec3::ZERO),
            "zero inverse inertia body is never spun"
        );
        assert_eq!(report.applied_count, 0);
        assert_eq!(report.applied_angular_impulse, Vec3::ZERO);
        // The linear reaction is still carried through the report.
        assert!(report.applied_linear_impulse.y > 0.0);
    }

    #[test]
    fn world_inverse_inertia_identity_orientation_is_componentwise() {
        // With identity orientation, I⁻¹_world is just the diagonal, so Δω is a
        // component-wise product of the inverse inertia and the impulse.
        let inv_inertia = Vec3::new(2.0, 0.5, 4.0);
        let impulse = Vec3::new(1.0, 3.0, -2.0);
        let got = world_inverse_inertia_apply(Quat::IDENTITY, inv_inertia, impulse);
        assert!((got - inv_inertia * impulse).length() < 1e-6, "got {got:?}");
    }

    #[test]
    fn world_inverse_inertia_rotates_axes() {
        // A 90-degree yaw maps body-local X to world -Z (glam right-handed), so
        // a spin-rich inverse inertia on local X applied to a world-X impulse
        // produces a Δω along world ... let's just assert the magnitude is
        // preserved for an isotropic inertia and the vector rotates for an
        // anisotropic one.
        let inv_inertia = Vec3::new(5.0, 1.0, 1.0);
        let orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let impulse = Vec3::new(0.0, 0.0, 1.0);
        // World +Z maps back to body-local +X under Rᵀ, picks up the large 5.0
        // inverse inertia, and rotates back to world +Z scaled by 5.
        let got = world_inverse_inertia_apply(orientation, inv_inertia, impulse);
        assert!(
            (got - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-5,
            "got {got:?}"
        );
    }
}
