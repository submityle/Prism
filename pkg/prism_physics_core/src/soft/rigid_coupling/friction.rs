//! Per-contact *Coulomb friction* driver for the cloth↔rigid two-way coupling
//! bridge.
//!
//! The shared linear kernel ([`crate::soft::resolve_two_way_coupling`]) and the
//! angular driver ([`super::couple_cloth_to_rigid_angular`]) only exchange a
//! *normal* reaction between the cloth particles and the rigid proxies: the
//! push-out and its lever-arm torque. Neither brakes the *tangential* slide
//! across the contact, so a prop resting on cloth would glide frictionlessly.
//! This module closes that gap with a real, analytic Coulomb friction pass
//! layered *on top of* the normal bridge, without touching the shared kernel or
//! its GPU twins (their parity contract stays intact):
//!
//! 1. Snapshot the pre-pass proxies and particle positions *before* the normal
//!    passes run, exactly like the angular driver, so the friction arms and
//!    contact normals are measured against the pre-pass contacts.
//! 2. Run the untouched normal passes: the angular driver when
//!    [`is_angular`](super::ClothRigidCouplingConfig::is_angular) is set (which runs
//!    the linear bridge internally), otherwise the linear bridge alone. All
//!    normal effects stay **bit-identical** to the linear/angular path.
//! 3. For every proxy, re-evaluate each particle against the proxy's **pre-pass**
//!    collider with the shared per-particle kernel
//!    ([`crate::soft::couple_particle_against_body`]) to recover the contact
//!    point, outward normal `n`, and normal reaction magnitude `|jₙ|`. The
//!    relative tangential velocity at the contact is
//!    `v_t = v_rel − (v_rel·n) n`, with
//!    `v_rel = v_particle − (v_body + ω × arm)` and the particle velocity read
//!    from the post-pass positions and the caller-supplied frame-start
//!    `prev_positions` (`v_particle = (pos − prev) / dt`).
//! 4. The tangential impulse that would arrest the slide is `jt = |v_t| · m_eff`,
//!    where `m_eff = 1 / (w_particle + w_body + (arm×t̂)·I⁻¹_world·(arm×t̂))` is
//!    the full sequential-impulse tangential effective mass (including the
//!    body's rotational term). It is clamped by the Coulomb cone: inside the
//!    static cone (`jt ≤ μ_s·|jₙ|`) the slide is fully arrested, otherwise the
//!    impulse is clamped to the dynamic value `μ_d·|jₙ|`.
//! 5. The friction impulse `jt·t̂` is written **equal and opposite** onto both
//!    sides: a position-level correction `−jt·t̂·w_particle·dt` on the particle
//!    (Jacobi: accumulated and applied once) and a `+jt·t̂` linear impulse plus
//!    `arm × (jt·t̂)` angular impulse on the body (mapped through the same
//!    world-space inverse inertia the angular driver uses).
//!
//! This is a strict superset of the normal bridge, so it is **opt-in** behind
//! [`ClothRigidCouplingConfig::friction`](super::ClothRigidCouplingConfig): the
//! driver runs only the normal passes (and returns a default-friction report)
//! unless [`is_friction`](super::ClothRigidCouplingConfig::is_friction) is set,
//! keeping the linear/angular goldens and the rigid-only path bit-identical.
//!
//! # Determinism
//!
//! Proxies are gathered in storage slot order, the pre-pass positions are
//! snapshotted once, the per-particle kernel is pure, the friction impulses are
//! summed in index order, and the particle corrections are accumulated then
//! applied once (Jacobi), so identical inputs produce an identical
//! [`PhysicsWorld`] state and [`FrictionCouplingReport`]. Only `sqrt` is used
//! (via `glam`), no branch panics, and no path can produce a `NaN` (the
//! coefficients are sanitised into `[0, 1]`, zero-length tangents and
//! degenerate contacts are skipped).
//!
//! # Limitations (honest)
//!
//! Friction is derived from a single re-evaluated pre-pass normal impulse, not
//! from a coupled normal+friction LCP, and solved in one Jacobi pass with no
//! iterative friction solver or contact graph. The particle side is
//! position-level while the body side is velocity-level (a hybrid, not a
//! unified solve). There is no anisotropic/velocity-dependent friction and no
//! rolling resistance.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Coulomb friction cone (static/dynamic coefficients) and the
//! sequential-impulse tangential effective mass (including the body rotational
//! term `(arm×t̂)·I⁻¹·(arm×t̂)`) are textbook rigid-body contact dynamics
//! (Catto, *Modeling and Solving Constraints*, GDC 2009; Macklin et al.,
//! *Unified Particle Physics*, 2014 for position-level friction); the
//! body-to-world inverse-inertia rotation `R·diag(I⁻¹)·Rᵀ` is reused from the
//! angular driver.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::{couple_particle_against_body, CouplingContribution};
use crate::world::PhysicsWorld;

use super::angular::{
    couple_cloth_to_rigid_angular, world_inverse_inertia_apply, AngularCouplingReport,
};
use super::driver::{couple_cloth_to_rigid, gather_rigid_proxies};
use super::proxy::{collider_anchor, Aabb};

/// Normal reaction magnitudes at or below this are treated as no contact, so a
/// grazing/degenerate contact cannot seed a friction cone.
const NORMAL_IMPULSE_EPS: Real = 1e-12;

/// Below this squared length the push-out has no defined direction, so no
/// contact normal is formed and the particle is skipped.
const CORRECTION_EPS_SQ: Real = 1e-18;

/// Tangential speeds at or below this are treated as no slide, so numerical
/// dust cannot seed a friction impulse.
const TANGENT_SPEED_EPS: Real = 1e-9;

/// A summary of what one [`couple_cloth_to_rigid_friction`] pass did, for tests,
/// debugging, and force-feedback readouts.
///
/// It carries the normal-pass [`AngularCouplingReport`] (which itself carries the
/// linear totals) and adds the tangential friction totals: the number of
/// contacts that produced a non-zero friction impulse, the number of bodies
/// that received a friction write-back, and the net tangential linear impulse
/// and torque applied to the bodies.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrictionCouplingReport {
    /// The report of the normal (linear + optional angular) pass that ran
    /// before friction. When the angular bridge is off this still carries the
    /// linear totals (proxy count and net linear impulse).
    pub angular: AngularCouplingReport,
    /// Number of per-particle contacts that produced a non-zero clamped
    /// friction impulse.
    pub friction_contacts: usize,
    /// Number of bodies that received a friction write-back (linear and/or
    /// angular).
    pub applied_count: usize,
    /// Sum of the net tangential (friction) linear impulses written onto the
    /// bodies.
    pub applied_friction_impulse: Vec3,
    /// Sum of the net tangential (friction) torques (`Σ arm × jt·t̂`) written
    /// onto the bodies.
    pub applied_friction_torque: Vec3,
}

/// Runs one substep of two-way cloth↔rigid coupling with the Coulomb friction
/// bridge layered on the normal (linear + optional angular) one, writing the
/// tangential reaction back onto both the particles and the rigid bodies.
///
/// `positions` is the soft body's particle position column (corrected in place
/// by the normal passes and then by friction), `prev_positions` the
/// index-aligned frame-start positions used to recover each particle's velocity
/// (`v = (pos − prev) / dt`), `inverse_masses` the index-aligned inverse-mass
/// column, and `dt` the substep. The function:
///
/// 1. Snapshots the pre-pass proxies and positions when the friction bridge is
///    armed ([`is_friction`](super::ClothRigidCouplingConfig::is_friction) is set,
///    `dt > 0`, and the particle array is non-empty).
/// 2. Runs the untouched normal passes — the angular driver when the angular
///    bridge is armed, otherwise the linear bridge alone — so the normal path
///    stays bit-identical.
/// 3. When armed, re-derives each contact's tangential slide from the pre-pass
///    contacts, clamps the friction impulse by the Coulomb cone, and writes the
///    equal-and-opposite tangential reaction onto the particles and bodies.
///
/// Returns a [`FrictionCouplingReport`] carrying both the normal-pass report and
/// the friction totals. When the friction bridge is not armed this is exactly
/// the normal path plus a default-friction report (bit-identical state).
pub fn couple_cloth_to_rigid_friction(
    world: &mut PhysicsWorld,
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    dt: Real,
) -> FrictionCouplingReport {
    let friction_armed = world.cloth_coupling.is_friction() && dt > 0.0 && !positions.is_empty();

    // Snapshot the pre-pass proxies and positions BEFORE the normal passes: the
    // friction arms and contact normals must be measured against the contacts
    // as they were *before* the normal kernels corrected the particles.
    let pre = if friction_armed {
        Aabb::of_points(positions)
            .map(|aabb| (gather_rigid_proxies(world, &aabb), positions.to_vec()))
    } else {
        None
    };

    // Run the untouched normal passes. The angular driver early-returns doing
    // NOTHING (not even the linear pass) when the angular bridge is off, so
    // branch explicitly and wrap the linear-only report when it is off.
    let angular = if world.cloth_coupling.is_angular() {
        couple_cloth_to_rigid_angular(world, positions, inverse_masses, dt)
    } else {
        let linear = couple_cloth_to_rigid(world, positions, inverse_masses, dt);
        AngularCouplingReport {
            proxy_count: linear.proxy_count,
            applied_linear_impulse: linear.applied_impulse,
            ..AngularCouplingReport::default()
        }
    };

    let mut report = FrictionCouplingReport {
        angular,
        ..FrictionCouplingReport::default()
    };

    // Not armed, no overlap, or an empty soft AABB: the normal path already ran
    // and the friction totals stay default, bit-identical to the normal path.
    let Some((proxies, originals)) = pre else {
        return report;
    };
    if proxies.is_empty() {
        return report;
    }
    let count = originals
        .len()
        .min(inverse_masses.len())
        .min(prev_positions.len());
    if count == 0 {
        return report;
    }

    let mu_static = world.cloth_coupling.friction_static();
    let mu_dynamic = world.cloth_coupling.friction_dynamic();

    // Accumulate particle corrections and apply once (Jacobi) so the friction
    // pass is order-independent across particles.
    let mut particle_delta = vec![Vec3::ZERO; positions.len()];

    for proxy in &proxies {
        let w_body = proxy.body.inverse_mass;
        let collider = proxy.body.collider;
        // Center of mass and the proxy's post-pass velocity: friction acts on
        // the relative velocity at the contact, so the body's contact-point
        // velocity uses the velocity it has after the normal passes.
        let com = collider_anchor(collider);
        let v_body = world
            .bodies
            .linear_velocity(proxy.handle)
            .unwrap_or(Vec3::ZERO);
        let omega = world
            .bodies
            .angular_velocity(proxy.handle)
            .unwrap_or(Vec3::ZERO);
        let orientation = world.bodies.orientation(proxy.handle).unwrap_or_default();
        let inv_inertia = world
            .bodies
            .mass_properties(proxy.handle)
            .map_or(Vec3::ZERO, |m| m.inv_inertia);

        let mut body_friction_impulse = Vec3::ZERO;
        let mut body_friction_torque = Vec3::ZERO;

        for i in 0..count {
            let w_particle = inverse_masses[i].max(0.0);
            // Re-evaluate the pre-pass contact to recover the normal reaction
            // the kernel reduced away, and the contact point / normal.
            let contribution =
                couple_particle_against_body(originals[i], inverse_masses[i], collider, w_body, dt);
            if contribution == CouplingContribution::ZERO {
                continue;
            }
            let jn_mag = contribution.impulse.length();
            if jn_mag <= NORMAL_IMPULSE_EPS {
                continue;
            }
            let contact = collider.project(originals[i]);
            let correction = contact - originals[i];
            if correction.length_squared() <= CORRECTION_EPS_SQ {
                continue;
            }
            let n = correction.normalize();
            let arm = contact - com;

            // Relative tangential velocity at the contact (post-pass).
            let v_particle = (positions[i] - prev_positions[i]) / dt;
            let v_contact = v_body + omega.cross(arm);
            let v_rel = v_particle - v_contact;
            let v_t = v_rel - n * v_rel.dot(n);
            let vt_len = v_t.length();
            if vt_len <= TANGENT_SPEED_EPS {
                continue;
            }
            let t_hat = v_t / vt_len;

            // Sequential-impulse tangential effective mass: particle + body
            // linear + body rotational term along the tangent.
            let r_x_t = arm.cross(t_hat);
            let angular_term =
                r_x_t.dot(world_inverse_inertia_apply(orientation, inv_inertia, r_x_t));
            let k = w_particle + w_body + angular_term;
            if k <= 0.0 {
                continue;
            }
            let jt_full = vt_len / k;
            if jt_full <= 0.0 {
                continue;
            }

            // Coulomb cone: fully arrest inside the static cone, otherwise clamp
            // to the dynamic (sliding) value.
            let jt = if jt_full <= mu_static * jn_mag {
                jt_full
            } else {
                mu_dynamic * jn_mag
            };
            if jt <= 0.0 {
                continue;
            }

            let jt_vec = t_hat * jt;
            // Friction opposes the slide on the particle and drives the body the
            // other way (equal and opposite).
            particle_delta[i] -= jt_vec * (w_particle * dt);
            body_friction_impulse += jt_vec;
            body_friction_torque += arm.cross(jt_vec);
            report.friction_contacts += 1;
        }

        // Write the net friction reaction back onto the body: a linear velocity
        // delta from the tangential impulse and an angular delta from its lever
        // arm. Immovable / non-rotatable bodies skip the respective half.
        let can_translate = w_body > 0.0;
        let can_rotate = inv_inertia != Vec3::ZERO;
        let mut applied = false;
        if can_translate
            && body_friction_impulse != Vec3::ZERO
            && let Some(v0) = world.bodies.linear_velocity(proxy.handle)
        {
            world
                .bodies
                .set_linear_velocity(proxy.handle, v0 + body_friction_impulse * w_body);
            applied = true;
        }
        if can_rotate && body_friction_torque != Vec3::ZERO {
            let delta_omega =
                world_inverse_inertia_apply(orientation, inv_inertia, body_friction_torque);
            if let Some(w0) = world.bodies.angular_velocity(proxy.handle) {
                world
                    .bodies
                    .set_angular_velocity(proxy.handle, w0 + delta_omega);
                applied = true;
            }
        }
        if applied {
            report.applied_count += 1;
            report.applied_friction_impulse += body_friction_impulse;
            report.applied_friction_torque += body_friction_torque;
        }
    }

    // Apply the accumulated tangential corrections to the particles (Jacobi).
    for (pos, delta) in positions.iter_mut().zip(particle_delta.iter()) {
        *pos += *delta;
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
    use crate::state::handle::BodyHandle;
    use crate::world::PhysicsWorld;
    use crate::WorldConfig;

    const DT: Real = 1.0 / 60.0;

    /// Box prop centered at `(0, 0.2, 0)` with half-extents `(1, 0.3, 1)`, so its
    /// bottom face is at `y = -0.1`. A free particle at `(0.5, 0, 0)` sits inside
    /// the box; its least-penetration face is the bottom `-Y` face, so the normal
    /// is `-Y`, the body reaction is `+Y`, and the lever arm about the COM is
    /// `(0.5, -0.3, 0)`.
    fn spawn_box(world: &mut PhysicsWorld, inv_mass: Real, inv_inertia: Vec3) -> BodyHandle {
        let shape = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(1.0, 0.3, 1.0),
        });
        let desc = BodyDesc {
            kind: BodyKind::Dynamic,
            position: Vec3::new(0.0, 0.2, 0.0),
            collider: Some(shape),
            mass_properties: MassProperties {
                inv_mass,
                inv_inertia,
            },
            ..BodyDesc::default()
        };
        world.spawn(desc)
    }

    /// The analytic normal reaction magnitude |jₙ| the kernel produces for the
    /// canonical box/particle contact, so the Coulomb clamp can be checked
    /// against the real value instead of a hard-coded number.
    fn canonical_jn(inv_mass: Real) -> Real {
        let collider = BodyCollider::Obb {
            center: Vec3::new(0.0, 0.2, 0.0),
            orientation: glam::Quat::IDENTITY,
            half_extents: Vec3::new(1.0, 0.3, 1.0),
        };
        couple_particle_against_body(Vec3::new(0.5, 0.0, 0.0), 1.0, collider, inv_mass, DT)
            .impulse
            .length()
    }

    #[test]
    fn tangential_load_on_rough_surface_brakes_and_reacts() {
        // A particle sliding +X on a rough (high-μ) box: friction must brake the
        // particle (−X position correction), drive the body +X linearly, and
        // spin it +Z about the contact arm, while the normal superset still
        // dents the cloth and lifts the light box.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active_friction(0.9, 0.8);
        let handle = spawn_box(&mut world, 50.0, Vec3::ONE);

        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.45, 0.0, 0.0)]; // slide +X
        let inverse_masses = [1.0];

        let report =
            couple_cloth_to_rigid_friction(&mut world, &mut positions, &prev, &inverse_masses, DT);

        // Particle braked in −X (its post-friction x dropped below the start 0.5).
        assert!(
            positions[0].x < 0.5,
            "particle not braked, x = {}",
            positions[0].x
        );
        // Normal superset: dented down and the light body lifted up.
        assert!(
            positions[0].y < 0.0,
            "cloth not dented, y = {}",
            positions[0].y
        );
        assert!(
            world.bodies.position(handle).unwrap().y > 0.2,
            "body not lifted"
        );

        // Body friction reaction: +X linear, +Z angular.
        let v = world.bodies.linear_velocity(handle).unwrap();
        let w = world.bodies.angular_velocity(handle).unwrap();
        assert!(v.x > 1e-6, "expected +X friction reaction, got {v:?}");
        assert!(w.z > 1e-6, "expected +Z friction spin, got {w:?}");
        assert!(
            w.x.abs() < 1e-6 && w.y.abs() < 1e-6,
            "spurious spin axis {w:?}"
        );

        assert_eq!(report.friction_contacts, 1);
        assert_eq!(report.applied_count, 1);
        assert!(report.applied_friction_impulse.x > 1e-6);
        assert!(report.applied_friction_torque.z > 1e-6);
    }

    #[test]
    fn zero_mu_is_frictionless() {
        // μ = 0 ⇒ the friction clamp zeroes every tangential impulse, so the
        // result is bit-identical to the plain linear path.
        let mut a = PhysicsWorld::new(WorldConfig::default());
        a.cloth_coupling = ClothRigidCouplingConfig::active_friction(0.0, 0.0);
        let ha = spawn_box(&mut a, 50.0, Vec3::ONE);
        let mut pa = [Vec3::new(0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.45, 0.0, 0.0)];
        let im = [1.0];
        let report = couple_cloth_to_rigid_friction(&mut a, &mut pa, &prev, &im, DT);

        let mut b = PhysicsWorld::new(WorldConfig::default());
        b.cloth_coupling = ClothRigidCouplingConfig::active();
        let hb = spawn_box(&mut b, 50.0, Vec3::ONE);
        let mut pb = [Vec3::new(0.5, 0.0, 0.0)];
        couple_cloth_to_rigid(&mut b, &mut pb, &im, DT);

        assert_eq!(pa, pb, "positions diverged under μ=0");
        assert_eq!(a.bodies.position(ha), b.bodies.position(hb));
        assert_eq!(a.bodies.linear_velocity(ha), b.bodies.linear_velocity(hb));
        assert_eq!(a.bodies.angular_velocity(ha), b.bodies.angular_velocity(hb));
        assert_eq!(report.friction_contacts, 0);
        assert_eq!(report.applied_count, 0);
        assert_eq!(report.applied_friction_impulse, Vec3::ZERO);
        assert_eq!(report.applied_friction_torque, Vec3::ZERO);
    }

    #[test]
    fn static_scales_with_speed_then_dynamic_clamps() {
        // Inside the static cone the friction impulse grows linearly with the
        // slide speed; past it the dynamic clamp pins it to μ_d·|jₙ| regardless
        // of speed.
        let run = |prev_x: f32| -> Vec3 {
            let mut world = PhysicsWorld::new(WorldConfig::default());
            world.cloth_coupling = ClothRigidCouplingConfig::active_friction(0.6, 0.3);
            spawn_box(&mut world, 50.0, Vec3::ONE);
            let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
            let prev = [Vec3::new(prev_x, 0.0, 0.0)];
            let im = [1.0];
            couple_cloth_to_rigid_friction(&mut world, &mut positions, &prev, &im, DT)
                .applied_friction_impulse
        };

        let static1 = run(0.49).x; // slide 0.01
        let static2 = run(0.48).x; // slide 0.02 (2×)
        let dyn_a = run(-0.5).x; //   slide 1.00 (saturated)
        let dyn_b = run(-2.0).x; //   slide 2.50 (saturated)

        // Static regime: impulse is linear in slide speed.
        assert!(static1 > 0.0 && static2 > 0.0);
        assert!(
            (static2 - 2.0 * static1).abs() < 1e-6,
            "static not linear: {static1} vs {static2}"
        );
        // Dynamic regime: clamp is speed-independent and equals μ_d·|jₙ|.
        assert!(
            (dyn_a - dyn_b).abs() < 1e-9,
            "dynamic clamp not constant: {dyn_a} vs {dyn_b}"
        );
        let jn = canonical_jn(50.0);
        assert!(
            (dyn_a - 0.3 * jn).abs() < 1e-5,
            "dynamic clamp != μ_d·jn: {dyn_a} vs {}",
            0.3 * jn
        );
        // The clamp sits above the small static values here.
        assert!(dyn_a > static2);
    }

    #[test]
    fn symmetric_load_has_no_net_tangential_reaction() {
        // Two particles sliding in mirrored +X / −X directions about the box's
        // YZ plane: their tangential impulses and lever-arm torques cancel, so
        // the net friction linear impulse and spin are ~zero.
        let mut world = PhysicsWorld::new(WorldConfig::default());
        world.cloth_coupling = ClothRigidCouplingConfig::active_friction(0.9, 0.8);
        let handle = spawn_box(&mut world, 50.0, Vec3::ONE);

        let mut positions = [Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.4, 0.0, 0.0), Vec3::new(-0.4, 0.0, 0.0)];
        let im = [1.0, 1.0];

        let report = couple_cloth_to_rigid_friction(&mut world, &mut positions, &prev, &im, DT);

        assert_eq!(
            report.friction_contacts, 2,
            "both contacts produced friction"
        );
        let w = world.bodies.angular_velocity(handle).unwrap();
        assert!(w.length() < 1e-5, "expected ~zero net spin, got {w:?}");
        assert!(
            report.applied_friction_impulse.length() < 1e-5,
            "expected ~zero net friction impulse, got {:?}",
            report.applied_friction_impulse
        );
    }

    #[test]
    fn is_deterministic() {
        let run = || {
            let mut world = PhysicsWorld::new(WorldConfig::default());
            world.cloth_coupling = ClothRigidCouplingConfig::active_friction(0.6, 0.3);
            let handle = spawn_box(&mut world, 50.0, Vec3::ONE);
            let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
            let prev = [Vec3::new(0.45, 0.0, 0.0)];
            let im = [1.0];
            let report = couple_cloth_to_rigid_friction(&mut world, &mut positions, &prev, &im, DT);
            (
                world.bodies.linear_velocity(handle).unwrap(),
                world.bodies.angular_velocity(handle).unwrap(),
                positions,
                report,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0, "linear velocity not deterministic");
        assert_eq!(a.1, b.1, "angular velocity not deterministic");
        assert_eq!(a.2, b.2, "positions not deterministic");
        assert_eq!(a.3, b.3, "report not deterministic");
    }

    #[test]
    fn flag_off_matches_angular_path() {
        // With friction off but the angular bridge on, the friction driver must
        // be a complete no-op over the angular path: identical body state and
        // positions, and a default-friction (zero) report half.
        let mut a = PhysicsWorld::new(WorldConfig::default());
        a.cloth_coupling = ClothRigidCouplingConfig::active_angular();
        let ha = spawn_box(&mut a, 50.0, Vec3::ONE);
        let mut pa = [Vec3::new(0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.45, 0.0, 0.0)];
        let im = [1.0];
        let report = couple_cloth_to_rigid_friction(&mut a, &mut pa, &prev, &im, DT);

        let mut b = PhysicsWorld::new(WorldConfig::default());
        b.cloth_coupling = ClothRigidCouplingConfig::active_angular();
        let hb = spawn_box(&mut b, 50.0, Vec3::ONE);
        let mut pb = [Vec3::new(0.5, 0.0, 0.0)];
        let angular_b = couple_cloth_to_rigid_angular(&mut b, &mut pb, &im, DT);

        assert_eq!(pa, pb, "positions diverged with friction off");
        assert_eq!(a.bodies.linear_velocity(ha), b.bodies.linear_velocity(hb));
        assert_eq!(a.bodies.angular_velocity(ha), b.bodies.angular_velocity(hb));
        assert_eq!(
            report.angular, angular_b,
            "angular half must match the angular path"
        );
        assert_eq!(report.friction_contacts, 0);
        assert_eq!(report.applied_count, 0);
        assert_eq!(report.applied_friction_impulse, Vec3::ZERO);
        assert_eq!(report.applied_friction_torque, Vec3::ZERO);
    }

    #[test]
    fn flag_off_linear_only_matches_linear_path() {
        // Linear-only config (friction + angular off) ⇒ the driver is the plain
        // linear bridge plus a zero friction report.
        let mut a = PhysicsWorld::new(WorldConfig::default());
        a.cloth_coupling = ClothRigidCouplingConfig::active();
        let ha = spawn_box(&mut a, 50.0, Vec3::ONE);
        let mut pa = [Vec3::new(0.5, 0.0, 0.0)];
        let prev = [Vec3::new(0.45, 0.0, 0.0)];
        let im = [1.0];
        let report = couple_cloth_to_rigid_friction(&mut a, &mut pa, &prev, &im, DT);

        let mut b = PhysicsWorld::new(WorldConfig::default());
        b.cloth_coupling = ClothRigidCouplingConfig::active();
        let hb = spawn_box(&mut b, 50.0, Vec3::ONE);
        let mut pb = [Vec3::new(0.5, 0.0, 0.0)];
        let linear_b = couple_cloth_to_rigid(&mut b, &mut pb, &im, DT);

        assert_eq!(pa, pb);
        assert_eq!(a.bodies.position(ha), b.bodies.position(hb));
        assert_eq!(a.bodies.linear_velocity(ha), b.bodies.linear_velocity(hb));
        assert_eq!(report.friction_contacts, 0);
        assert_eq!(
            report.angular.applied_linear_impulse,
            linear_b.applied_impulse
        );
    }
}
