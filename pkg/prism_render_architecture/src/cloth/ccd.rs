//! Continuous collision detection (CCD) for cloth particles.
//!
//! The stateless [`dynamics`](super::dynamics) solver only projects particles
//! out of colliders at their *end-of-step* position. For a thin garment moving
//! fast against a thin collider that is not enough: a particle can start in
//! front of a wall and finish behind it in a single substep, tunnelling through
//! without the projection ever seeing an overlap. This module closes that gap
//! by sweeping the segment `prev -> curr` against each analytic body collider
//! and solving for the earliest time of impact (TOI) along the segment, then
//! snapping the particle to the surface with a skin offset and reflecting its
//! normal velocity by a restitution coefficient.
//!
//! The authoritative closed-form TOI solvers and the sweep driver live in
//! [`prism_physics_core`]; this module is a thin render-side façade that keeps
//! the render particle layout and the public [`CcdParams`] API, converts
//! through [`physics_bridge`](super::physics_bridge), and projects through the
//! single physics-engine implementation so there is exactly one copy of the
//! CCD math in the engine.

use alloc::vec::Vec;

use super::collision::{to_physics_collider, BodyCollider};
use super::{physics_bridge, ClothParticle, Vec3};

use prism_physics_core::soft::collision as physics_collision;

/// Tuning for the continuous-collision sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcdParams {
    /// How far outside the collider surface (along the outward normal) a
    /// particle is placed after a hit, so the next substep starts strictly
    /// outside and does not immediately re-penetrate. Non-negative; a value of
    /// zero snaps exactly to the surface.
    pub skin: f32,
    /// Normal restitution in `[0, 1]`: `0` is a fully inelastic stop (the
    /// inbound normal velocity is cancelled) and `1` is a perfect bounce (the
    /// normal velocity is mirrored). Values are clamped into range.
    pub restitution: f32,
    /// Master switch; when `false`, [`resolve_ccd`] is a no-op so callers can
    /// disable the sweep without restructuring the pipeline.
    pub enabled: bool,
}

impl Default for CcdParams {
    /// A conservative default: a small skin, no bounce, sweep enabled.
    fn default() -> Self {
        Self {
            skin: 1e-3,
            restitution: 0.0,
            enabled: true,
        }
    }
}

impl CcdParams {
    /// Returns a copy with `skin` forced non-negative, `restitution` clamped to
    /// `[0, 1]`, and any `NaN` replaced by a safe value, so a mis-authored asset
    /// can never inject a `NaN` or a negative skin into the sweep.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let skin = if self.skin.is_nan() || self.skin < 0.0 {
            0.0
        } else {
            self.skin
        };
        let restitution = if self.restitution.is_nan() {
            0.0
        } else {
            self.restitution.clamp(0.0, 1.0)
        };
        Self {
            skin,
            restitution,
            enabled: self.enabled,
        }
    }
}

/// Converts render [`CcdParams`] into the physics-engine params the
/// single-source sweep consumes. The fields line up exactly; the physics
/// resolver sanitizes internally, so this is a lossless field copy.
#[must_use]
fn to_physics_params(params: CcdParams) -> physics_collision::CcdParams {
    physics_collision::CcdParams {
        skin: params.skin,
        restitution: params.restitution,
        enabled: params.enabled,
    }
}

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` is on or inside the sphere `(center, radius)`, or `None` when
/// the swept segment never reaches the sphere. Delegates to the physics-engine
/// closed form.
#[must_use]
pub fn sphere_toi(prev: Vec3, curr: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    physics_collision::sphere_toi(
        physics_bridge::to_glam(prev),
        physics_bridge::to_glam(curr),
        physics_bridge::to_glam(center),
        radius,
    )
}

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` crosses into the infeasible side of the half-space
/// `normal.dot(x) >= offset`, or `None` when it stays in front for the whole
/// segment. Delegates to the physics-engine closed form.
#[must_use]
pub fn half_space_toi(prev: Vec3, curr: Vec3, normal: Vec3, offset: f32) -> Option<f32> {
    physics_collision::half_space_toi(
        physics_bridge::to_glam(prev),
        physics_bridge::to_glam(curr),
        physics_bridge::to_glam(normal),
        offset,
    )
}

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` enters the capsule (segment `p0 -> p1`, `radius`), or `None`
/// when the swept segment never reaches it. Delegates to the physics-engine
/// closed form (cylinder slab unioned with the two end-cap spheres).
#[must_use]
pub fn capsule_toi(prev: Vec3, curr: Vec3, p0: Vec3, p1: Vec3, radius: f32) -> Option<f32> {
    physics_collision::capsule_toi(
        physics_bridge::to_glam(prev),
        physics_bridge::to_glam(curr),
        physics_bridge::to_glam(p0),
        physics_bridge::to_glam(p1),
        radius,
    )
}

/// Sweeps every free particle from its previous position to its current
/// position against every collider and resolves the earliest tunnelling hit.
///
/// For each free particle the segment `prev_positions[i] -> particles[i]` is
/// swept against all colliders; the earliest valid TOI wins. On a hit the
/// particle is placed on the collider surface plus `params.skin` along the
/// outward normal, and its normal velocity is reflected by `params.restitution`
/// (recomputed against the corrected motion using `dt`). Pinned particles, a
/// disabled sweep, an empty collider slice, and a `prev_positions` slice
/// shorter than `particles` are all handled without panicking, and a
/// (near) zero `dt` leaves velocities untouched.
///
/// After the normal velocity is reflected the particle's tangential slide
/// across the swept segment is damped by Coulomb friction against the contact
/// (Macklin et al. 2014); `friction` is the fabric's `FabricMaterial::friction`
/// coefficient, clamped to `0..=1` with a non-finite value treated as `0`, so
/// `0` reproduces the frictionless bounce exactly.
///
/// This is a thin façade over the authoritative physics-engine sweep: the
/// render particle columns are converted to structure-of-arrays, projected
/// through [`physics_collision::resolve_ccd`], then written back. Cost is
/// `O(particles * colliders)`; visiting order is deterministic.
pub fn resolve_ccd(
    particles: &mut [ClothParticle],
    prev_positions: &[Vec3],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: f32,
    friction: f32,
) {
    if !params.enabled || colliders.is_empty() {
        return;
    }
    let (mut positions, mut velocities, inverse_masses) = physics_bridge::to_soa_full(particles);
    let prev: Vec<_> = prev_positions
        .iter()
        .copied()
        .map(physics_bridge::to_glam)
        .collect();
    let physics_colliders: Vec<_> = colliders.iter().copied().map(to_physics_collider).collect();
    physics_collision::resolve_ccd(
        &mut positions,
        &prev,
        &mut velocities,
        &inverse_masses,
        &physics_colliders,
        to_physics_params(params),
        dt,
        friction,
    );
    physics_bridge::write_positions_back(particles, &positions);
    physics_bridge::write_velocities_back(particles, &velocities);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a free particle at `position` with unit inverse mass.
    fn free_particle(position: Vec3) -> ClothParticle {
        ClothParticle {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    #[test]
    fn sphere_toi_reports_entry_crossing() {
        let t = sphere_toi(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
        )
        .expect("segment crosses the sphere");
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn sphere_toi_misses_when_segment_passes_by() {
        let t = sphere_toi(
            Vec3::new(-2.0, 3.0, 0.0),
            Vec3::new(2.0, 3.0, 0.0),
            Vec3::ZERO,
            1.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn sphere_toi_zero_when_starting_inside() {
        let t = sphere_toi(Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, 1.0)
            .expect("start inside reports zero");
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn half_space_toi_reports_crossing() {
        let t = half_space_toi(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        )
        .expect("segment crosses the plane");
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn half_space_toi_none_when_moving_away() {
        let t = half_space_toi(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn capsule_toi_hits_cylindrical_side() {
        let t = capsule_toi(
            Vec3::new(-3.0, 0.0, 2.0),
            Vec3::new(3.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            1.0,
        )
        .expect("segment crosses the capsule side");
        assert!((t - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn capsule_toi_hits_end_cap() {
        let t = capsule_toi(
            Vec3::new(0.0, 0.0, 7.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            1.0,
        )
        .expect("segment crosses the end cap");
        assert!((t - 0.5).abs() < 1e-5);
    }

    #[test]
    fn resolve_ccd_prevents_tunnelling_through_a_plane() {
        let mut particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.01,
            restitution: 0.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, 0.0);
        // The particle ends in front of the plane at the skin offset, not below.
        assert!(particles[0].position.y >= 0.0);
        assert!((particles[0].position.y - 0.01).abs() < 1e-4);
        // The downward normal velocity has been cancelled (restitution 0).
        assert!(particles[0].velocity.y >= -1e-3);
    }

    #[test]
    fn resolve_ccd_bounces_with_restitution() {
        let mut particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 1.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, 0.0);
        // A perfect bounce flips the normal velocity to point away from the wall.
        assert!(particles[0].velocity.y > 0.0);
    }

    /// Fixture: a particle sweeping diagonally from `(0, 1, 0)` to `(2, -1, 0)`
    /// crosses the plane `y >= 0` at `t = 0.5`, is snapped to `(1, 0, 0)` with
    /// push-out depth `1` and a tangential slide of length `1` along `+X`; the
    /// friction-adjusted `x` is therefore `1 - min(mu, 1)`.
    fn diagonal_plane_hit(mu: f32) -> ClothParticle {
        let mut particles = [free_particle(Vec3::new(2.0, -1.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 0.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, mu);
        particles[0]
    }

    #[test]
    fn resolve_ccd_zero_friction_keeps_tangential_slide() {
        // mu = 0 must reproduce the frictionless snap: the full tangential
        // slide survives, so x stays at 1.
        let hit = diagonal_plane_hit(0.0);
        assert!(
            (hit.position.x - 1.0).abs() < 1e-6,
            "x = {}",
            hit.position.x
        );
        assert!(hit.position.y.abs() < 1e-6, "y = {}", hit.position.y);
    }

    #[test]
    fn resolve_ccd_dynamic_friction_shrinks_slide_by_mu() {
        // Dynamic regime: x = 1 - mu * push / ||slide|| = 1 - 0.5.
        let hit = diagonal_plane_hit(0.5);
        assert!(
            (hit.position.x - 0.5).abs() < 1e-6,
            "x = {}",
            hit.position.x
        );
    }

    #[test]
    fn resolve_ccd_full_friction_locks_tangential_slide() {
        // mu = 1 saturates the static cone here (mu * push == ||slide||), so the
        // whole tangential slide is cancelled and x collapses to 0.
        let hit = diagonal_plane_hit(1.0);
        assert!(hit.position.x.abs() < 1e-6, "x = {}", hit.position.x);
    }

    #[test]
    fn resolve_ccd_more_friction_slides_less() {
        // Strictly monotone: heavier friction leaves less residual tangential
        // travel along +X.
        let low = diagonal_plane_hit(0.25);
        let high = diagonal_plane_hit(0.75);
        assert!(high.position.x < low.position.x);
    }

    #[test]
    fn resolve_ccd_ignores_pinned_and_empty() {
        let mut pinned = [ClothParticle::pinned(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        resolve_ccd(
            &mut pinned,
            &prev,
            &colliders,
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert!((pinned[0].position.y - (-5.0)).abs() < 1e-6);

        let mut free = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        resolve_ccd(&mut free, &prev, &[], CcdParams::default(), 1.0 / 60.0, 0.0);
        assert!((free[0].position.y - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn resolve_ccd_is_deterministic() {
        let build = || {
            let mut p = [
                free_particle(Vec3::new(0.0, -5.0, 0.0)),
                free_particle(Vec3::new(0.5, -4.0, 0.1)),
            ];
            let prev = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.5, 2.0, 0.1)];
            let colliders = [
                BodyCollider::HalfSpace {
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    offset: 0.0,
                },
                BodyCollider::Sphere {
                    center: Vec3::new(0.5, -3.0, 0.1),
                    radius: 0.5,
                },
            ];
            resolve_ccd(
                &mut p,
                &prev,
                &colliders,
                CcdParams::default(),
                1.0 / 60.0,
                0.0,
            );
            p
        };
        let a = build();
        let b = build();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!(pa.position.distance_squared(pb.position) < 1e-12);
            assert!(pa.velocity.distance_squared(pb.velocity) < 1e-12);
        }
    }
}

