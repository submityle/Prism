//! Continuous self-collision (cloth-vs-cloth CCD) for the cloth solver.
//!
//! The discrete self-collision tier in [`super::collision`] only measures
//! interpenetration at the *end-of-step* positions: two layers closer than a
//! cloth `thickness` at the frame end are pushed apart. That is enough while
//! nothing moves more than a `thickness` per step, but a fast fold (a cracked
//! whip, a cape snapped in a gust, a skirt struck by an impact) can carry one
//! layer clean through another between substeps without the discrete pass ever
//! seeing an overlap — the classic tunnelling failure the design flags as the
//! single most important self-collision feature (design §6.2).
//!
//! This module closes that gap the same way [`super::ccd`] closes it for the
//! body proxies, but for every *pair of cloth particles*: each particle's
//! frame-start position is swept to its current position, every candidate pair
//! is solved for the earliest time of impact (the instant their swept
//! separation first reaches `thickness`), and the pair is resolved *at the TOI*
//! so the layers can never be integrated past the contact instant.
//!
//! The authoritative broad-phase spatial hash, the closed-form swept-pair TOI,
//! and the Gauss-Seidel narrow-phase resolver all live in
//! [`prism_physics_core`]. This module is a thin render-side façade that keeps
//! the render particle layout and the public [`SelfCcdParams`] API, converts
//! through [`physics_bridge`](super::physics_bridge), and projects through the
//! single physics-engine implementation so there is exactly one copy of the
//! continuous self-collision math in the engine.

use alloc::vec::Vec;

use super::{physics_bridge, ClothParticle, Vec3};

use prism_physics_core::soft::collision as physics_collision;

/// Smallest spatial-hash cell edge the broad phase will use; guards a
/// non-positive or non-finite authored `cell_size` from producing a division by
/// zero or an unbounded cell enumeration.
const MIN_CELL_SIZE: f32 = 1e-4;

/// Tuning for the continuous self-collision sweep.
///
/// Mirrors the shape of [`super::ccd::CcdParams`] so the two CCD passes tune
/// alike: a master `enabled` switch, the contact `thickness` the layers are
/// held apart by, and the normal `restitution` of a cloth-cloth impact. The
/// `cell_size` sets the broad-phase spatial-hash resolution; a value near the
/// mean particle spacing keeps buckets small without exploding the swept-box
/// cell count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfCcdParams {
    /// Spatial-hash cell edge length used to bucket swept bounding boxes.
    pub cell_size: f32,
    /// Minimum enforced separation between two cloth particles; the TOI is the
    /// instant their swept distance first reaches this value.
    pub thickness: f32,
    /// Normal restitution in `[0, 1]`: `0` is a fully inelastic stop (the
    /// inbound relative normal velocity is cancelled) and `1` is a perfect
    /// bounce (it is mirrored). Values are clamped into range.
    pub restitution: f32,
    /// Master switch; when `false`, [`resolve_self_ccd`] is a no-op so a LOD
    /// tier or budget decision can disable the sweep without restructuring the
    /// pipeline.
    pub enabled: bool,
}

impl Default for SelfCcdParams {
    /// A conservative default: sweep disabled, so a garment pays for continuous
    /// self-collision only when it opts in (the discrete self-collision tier
    /// governs otherwise), matching the `ccd_enabled == false` body default.
    fn default() -> Self {
        Self {
            cell_size: 0.1,
            thickness: 0.05,
            restitution: 0.0,
            enabled: false,
        }
    }
}

impl SelfCcdParams {
    /// Builds enabled continuous self-collision settings with no bounce.
    #[must_use]
    pub fn new(cell_size: f32, thickness: f32) -> Self {
        Self {
            cell_size,
            thickness,
            restitution: 0.0,
            enabled: true,
        }
    }

    /// Returns a copy with every scalar forced into a safe, finite range so the
    /// resolver can trust its inputs: `cell_size` is floored at
    /// [`MIN_CELL_SIZE`] (a non-finite or non-positive value falls back to twice
    /// the sanitized `thickness`, itself floored at [`MIN_CELL_SIZE`]),
    /// `thickness` is clamped non-negative, and `restitution` is clamped to
    /// `[0, 1]` with a non-finite value treated as `0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let thickness = if self.thickness.is_finite() && self.thickness > 0.0 {
            self.thickness
        } else {
            0.0
        };
        let cell_size = if self.cell_size.is_finite() && self.cell_size > MIN_CELL_SIZE {
            self.cell_size
        } else {
            (thickness * 2.0).max(MIN_CELL_SIZE)
        };
        let restitution = if self.restitution.is_finite() {
            self.restitution.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            cell_size,
            thickness,
            restitution,
            enabled: self.enabled,
        }
    }
}

/// Converts render [`SelfCcdParams`] into the physics-engine params the
/// single-source sweep consumes. The fields line up exactly; the physics
/// resolver sanitizes internally, so this is a lossless field copy.
#[must_use]
fn to_physics_params(params: SelfCcdParams) -> physics_collision::SelfCcdParams {
    physics_collision::SelfCcdParams {
        cell_size: params.cell_size,
        thickness: params.thickness,
        restitution: params.restitution,
        enabled: params.enabled,
    }
}

/// Solves the earliest time of impact for a swept pair of particles, delegating
/// to the physics-engine closed form.
///
/// Particle `a` moves along `prev_a -> curr_a` and particle `b` along
/// `prev_b -> curr_b` over the unit frame interval; the first `t` at which their
/// separation reaches `thickness` is returned. See
/// [`physics_collision::swept_pair_toi`] for the exact root selection and the
/// degenerate-input fall-backs.
#[must_use]
pub fn swept_pair_toi(
    prev_a: Vec3,
    curr_a: Vec3,
    prev_b: Vec3,
    curr_b: Vec3,
    thickness: f32,
) -> Option<f32> {
    physics_collision::swept_pair_toi(
        physics_bridge::to_glam(prev_a),
        physics_bridge::to_glam(curr_a),
        physics_bridge::to_glam(prev_b),
        physics_bridge::to_glam(curr_b),
        thickness,
    )
}

/// Resolves cloth-vs-cloth continuous self-collision, snapping tunnelling pairs
/// back to their time-of-impact contact.
///
/// For each particle the frame-start position (`prev_positions[i]`) is swept to
/// its current position (`particles[i].position`); the swept box, grown by
/// `params.thickness`, is bucketed into a uniform spatial hash. Every pair that
/// shares a cell is a candidate; each candidate's [`swept_pair_toi`] is solved,
/// and on a hit both particles are placed at their TOI positions, separated
/// symmetrically by inverse mass to restore the `thickness` gap, and their
/// inbound relative normal velocity is exchanged by a restitution impulse.
///
/// Pairs are visited in deterministic order and resolved in place
/// (Gauss-Seidel). A pinned particle (`inverse_mass <= 0`) is never written, a
/// pair of two pinned particles is skipped, a disabled or zero-`thickness` sweep
/// and a `prev_positions` slice shorter than `particles` are all no-ops, and a
/// coincident pair falls back to a fixed `+X` contact normal so the result never
/// contains `NaN`.
///
/// This is a thin façade over the authoritative physics-engine sweep: the render
/// particle columns are converted to structure-of-arrays, projected through
/// [`physics_collision::resolve_self_ccd`], then written back.
pub fn resolve_self_ccd(
    particles: &mut [ClothParticle],
    prev_positions: &[Vec3],
    params: SelfCcdParams,
    dt: f32,
) {
    // Cheap early-out mirrors the physics resolver's own guard so a disabled or
    // zero-thickness sweep (the default) never pays for the SoA conversion.
    let sanitized = params.sanitized();
    if !sanitized.enabled || sanitized.thickness <= 0.0 {
        return;
    }
    let (mut positions, mut velocities, inverse_masses) = physics_bridge::to_soa_full(particles);
    let prev: Vec<_> = prev_positions
        .iter()
        .copied()
        .map(physics_bridge::to_glam)
        .collect();
    physics_collision::resolve_self_ccd(
        &mut positions,
        &prev,
        &mut velocities,
        &inverse_masses,
        to_physics_params(params),
        dt,
    );
    physics_bridge::write_positions_back(particles, &positions);
    physics_bridge::write_velocities_back(particles, &velocities);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Builds a free particle at `position` with the given velocity and unit
    /// inverse mass.
    fn moving(position: Vec3, velocity: Vec3) -> ClothParticle {
        ClothParticle {
            position,
            velocity,
            inverse_mass: 1.0,
        }
    }

    /// Builds a pinned particle (`inverse_mass == 0`) at a fixed position.
    fn pinned(position: Vec3) -> ClothParticle {
        ClothParticle {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 0.0,
        }
    }

    /// Enabled params with a unit cell and the given thickness.
    fn params(thickness: f32) -> SelfCcdParams {
        SelfCcdParams {
            cell_size: 1.0,
            thickness,
            restitution: 0.0,
            enabled: true,
        }
    }

    #[test]
    /// A head-on approach solves the entry root of the swept quadratic.
    fn swept_toi_catches_a_head_on_approach() {
        // `a` at x=0 moving to x=2; `b` fixed at x=2. Gap starts at 2, closes to
        // thickness 0.5 at t = 1.5/2 = 0.75.
        let t = swept_pair_toi(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            0.5,
        )
        .expect("a closing pair must report a TOI");
        assert!((t - 0.75).abs() < 1e-5, "expected TOI 0.75, got {t}");
    }

    #[test]
    /// A pair that never gets within `thickness` reports no impact.
    fn swept_toi_misses_a_separating_pair() {
        // Parallel motion 3 apart, thickness 0.5: never touches.
        let t = swept_pair_toi(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
            0.5,
        );
        assert!(t.is_none(), "a non-closing pair must not report a TOI");
    }

    #[test]
    /// A pair already inside the contact band reports an immediate TOI of zero.
    fn swept_toi_reports_zero_when_already_overlapping() {
        let t = swept_pair_toi(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            0.5,
        );
        assert_eq!(t, Some(0.0));
    }

    #[test]
    /// The headline case: two particles swap sides in one step (their end
    /// positions are already separated), yet the sweep catches the crossing and
    /// clamps them to a `thickness`-separated contact instead of letting them
    /// tunnel through each other.
    fn resolves_a_full_tunnel_through() {
        let thickness = 0.4;
        // `a`: x = -1 -> +1. `b`: x = +1 -> -1. They cross at the origin; both
        // end positions are 2 apart again, so the discrete pass would miss it.
        let mut particles = vec![
            moving(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
            moving(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
        ];
        let prev = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        resolve_self_ccd(&mut particles, &prev, params(thickness), 1.0);

        let separation = particles[0].position.distance(particles[1].position);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "post-CCD separation {separation} must equal thickness {thickness}"
        );
        // The mover that started on the -x side must not have ended on the +x
        // side of its partner: the crossing was arrested.
        assert!(
            particles[0].position.x <= particles[1].position.x + 1e-4,
            "particle 0 tunnelled past particle 1"
        );
        // The inbound closing velocity must have been damped along the normal.
        let vrel_n = particles[0]
            .velocity
            .sub(particles[1].velocity)
            .dot(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            vrel_n >= -1e-4,
            "relative normal velocity {vrel_n} must not still be closing"
        );
    }

    #[test]
    /// A pinned partner absorbs none of the correction: the free particle takes
    /// the whole separation and the pinned one never moves.
    fn pinned_partner_stays_put() {
        let thickness = 0.5;
        let pin_pos = Vec3::new(0.0, 0.0, 0.0);
        // The free particle sweeps from +2 (prev) through the pin to -2 (curr).
        let mut particles = vec![
            pinned(pin_pos),
            moving(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let prev = vec![pin_pos, Vec3::new(2.0, 0.0, 0.0)];
        resolve_self_ccd(&mut particles, &prev, params(thickness), 1.0);

        assert_eq!(particles[0].position, pin_pos, "pinned partner moved");
        assert_eq!(
            particles[0].velocity,
            Vec3::ZERO,
            "pinned partner gained velocity"
        );
        let separation = particles[1].position.distance(pin_pos);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "free particle must sit exactly `thickness` from the pin, got {separation}"
        );
    }

    #[test]
    /// Two pinned particles inside the contact band are a no-op (no weight to
    /// move either).
    fn two_pinned_partners_are_a_no_op() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.1, 0.0, 0.0);
        let mut particles = vec![pinned(a), pinned(b)];
        let prev = vec![a, b];
        resolve_self_ccd(&mut particles, &prev, params(0.5), 1.0);
        assert_eq!(particles[0].position, a);
        assert_eq!(particles[1].position, b);
    }

    #[test]
    /// Coincident particles resolve along the fixed `+X` fallback normal and
    /// never produce `NaN`.
    fn coincident_pair_is_finite() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let mut particles = vec![moving(origin, Vec3::ZERO), moving(origin, Vec3::ZERO)];
        let prev = vec![origin, origin];
        resolve_self_ccd(&mut particles, &prev, params(0.5), 1.0);
        for p in &particles {
            assert!(p.position.x.is_finite());
            assert!(p.position.y.is_finite());
            assert!(p.position.z.is_finite());
        }
        // They are pushed apart along +/- X by half the thickness each.
        let separation = particles[0].position.distance(particles[1].position);
        assert!((separation - 0.5).abs() < 1e-4, "got {separation}");
    }

    #[test]
    /// A disabled sweep leaves every particle untouched.
    fn disabled_sweep_is_a_no_op() {
        let mut particles = vec![
            moving(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
            moving(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
        ];
        let prev = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let before = particles.clone();
        let mut p = params(0.4);
        p.enabled = false;
        resolve_self_ccd(&mut particles, &prev, p, 1.0);
        assert_eq!(particles, before);
    }

    #[test]
    /// A `prev_positions` slice shorter than `particles` is handled without a
    /// panic (the missing tail is simply not swept).
    fn short_prev_slice_does_not_panic() {
        let mut particles = vec![
            moving(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            moving(Vec3::new(0.1, 0.0, 0.0), Vec3::ZERO),
        ];
        let prev = vec![Vec3::new(0.0, 0.0, 0.0)]; // one short
        resolve_self_ccd(&mut particles, &prev, params(0.5), 1.0);
        // count == 1 < 2, so nothing is tested; must not panic and must not move.
        assert_eq!(particles[1].position, Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    /// The pass is deterministic: the same inputs produce bit-identical output
    /// across runs.
    fn resolution_is_deterministic() {
        let build = || {
            (
                vec![
                    moving(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
                    moving(Vec3::new(-1.0, 0.1, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
                    moving(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO),
                ],
                vec![
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.1, 0.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ],
            )
        };
        let (mut a, prev_a) = build();
        let (mut b, prev_b) = build();
        resolve_self_ccd(&mut a, &prev_a, params(0.4), 1.0);
        resolve_self_ccd(&mut b, &prev_b, params(0.4), 1.0);
        assert_eq!(a, b);
    }

    #[test]
    /// A higher restitution yields a faster rebound: the post-impact separation
    /// speed grows with the coefficient.
    fn restitution_controls_rebound_speed() {
        let build = || {
            (
                vec![
                    moving(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
                    moving(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
                ],
                vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            )
        };
        let sep_speed = |restitution: f32| -> f32 {
            let (mut particles, prev) = build();
            let mut p = params(0.4);
            p.restitution = restitution;
            resolve_self_ccd(&mut particles, &prev, p, 1.0);
            // Separation speed is the relative velocity projected onto the axis
            // joining the resolved pair (particle 0 minus particle 1); positive
            // means the layers are moving apart.
            let axis = particles[0]
                .position
                .sub(particles[1].position)
                .normalize_or_zero();
            particles[0].velocity.sub(particles[1].velocity).dot(axis)
        };
        let inelastic = sep_speed(0.0);
        let bouncy = sep_speed(1.0);
        assert!(
            bouncy > inelastic + 1e-3,
            "restitution 1 ({bouncy}) must separate faster than 0 ({inelastic})"
        );
    }
}

