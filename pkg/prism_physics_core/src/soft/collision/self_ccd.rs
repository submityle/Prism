//! Continuous self-collision (cloth-vs-cloth CCD) for the soft-body kernel.
//!
//! The discrete self-collision tier in [`super::self_collision`] only measures
//! interpenetration at the *end-of-step* positions: two layers closer than a
//! cloth `thickness` at the frame end are pushed apart. That is enough while
//! nothing moves more than a `thickness` per step, but a fast fold (a cracked
//! whip, a cape snapped in a gust, a skirt struck by an impact) can carry one
//! layer clean through another between substeps without the discrete pass ever
//! seeing an overlap — the classic tunnelling failure.
//!
//! This module closes that gap the same way [`super::ccd`] closes it for the
//! body proxies, but for every *pair of soft-body particles* rather than for
//! particle-vs-analytic-collider. Each particle's frame-start position is swept
//! to its current position, and for every candidate pair the earliest time of
//! impact (TOI) — the instant their swept separation first reaches `thickness`
//! — is solved in closed form. The pair is then resolved *at the TOI*: both
//! particles are snapped to their TOI positions, separated symmetrically by
//! inverse mass, and their inbound normal velocity is exchanged through a
//! restitution impulse. Clamping to the TOI is what defeats tunnelling: the
//! layers can never be integrated past the contact instant.
//!
//! # Broad phase
//!
//! Testing every pair is `O(n^2)`. Instead each particle is bucketed into a
//! uniform spatial hash by the integer cells its *swept* axis-aligned bounding
//! box (the box of `prev` and `curr`, expanded by `thickness` on every side)
//! overlaps. Two particles are a candidate pair only if they share a cell.
//!
//! The `thickness` expansion is what lets the broad phase test a single cell
//! per bucket instead of a 27-cell neighbourhood: if a pair reaches separation
//! `<= thickness` at some TOI, then each particle's TOI position lies in its
//! own swept box, and because the boxes are grown by `thickness` the two TOI
//! positions (which are within `thickness` of each other) both fall in the
//! intersection of the two expanded boxes, so the pair necessarily shares at
//! least one cell. A [`BTreeSet`] deduplicates pairs that co-occupy several
//! cells so each is resolved exactly once.
//!
//! # Determinism and robustness
//!
//! Everything is deterministic array-in / array-out math: a [`BTreeMap`] keyed
//! by integer cell fixes the bucket order, indices are inserted in ascending
//! order so each bucket's occupants stay sorted, the [`BTreeSet`] fixes the
//! pair order, and a coincident pair falls back to a fixed `+X` normal. Only
//! [`f32::sqrt`] is used; there are no transcendental calls. Pinned particles
//! (`inverse_mass <= 0`) are never moved, a pair of two pinned particles is
//! skipped, out-of-range or too-short inputs are handled without panicking, and
//! degenerate inputs (no relative motion, coincident particles, zero
//! `thickness`) fall back deterministically and never produce a [`f32::NAN`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! closed-form swept-pair TOI and the uniform spatial-hash broad phase are
//! standard position-based continuous-collision techniques; the restitution
//! impulse is the textbook inverse-mass-weighted normal exchange.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::cell_of;

/// Numerical floor below which the top quadratic coefficient (the squared
/// relative speed of a pair) is treated as zero, i.e. the pair has no relative
/// motion over the frame and cannot cross a `thickness` gap.
const EPS_REL_MOTION: Real = 1e-12;

/// Smallest spatial-hash cell edge the broad phase will use; guards a
/// non-positive or non-finite authored `cell_size` from producing a division by
/// zero or an unbounded cell enumeration.
const MIN_CELL_SIZE: Real = 1e-4;

/// Tuning for the continuous self-collision sweep.
///
/// Mirrors the shape of [`super::ccd::CcdParams`] so the two CCD passes tune
/// alike: a master `enabled` switch, the contact `thickness` the layers are
/// held apart by, and the normal `restitution` of a cloth-cloth impact. The
/// `cell_size` sets the broad-phase spatial-hash resolution; a value near the
/// mean particle spacing keeps buckets small without exploding the swept-box
/// cell count.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SelfCcdParams {
    /// Spatial-hash cell edge length used to bucket swept bounding boxes.
    pub cell_size: Real,
    /// Minimum enforced separation between two particles; the TOI is the
    /// instant their swept distance first reaches this value.
    pub thickness: Real,
    /// Normal restitution in `0..=1`: `0` is a fully inelastic stop (the
    /// inbound relative normal velocity is cancelled) and `1` is a perfect
    /// bounce (it is mirrored). Values are clamped into range.
    pub restitution: Real,
    /// Master switch; when `false`, [`resolve_self_ccd`] is a no-op so a LOD
    /// tier or budget decision can disable the sweep without restructuring the
    /// pipeline.
    pub enabled: bool,
}

impl Default for SelfCcdParams {
    /// A conservative default: sweep disabled, so a garment pays for continuous
    /// self-collision only when it opts in (the discrete self-collision tier
    /// governs otherwise), matching the body sweep's disabled-by-cost posture.
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
    pub fn new(cell_size: Real, thickness: Real) -> Self {
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
    /// `0..=1` with a non-finite value treated as `0`.
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

/// Solves the earliest time of impact for a swept pair of particles.
///
/// Particle `a` moves along the segment `prev_a -> curr_a` and particle `b`
/// along `prev_b -> curr_b` over the unit frame interval `t` in `0..=1`. Their
/// relative offset is `d(t) = d0 + t * dv` with `d0 = prev_a - prev_b` and
/// `dv = (curr_a - curr_b) - d0`, so the first `t` at which `|d(t)| == thickness`
/// is the smaller root of the quadratic `|d0 + t*dv|^2 == thickness^2`, i.e.
/// `a*t^2 + b*t + c == 0` with `a = dv.dv`, `b = 2*(d0.dv)`,
/// `c = d0.d0 - thickness^2`.
///
/// Returns:
/// - `Some(0.0)` when the pair already starts within `thickness` (`c <= 0`);
/// - [`None`] when there is no relative motion (`a` negligibly small) and they
///   do not start overlapping, when the discriminant is negative (they never
///   reach `thickness`), or when the entry root lies outside `0..=1`;
/// - `Some(t)` with the entry root otherwise.
///
/// Only [`f32::sqrt`] is used and every early-out is comparison-based, so the
/// result is deterministic and free of a [`f32::NAN`].
#[must_use]
pub fn swept_pair_toi(
    prev_a: Vec3,
    curr_a: Vec3,
    prev_b: Vec3,
    curr_b: Vec3,
    thickness: Real,
) -> Option<Real> {
    let d0 = prev_a - prev_b;
    let d1 = curr_a - curr_b;
    let dv = d1 - d0;
    let c = d0.dot(d0) - thickness * thickness;
    // Already within the contact band at frame start: contact is immediate.
    if c <= 0.0 {
        return Some(0.0);
    }
    let a = dv.dot(dv);
    // No (appreciable) relative motion and not already overlapping: the gap
    // never closes, so there is no crossing to catch.
    if a <= EPS_REL_MOTION {
        return None;
    }
    let b = 2.0 * d0.dot(dv);
    let disc = b * b - 4.0 * a * c;
    // The swept separation never reaches `thickness`.
    if disc < 0.0 {
        return None;
    }
    // Entry root (the smaller of the two): first moment the gap closes to
    // `thickness`. `a > 0` here, so the denominator is safe.
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// Resolves cloth-vs-cloth continuous self-collision, snapping tunnelling pairs
/// back to their time-of-impact contact.
///
/// For each particle the frame-start position (`prev_positions[i]`) is swept to
/// its current position (`positions[i]`); the swept box, grown by
/// `params.thickness`, is bucketed into a uniform spatial hash. Every pair that
/// shares a cell is a candidate; each candidate's [`swept_pair_toi`] is solved,
/// and on a hit both particles are placed at their TOI positions, separated
/// symmetrically by inverse mass to restore the `thickness` gap, and their
/// inbound relative normal velocity is exchanged by a restitution impulse
/// (velocity recovered from the TOI motion over `dt`, matching the body sweep
/// in [`super::ccd::resolve_ccd`]).
///
/// Pairs are visited in [`BTreeSet`] order and resolved in place
/// (Gauss-Seidel), so a later pair sees an earlier pair's correction; the fixed
/// order keeps the whole pass deterministic. A pinned particle
/// (`inverse_mass <= 0`) is never written, a pair of two pinned particles is
/// skipped, and a disabled sweep, a non-positive `thickness`, fewer than two
/// particles, or a `prev_positions` / `inverse_masses` slice shorter than
/// `positions` is handled without panicking. `velocities` is written only where
/// the column is long enough.
pub fn resolve_self_ccd(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[Real],
    params: SelfCcdParams,
    dt: Real,
) {
    let params = params.sanitized();
    if !params.enabled || params.thickness <= 0.0 {
        return;
    }
    let count = positions
        .len()
        .min(prev_positions.len())
        .min(inverse_masses.len());
    if count < 2 {
        return;
    }

    // Broad phase: bucket each particle's thickness-expanded swept box into
    // every integer cell it overlaps. Ascending index insertion keeps each
    // bucket sorted for deterministic pairing.
    let cell_size = params.cell_size;
    let margin = Vec3::splat(params.thickness);
    let mut buckets: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for index in 0..count {
        let prev = prev_positions[index];
        let curr = positions[index];
        let lo = prev.min(curr) - margin;
        let hi = prev.max(curr) + margin;
        let (lx, ly, lz) = cell_of(lo, cell_size);
        let (hx, hy, hz) = cell_of(hi, cell_size);
        let mut cx = lx;
        while cx <= hx {
            let mut cy = ly;
            while cy <= hy {
                let mut cz = lz;
                while cz <= hz {
                    buckets.entry((cx, cy, cz)).or_default().push(index as u32);
                    cz += 1;
                }
                cy += 1;
            }
            cx += 1;
        }
    }

    // Collect unique candidate pairs across all shared cells.
    let mut pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
    for occupants in buckets.values() {
        for slot_a in 0..occupants.len() {
            for slot_b in (slot_a + 1)..occupants.len() {
                pairs.insert((occupants[slot_a], occupants[slot_b]));
            }
        }
    }

    let inv_dt = if dt.abs() <= EPS_REL_MOTION {
        0.0
    } else {
        1.0 / dt
    };

    // Narrow phase: solve and resolve each candidate pair at its TOI.
    for &(i, j) in &pairs {
        let ia = i as usize;
        let jb = j as usize;
        let wa = inverse_masses[ia].max(0.0);
        let wb = inverse_masses[jb].max(0.0);
        let wsum = wa + wb;
        // Two pinned partners cannot move: nothing to resolve.
        if wsum <= 0.0 {
            continue;
        }
        let prev_a = prev_positions[ia];
        let prev_b = prev_positions[jb];
        let curr_a = positions[ia];
        let curr_b = positions[jb];
        let Some(t) = swept_pair_toi(prev_a, curr_a, prev_b, curr_b, params.thickness) else {
            continue;
        };

        // Positions at the time of impact.
        let a_c = prev_a + (curr_a - prev_a) * t;
        let b_c = prev_b + (curr_b - prev_b) * t;
        let delta = a_c - b_c;
        let unit = delta.normalize_or_zero();
        let normal = if unit.length_squared() > 0.0 {
            unit
        } else {
            // Coincident TOI positions: pick a fixed, deterministic axis so the
            // separation still has a direction and never yields a `NaN`.
            Vec3::new(1.0, 0.0, 0.0)
        };
        let penetration = (params.thickness - delta.length()).max(0.0);
        let inv_wsum = 1.0 / wsum;

        // Snap both particles to the TOI contact, then push symmetrically apart
        // by inverse mass to restore the `thickness` gap. Clamping to the TOI is
        // the anti-tunnelling guarantee; a pinned partner (weight `0`) keeps its
        // TOI position, which equals its frame-start position.
        let pos_a = a_c + normal * (wa * inv_wsum * penetration);
        let pos_b = b_c - normal * (wb * inv_wsum * penetration);
        if wa > 0.0 {
            positions[ia] = pos_a;
        }
        if wb > 0.0 {
            positions[jb] = pos_b;
        }

        // Exchange the inbound relative normal velocity with a restitution
        // impulse. Velocity is recovered from the TOI motion over `dt` (the
        // approach velocity), so `vrel_n < 0` means the layers were closing and
        // an impulse `j = -(1 + e) * vrel_n / (wa + wb)` leaves them separating
        // at `-e` times the inbound speed.
        let va_in = (a_c - prev_a) * inv_dt;
        let vb_in = (b_c - prev_b) * inv_dt;
        let vrel_n = (va_in - vb_in).dot(normal);
        if vrel_n < 0.0 {
            let impulse = -(1.0 + params.restitution) * vrel_n * inv_wsum;
            if wa > 0.0 && ia < velocities.len() {
                velocities[ia] = va_in + normal * (wa * impulse);
            }
            if wb > 0.0 && jb < velocities.len() {
                velocities[jb] = vb_in - normal * (wb * impulse);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enabled, no-bounce settings for a given `thickness`.
    fn params(thickness: Real) -> SelfCcdParams {
        SelfCcdParams::new(0.2, thickness)
    }

    /// Resolves `positions`/`prev` with unit-mass free particles unless an
    /// inverse-mass override is supplied.
    fn run(
        positions: &mut [Vec3],
        prev: &[Vec3],
        velocities: &mut [Vec3],
        inverse_masses: &[Real],
        p: SelfCcdParams,
        dt: Real,
    ) {
        resolve_self_ccd(positions, prev, velocities, inverse_masses, p, dt);
    }

    #[test]
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
    fn swept_toi_catches_crossing() {
        // `a`: -1 -> +1, `b`: +1 -> -1 on the X axis cross at the origin; with
        // thickness 0.4 they reach the band before the midpoint.
        let t = swept_pair_toi(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.4,
        )
        .expect("the crossing pair reaches thickness");
        assert!(t > 0.0 && t < 0.5, "t = {t}");
    }

    #[test]
    fn resolves_a_full_tunnel_through() {
        let thickness = 0.4;
        // `a`: x = -1 -> +1. `b`: x = +1 -> -1. They cross at the origin; both
        // end positions are 2 apart again, so the discrete pass would miss it.
        let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let mut velocities = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            params(thickness),
            1.0,
        );

        let separation = positions[0].distance(positions[1]);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "post-CCD separation {separation} must equal thickness {thickness}"
        );
        // The mover that started on the -x side must not have ended on the +x
        // side of its partner: the crossing was arrested.
        assert!(
            positions[0].x <= positions[1].x + 1e-4,
            "particle 0 tunnelled past particle 1"
        );
        // The inbound closing velocity must have been damped along the normal.
        let vrel_n = (velocities[0] - velocities[1]).dot(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            vrel_n >= -1e-4,
            "relative normal velocity {vrel_n} must not still be closing"
        );
    }

    #[test]
    fn pinned_partner_stays_put() {
        let thickness = 0.5;
        let pin_pos = Vec3::new(0.0, 0.0, 0.0);
        // The free particle sweeps from +2 (prev) through the pin to -2 (curr).
        let mut positions = [pin_pos, Vec3::new(-2.0, 0.0, 0.0)];
        let prev = [pin_pos, Vec3::new(2.0, 0.0, 0.0)];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [0.0, 1.0];
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            params(thickness),
            1.0,
        );

        assert_eq!(positions[0], pin_pos, "pinned partner moved");
        assert_eq!(velocities[0], Vec3::ZERO, "pinned partner gained velocity");
        let separation = positions[1].distance(pin_pos);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "free particle must sit exactly `thickness` from the pin, got {separation}"
        );
    }

    #[test]
    fn two_pinned_partners_are_a_no_op() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.1, 0.0, 0.0);
        let mut positions = [a, b];
        let prev = [a, b];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [0.0, 0.0];
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            params(0.5),
            1.0,
        );
        assert_eq!(positions[0], a);
        assert_eq!(positions[1], b);
    }

    #[test]
    fn coincident_pair_is_finite() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let mut positions = [origin, origin];
        let prev = [origin, origin];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            params(0.5),
            1.0,
        );
        for p in &positions {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        // They are pushed apart along +/- X by half the thickness each.
        let separation = positions[0].distance(positions[1]);
        assert!((separation - 0.5).abs() < 1e-4, "got {separation}");
    }

    #[test]
    fn disabled_sweep_is_a_no_op() {
        let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let before = positions;
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        let mut p = params(0.4);
        p.enabled = false;
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            p,
            1.0,
        );
        assert_eq!(positions, before);
    }

    #[test]
    fn short_prev_slice_does_not_panic() {
        let mut positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.0, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)]; // one short
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        run(
            &mut positions,
            &prev,
            &mut velocities,
            &inverse_masses,
            params(0.5),
            1.0,
        );
        // count == 1 < 2, so nothing is tested; must not panic and must not move.
        assert_eq!(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn resolution_is_deterministic() {
        let build = || {
            (
                [
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-1.0, 0.1, 0.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ],
                [
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.1, 0.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ],
            )
        };
        let masses = [1.0, 1.0, 1.0];
        let (mut pa, prev_a) = build();
        let (mut pb, prev_b) = build();
        let mut va = [Vec3::ZERO; 3];
        let mut vb = [Vec3::ZERO; 3];
        run(&mut pa, &prev_a, &mut va, &masses, params(0.4), 1.0);
        run(&mut pb, &prev_b, &mut vb, &masses, params(0.4), 1.0);
        assert_eq!(pa, pb);
        assert_eq!(va, vb);
    }

    #[test]
    fn restitution_controls_rebound_speed() {
        let sep_speed = |restitution: Real| -> Real {
            let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
            let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
            let mut velocities = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
            let inverse_masses = [1.0, 1.0];
            let mut p = params(0.4);
            p.restitution = restitution;
            run(
                &mut positions,
                &prev,
                &mut velocities,
                &inverse_masses,
                p,
                1.0,
            );
            // Separation speed is the relative velocity projected onto the axis
            // joining the resolved pair (particle 0 minus particle 1); positive
            // means the layers are moving apart.
            let axis = (positions[0] - positions[1]).normalize_or_zero();
            (velocities[0] - velocities[1]).dot(axis)
        };
        let inelastic = sep_speed(0.0);
        let bouncy = sep_speed(1.0);
        assert!(
            bouncy > inelastic + 1e-3,
            "restitution 1 ({bouncy}) must separate faster than 0 ({inelastic})"
        );
    }
}
