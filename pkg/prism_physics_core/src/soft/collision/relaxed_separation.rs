//! Relaxed inverse-mass-weighted pairwise separation (the strand/`TressFX`-style
//! self-collision inner kernel).
//!
//! [`resolve_self_collision`](super::resolve_self_collision) resolves every
//! penetrating pair *fully* in one pass and separates a coincident pair along a
//! fixed `+X` axis. Strand grooms instead want a *fractional* relaxation: each
//! overlapping pair is pushed apart only a `stiffness` fraction per pass so a
//! thick braid converges smoothly over several frames (design §6.2 / §8), and a
//! coincident pair (no separating direction) is left untouched rather than
//! nudged along an arbitrary axis so the result never fabricates motion.
//!
//! Those two differences — the per-pass `stiffness` fraction and the
//! coincident-pair *skip* — are the only things that distinguish the strand
//! push from the cloth push; the inverse-mass split and the `+normal` geometry
//! are shared. Rather than let the render-side hair module keep its own copy of
//! that arithmetic, the shared push-out lives here as a single reusable
//! projection and the hair module delegates to it (see
//! `prism_render_architecture::hair::physics_bridge`), keeping the strand grid
//! traversal (deterministic gather + sort, aligned to the GPU twin) on the
//! render side while the physics engine owns the per-pair math.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! inverse-mass-weighted pairwise separation is a standard position-based
//! dynamics technique; the fractional-stiffness relaxation is the one described
//! for strand self-collision by AMD `TressFX`.

use glam::Vec3;

use crate::math::scalar::Real;

/// Squared-length floor below which the pair is treated as coincident and left
/// untouched (no separating direction can be recovered).
///
/// This is intentionally far tighter than the cloth self-collision floor
/// ([`super::EPS_LEN_SQ`] = `1e-12`): the strand kernel it mirrors treats only
/// *genuinely* coincident particles as directionless, matching the render-side
/// twin bit-for-bit.
const COINCIDENT_EPS_SQ: Real = 1.0e-24;

/// Projects one penetrating particle pair apart by a `stiffness` fraction of
/// their overlap, split by inverse mass, and returns the corrected
/// `(position_a, position_b)` — or [`None`] when the pair needs no move.
///
/// Two particles collide when their centers are closer than `min_separation`.
/// The pair is pushed apart along the line joining them, weighted by inverse
/// mass so a pinned partner (`inverse_mass <= 0`) stays put and its free partner
/// absorbs the whole correction. The move is scaled by `stiffness` (the caller
/// is expected to pass a fraction in `0..=1`): `1` separates the pair fully in
/// one call, smaller values relax gradually over frames for stability.
///
/// Returns [`None`] — leaving both particles exactly where they were — when the
/// pair is already at least `min_separation` apart, is coincident (closer than
/// the square root of [`COINCIDENT_EPS_SQ`], so no separating direction exists),
/// or is both-pinned (combined inverse mass `<= 0`). No path can produce a
/// [`f32::NAN`], and only [`f32::sqrt`] is used.
///
/// The arithmetic (including the reciprocal-multiply `delta * (1 / dist)` rather
/// than `delta / dist`) is kept bit-identical to the render-side strand
/// self-collision kernel it replaces, so delegating to it never changes the
/// simulated result.
#[must_use]
pub fn project_relaxed_pair_separation(
    position_a: Vec3,
    position_b: Vec3,
    inverse_mass_a: Real,
    inverse_mass_b: Real,
    min_separation: Real,
    stiffness: Real,
) -> Option<(Vec3, Vec3)> {
    let min_separation_sq = min_separation * min_separation;
    let delta = position_a - position_b;
    let dist_sq = delta.length_squared();
    if dist_sq >= min_separation_sq || dist_sq < COINCIDENT_EPS_SQ {
        // Far enough apart, or coincident (no separating direction).
        return None;
    }

    let wa = inverse_mass_a.max(0.0);
    let wb = inverse_mass_b.max(0.0);
    let w = wa + wb;
    if w <= 0.0 {
        // Both pinned: nothing can move.
        return None;
    }

    let dist = dist_sq.sqrt();
    let overlap = min_separation - dist;
    // Reciprocal-multiply (not `delta / dist`) to stay bit-identical to the
    // render-side strand kernel this primitive replaces.
    let normal = delta * (1.0 / dist);
    let correction = normal * (overlap * stiffness);
    // Split the push by inverse mass: the lighter (freer) particle moves more; a
    // pinned partner (weight 0) does not move at all.
    let new_a = position_a + correction * (wa / w);
    let new_b = position_b - correction * (wb / w);
    Some((new_a, new_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_SEP: Real = 1.0;

    #[test]
    fn equal_free_particles_split_symmetrically_to_min_separation() {
        // Overlap 0.6 (centers 0.4 apart, min sep 1.0); full stiffness pushes
        // each the same distance so they end exactly `min_sep` apart.
        let (a, b) = project_relaxed_pair_separation(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            1.0,
            1.0,
            MIN_SEP,
            1.0,
        )
        .expect("overlapping pair separates");
        assert!((a.x - -0.3).abs() < 1.0e-6, "a.x = {}", a.x);
        assert!((b.x - 0.7).abs() < 1.0e-6, "b.x = {}", b.x);
        assert!((b - a).length() >= MIN_SEP - 1.0e-6);
    }

    #[test]
    fn pinned_partner_absorbs_whole_correction() {
        let (a, b) = project_relaxed_pair_separation(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            0.0, // a pinned
            1.0,
            MIN_SEP,
            1.0,
        )
        .expect("overlapping pair separates");
        assert!((a.x - 0.0).abs() < 1.0e-6, "pinned a moved: {}", a.x);
        assert!((b.x - 1.0).abs() < 1.0e-6, "free b.x = {}", b.x);
    }

    #[test]
    fn half_stiffness_resolves_half_the_overlap() {
        // Overlap 0.6; half stiffness moves each by 0.15, leaving separation
        // 0.4 + 0.3 = 0.7.
        let (a, b) = project_relaxed_pair_separation(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            1.0,
            1.0,
            MIN_SEP,
            0.5,
        )
        .expect("overlapping pair separates");
        assert!(((b - a).length() - 0.7).abs() < 1.0e-6);
    }

    #[test]
    fn far_pair_is_untouched() {
        assert!(project_relaxed_pair_separation(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
            1.0,
            MIN_SEP,
            1.0,
        )
        .is_none());
    }

    #[test]
    fn coincident_pair_is_skipped() {
        assert!(
            project_relaxed_pair_separation(Vec3::ZERO, Vec3::ZERO, 1.0, 1.0, MIN_SEP, 1.0,)
                .is_none()
        );
    }

    #[test]
    fn both_pinned_pair_is_skipped() {
        assert!(project_relaxed_pair_separation(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            0.0,
            0.0,
            MIN_SEP,
            1.0,
        )
        .is_none());
    }
}
