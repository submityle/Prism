//! Micro-scale occlusion from sub-texel normal statistics — CPU golden.
//!
//! A single shaded texel averages a whole patch of micro-geometry.  When those
//! sub-texel normals disagree, two things happen at once: the specular lobe
//! *widens* (more apparent roughness, the classic normal-map aliasing Toksvig
//! fixed by folding the normal-map variance into the gloss term), and the tiny
//! creases between the facets shadow one another, darkening the ambient term
//! (*cavity* ambient occlusion).  This module is the backend-neutral reference
//! for both effects, derived purely from the statistics of the sub-texel
//! normals.
//!
//! The whole derivation hangs off one scalar: the **mean-resultant length**
//! `r = |mean(N)|` of the (unit) sub-texel normals.  If every normal points the
//! same way the average is itself a unit vector, so `r = 1`; the more they
//! spread the shorter the average, so `r -> 0`.  Toksvig's observation (NVIDIA,
//! *Mipmapping Normal Maps*, 2005) is that this single length encodes the
//! angular variance of the patch:
//!
//! ```text
//! sigma^2  ~=  (1 - r) / r
//! ```
//!
//! which this module exposes as [`toksvig_variance`].  `r -> 1` gives
//! `sigma^2 -> 0` (a flat, mirror-tight patch); `r -> 0` gives `sigma^2 -> inf`
//! (a fully scrambled patch).  From there:
//!
//! * [`micro_bent_normal`] / [`micro_bent_normal_from_mean`] fit a
//!   [`MicroBentNormal`] — the average micro-direction, the half-angle of the
//!   still-visible cone, the Toksvig *effective roughness*, and the cavity AO.
//! * [`toksvig_variance`], [`effective_roughness`], and [`cavity_ao`] are the
//!   underlying closed forms, exposed so callers (and tests) can reuse or
//!   cross-check them directly.
//! * [`merge_micro_macro`] folds the micro cone together with a macro-scale
//!   bent-normal cone (e.g. the GTAO / world-space result) into a single
//!   equivalent [`MergedCone`], widening the aperture when the two directions
//!   disagree and multiplying their occlusion.
//!
//! # Conventions
//! * Directions are right-handed unit `Vec3`s.  Stored directions are always
//!   normalised to within `f32` round-off.  The micro-geometry "up" fallback is
//!   the tangent-space geometric normal `+Z`; the merge fallback is the macro
//!   direction it was handed.
//! * The mean-resultant length `r` is a scalar in `[0, 1]`; `1` means every
//!   sub-texel normal agrees, `0` means they fully cancel.
//! * A cone *aperture* is its half-angle in radians, in `[0, PI]`.  It is tied
//!   to `r` by the uniform-cone centroid identity `r = (1 + cos(aperture)) / 2`,
//!   i.e. `aperture = acos(2*r - 1)`, matching the engine's bent-normal
//!   convention: `r -> 1` closes the cone (narrow, all-visible) and `r -> 0`
//!   opens it (wide, fully scattered).
//! * Roughness, visibility (AO), and merged energy are all in `[0, 1]`.
//! * Degenerate inputs never produce `NaN`: an empty normal set, a zero-length
//!   mean, or non-finite components fall back to a *flat* patch (direction
//!   `+Z`, closed aperture, full visibility, base roughness).  Non-finite
//!   scalars are sanitised before use.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no `unsafe`, no allocation, and no global state.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Smallest mean-resultant length treated as non-degenerate.
///
/// Clamping `r` away from `0` keeps the Toksvig variance `(1 - r) / r` finite
/// (it tends to a very large but representable number as `r -> 0`) so the
/// derived quantities stay bounded instead of overflowing to infinity.
const MIN_RESULTANT: f32 = 1.0e-6;

/// The micro-scale bent normal fitted to a texel's sub-texel normal statistics.
///
/// Bundles the four quantities a shader needs to react to sub-texel detail: the
/// average [`direction`](Self::direction), the half-angle
/// [`aperture`](Self::aperture) of the cone the micro-normals still leave
/// visible, the Toksvig [`roughness`](Self::roughness) the widened lobe behaves
/// like, and the scalar cavity [`visibility`](Self::visibility).  The field
/// layout mirrors the packed `f32`s the GPU twin consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicroBentNormal {
    /// Unit-length average sub-texel normal (the micro bent normal).
    pub direction: Vec3,
    /// Visible-cone half-angle in radians, in `[0, PI]` (`0` tight, `PI` wide).
    pub aperture: f32,
    /// Toksvig effective roughness in `[0, 1]` (`0` mirror, `1` fully rough).
    pub roughness: f32,
    /// Scalar cavity ambient occlusion in `[0, 1]` (`1` fully visible).
    pub visibility: f32,
}

impl Default for MicroBentNormal {
    fn default() -> Self {
        Self::FLAT
    }
}

impl MicroBentNormal {
    /// A perfectly flat micro-patch: `+Z` axis, closed cone, mirror roughness,
    /// full visibility.
    ///
    /// This is the `r = 1` limit and the fallback used when there is no usable
    /// normal statistic to fit (an empty set or fully degenerate input).
    pub const FLAT: Self = Self {
        direction: Vec3::Z,
        aperture: 0.0,
        roughness: 0.0,
        visibility: 1.0,
    };
}

/// A single cone of visible directions: a direction, a half-angle, and the
/// scalar visibility it carries.
///
/// Produced by [`merge_micro_macro`] as the equivalent of a micro cone folded
/// together with a macro-scale bent-normal cone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergedCone {
    /// Unit-length cone axis (the combined bent normal).
    pub direction: Vec3,
    /// Cone half-angle in radians, in `[0, PI]`.
    pub aperture: f32,
    /// Combined scalar visibility / AO in `[0, 1]`.
    pub visibility: f32,
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero or
/// non-finite) input so the caller never divides by zero or propagates a `NaN`.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Returns `x` if it is finite, otherwise `fallback`.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Maps a mean-resultant length `r` to a cone half-angle via the uniform-cone
/// centroid identity `r = (1 + cos(aperture)) / 2`, i.e. `acos(2*r - 1)`.
///
/// `r` is clamped to `[0, 1]` first, so the result is always in `[0, PI]`
/// (`r = 1 -> 0`, `r = 0 -> PI`).
#[inline]
fn resultant_to_aperture(r: f32) -> f32 {
    let r = finite_or(r, 1.0).clamp(0.0, 1.0);
    ops::acos((2.0 * r - 1.0).clamp(-1.0, 1.0))
}

/// Inverse of [`resultant_to_aperture`]: the mean-resultant length of a uniform
/// cone of half-angle `aperture`, `(1 + cos(aperture)) / 2`, in `[0, 1]`.
#[inline]
fn aperture_to_resultant(aperture: f32) -> f32 {
    let a = finite_or(aperture, 0.0).clamp(0.0, PI);
    (0.5 * (1.0 + ops::cos(a))).clamp(0.0, 1.0)
}

/// The Toksvig normal-distribution variance `sigma^2 ~= (1 - r) / r`.
///
/// `r` is the mean-resultant length `|mean(N)|` of the sub-texel normals,
/// clamped to `[MIN_RESULTANT, 1]` so the quotient stays finite.  A perfectly
/// coherent patch (`r = 1`) has zero variance; a scrambled patch (`r -> 0`)
/// has a very large (but finite) variance.  Non-finite input is treated as the
/// coherent limit and returns `0`.
#[inline]
pub fn toksvig_variance(mean_length: f32) -> f32 {
    let r = finite_or(mean_length, 1.0).clamp(MIN_RESULTANT, 1.0);
    (1.0 - r) / r
}

/// Combines a base material roughness with the Toksvig sub-texel variance into
/// an effective roughness in `[0, 1]`.
///
/// The sub-texel angular variance adds to the lobe's spread, so this follows
/// the standard variance-combining form `alpha_eff^2 = alpha^2 + 2*sigma^2`
/// (clamped to `1`), with `sigma^2` from [`toksvig_variance`].  At `r = 1` the
/// variance vanishes and the result is just the (clamped) base roughness; as
/// `r -> 0` the lobe saturates to fully rough.  Non-finite inputs are sanitised
/// (`base_roughness -> 0`, `mean_length -> 1`).
#[inline]
pub fn effective_roughness(base_roughness: f32, mean_length: f32) -> f32 {
    let a = finite_or(base_roughness, 0.0).clamp(0.0, 1.0);
    let sigma_sq = toksvig_variance(mean_length);
    (a * a + 2.0 * sigma_sq).max(0.0).sqrt().clamp(0.0, 1.0)
}

/// Scalar cavity ambient occlusion in `[0, 1]` from the sub-texel normal spread.
///
/// Treats each crease as an independent attenuator so visibility follows
/// `1 / (1 + strength * sigma^2)` with `sigma^2` from [`toksvig_variance`].  A
/// coherent patch (`r = 1`, `sigma^2 = 0`) is fully lit (`1`); increasing the
/// variance (lower `r`) darkens the texel toward `0`, and a larger `strength`
/// darkens it faster.  With `strength = 1` this reduces exactly to the mean
/// length `r`, since `1 / (1 + (1 - r)/r) = r`.
///
/// `strength` is clamped to be non-negative.  Non-finite inputs fall back to
/// fully visible (`1`), the safe "no extra occlusion" default.
#[inline]
pub fn cavity_ao(mean_length: f32, strength: f32) -> f32 {
    if !mean_length.is_finite() || !strength.is_finite() {
        return 1.0;
    }
    let s = strength.max(0.0);
    let sigma_sq = toksvig_variance(mean_length);
    (1.0 / (1.0 + s * sigma_sq)).clamp(0.0, 1.0)
}

/// Returns the normalised mean sub-texel normal and its mean-resultant length.
///
/// The mean is `sum(normalize(N_i)) / count` over the *finite, non-degenerate*
/// normals; its length is the mean-resultant length `r` in `[0, 1]`.
/// Degenerate members (zero-length or non-finite) are skipped entirely so they
/// neither bias the direction nor deflate `r`.  When nothing usable remains the
/// fallback is the flat default (`+Z`, `r = 1`), matching
/// [`MicroBentNormal::FLAT`].
#[inline]
pub fn mean_normal(normals: &[Vec3]) -> (Vec3, f32) {
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    for &n in normals {
        if !n.is_finite() {
            continue;
        }
        let unit = normalize_or(n, Vec3::ZERO);
        if unit == Vec3::ZERO {
            continue;
        }
        sum += unit;
        count += 1;
    }
    if count == 0 {
        return (Vec3::Z, 1.0);
    }
    let mean = sum / count as f32;
    let r = mean.length().clamp(0.0, 1.0);
    let dir = normalize_or(mean, Vec3::Z);
    (dir, r)
}

/// Fits a [`MicroBentNormal`] directly from a precomputed mean direction and
/// mean-resultant length.
///
/// Useful when `r = |mean(N)|` has already been accumulated (for example in a
/// mip chain) so the individual normals are no longer available.  The aperture
/// follows `acos(2*r - 1)`, the roughness follows [`effective_roughness`], and
/// the visibility follows [`cavity_ao`].  `mean_length` is clamped to `[0, 1]`;
/// a degenerate or non-finite `direction` falls back to `+Z`.
#[inline]
pub fn micro_bent_normal_from_mean(
    direction: Vec3,
    mean_length: f32,
    base_roughness: f32,
    ao_strength: f32,
) -> MicroBentNormal {
    let r = finite_or(mean_length, 1.0).clamp(0.0, 1.0);
    MicroBentNormal {
        direction: normalize_or(direction, Vec3::Z),
        aperture: resultant_to_aperture(r),
        roughness: effective_roughness(base_roughness, r),
        visibility: cavity_ao(r, ao_strength),
    }
}

/// Fits a [`MicroBentNormal`] to a set of sub-texel normals.
///
/// Computes the mean-resultant statistic with [`mean_normal`] and feeds it to
/// [`micro_bent_normal_from_mean`].  `base_roughness` is the material's own
/// roughness (folded in by the Toksvig term), and `ao_strength` scales the
/// cavity-AO darkening.
///
/// Degenerate handling (never `NaN`): an empty slice, or one whose normals are
/// all zero-length / non-finite, returns [`MicroBentNormal::FLAT`] with the
/// (clamped) base roughness — the flat, fully-visible `r = 1` limit.
#[inline]
pub fn micro_bent_normal(
    normals: &[Vec3],
    base_roughness: f32,
    ao_strength: f32,
) -> MicroBentNormal {
    let (dir, r) = mean_normal(normals);
    micro_bent_normal_from_mean(dir, r, base_roughness, ao_strength)
}

/// Folds a micro-scale bent-normal cone together with a macro-scale cone into a
/// single equivalent [`MergedCone`].
///
/// Each cone is turned back into a resultant vector `r_i * dir_i` (with
/// `r_i = (1 + cos(aperture_i)) / 2`); the two are averaged, so the combined
/// mean-resultant length is `|r_m*d_m + r_M*d_M| / 2`.  Averaging keeps the
/// aperture when the two agree and *widens* it when they disagree (a shorter
/// resultant), never spuriously sharpening it.  The combined visibility is the
/// product of the two scalar AOs — two independent occluders attenuate
/// multiplicatively — clamped to `[0, 1]`.
///
/// Degenerate handling (never `NaN`): a zero-length combined resultant (the two
/// directions cancel) opens the cone fully (`aperture = PI`) about the micro
/// direction; a degenerate macro direction falls back to the micro direction
/// for the axis.
#[inline]
pub fn merge_micro_macro(
    micro: MicroBentNormal,
    macro_direction: Vec3,
    macro_aperture: f32,
    macro_visibility: f32,
) -> MergedCone {
    let micro_dir = normalize_or(micro.direction, Vec3::Z);
    let macro_dir = normalize_or(macro_direction, micro_dir);

    let r_micro = aperture_to_resultant(micro.aperture);
    let r_macro = aperture_to_resultant(macro_aperture);

    let resultant = micro_dir * r_micro + macro_dir * r_macro;
    let len = resultant.length();

    let (direction, aperture) = if len > f32::MIN_POSITIVE {
        // Average of two unit lobes -> divide the resultant length by 2.
        let r = (len * 0.5).clamp(0.0, 1.0);
        (resultant * len.recip(), resultant_to_aperture(r))
    } else {
        // The two lobes cancel: no preferred axis, fully open cone.
        (micro_dir, PI)
    };

    let vis_micro = finite_or(micro.visibility, 1.0).clamp(0.0, 1.0);
    let vis_macro = finite_or(macro_visibility, 1.0).clamp(0.0, 1.0);

    MergedCone {
        direction,
        aperture,
        visibility: (vis_micro * vis_macro).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Deterministic directions spread about `axis` with a controllable cone
    /// half-angle `spread` (radians).  Larger `spread` -> lower resultant `r`.
    fn spread_normals(axis: Vec3, spread: f32, count: usize) -> Vec<Vec3> {
        let axis = normalize_or(axis, Vec3::Z);
        let up = if axis.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let t = normalize_or(up.cross(axis), Vec3::X);
        let b = axis.cross(t);
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            // Uniform cos(theta) in [cos(spread), 1] -> uniform cone.
            let u = (i as f32 + 0.5) / count as f32;
            let cos_min = ops::cos(spread);
            let cos_theta = cos_min + (1.0 - cos_min) * u;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = core::f32::consts::TAU * (i as f32 * 0.618_034);
            let local = Vec3::new(sin_theta * ops::cos(phi), sin_theta * ops::sin(phi), cos_theta);
            out.push(t * local.x + b * local.y + axis * local.z);
        }
        out
    }

    #[test]
    fn coherent_patch_is_flat_and_fully_visible() {
        let n = Vec3::new(0.3, -0.2, 1.0).normalize();
        let normals = [n; 64];
        let mb = micro_bent_normal(&normals, 0.25, 1.0);
        // Mean direction recovers the (shared) input normal.
        assert!(mb.direction.dot(n) > 0.9999, "dir {:?} n {:?}", mb.direction, n);
        assert!((mb.direction.length() - 1.0).abs() < 1e-6);
        // r = 1 -> closed cone.
        assert!(mb.aperture < 5e-3, "aperture {}", mb.aperture);
        // Variance zero -> roughness is just the base.
        assert!((mb.roughness - 0.25).abs() < 1e-5, "roughness {}", mb.roughness);
        // Fully visible.
        assert!((mb.visibility - 1.0).abs() < 1e-6, "vis {}", mb.visibility);
    }

    #[test]
    fn high_variance_lowers_ao_and_widens_cone() {
        let coherent = micro_bent_normal(&spread_normals(Vec3::Z, 0.05, 256), 0.1, 1.0);
        let scattered = micro_bent_normal(&spread_normals(Vec3::Z, 1.2, 256), 0.1, 1.0);
        // Wider spread -> lower AO, both still bounded in [0, 1].
        assert!(scattered.visibility < coherent.visibility, "coh {} scat {}", coherent.visibility, scattered.visibility);
        for v in [coherent.visibility, scattered.visibility, scattered.roughness] {
            assert!((0.0..=1.0).contains(&v), "out of range {}", v);
        }
        // Wider spread -> wider cone and rougher lobe.
        assert!(scattered.aperture > coherent.aperture, "coh {} scat {}", coherent.aperture, scattered.aperture);
        assert!(scattered.roughness > coherent.roughness, "coh {} scat {}", coherent.roughness, scattered.roughness);
    }

    #[test]
    fn fully_scattered_patch_ao_bounded_and_small() {
        // Normals over the whole sphere -> near-zero resultant.
        let mut normals = Vec::new();
        for i in 0..512 {
            let z = -1.0 + 2.0 * (i as f32 + 0.5) / 512.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = core::f32::consts::TAU * (i as f32 * 0.618_034);
            normals.push(Vec3::new(r * ops::cos(phi), r * ops::sin(phi), z));
        }
        let mb = micro_bent_normal(&normals, 0.3, 1.0);
        assert!((0.0..=1.0).contains(&mb.visibility), "vis {}", mb.visibility);
        assert!(mb.visibility < 0.2, "vis should be small, got {}", mb.visibility);
        assert!(mb.aperture > PI * 0.5, "aperture {}", mb.aperture);
        assert!(mb.visibility.is_finite() && mb.roughness.is_finite());
    }

    #[test]
    fn toksvig_variance_matches_closed_form() {
        for &r in &[1.0f32, 0.75, 0.5, 0.25, 0.1] {
            let expected = (1.0 - r) / r;
            assert!((toksvig_variance(r) - expected).abs() < 1e-5, "r {} got {}", r, toksvig_variance(r));
        }
        // r = 0.5 -> sigma^2 = 1 exactly.
        assert!((toksvig_variance(0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cavity_ao_unit_strength_equals_mean_length() {
        for &r in &[1.0f32, 0.8, 0.6, 0.4, 0.2, 0.05] {
            // 1 / (1 + (1-r)/r) == r.
            assert!((cavity_ao(r, 1.0) - r).abs() < 1e-5, "r {} ao {}", r, cavity_ao(r, 1.0));
        }
    }

    #[test]
    fn cavity_ao_matches_variance_closed_form() {
        for &r in &[0.9f32, 0.5, 0.2] {
            for &s in &[0.5f32, 1.0, 4.0] {
                let sigma_sq = (1.0 - r) / r;
                let expected = 1.0 / (1.0 + s * sigma_sq);
                assert!((cavity_ao(r, s) - expected).abs() < 1e-5, "r {} s {} got {}", r, s, cavity_ao(r, s));
            }
        }
    }

    #[test]
    fn cavity_ao_darkens_with_strength() {
        let r = 0.6;
        assert!(cavity_ao(r, 4.0) < cavity_ao(r, 1.0));
        assert!(cavity_ao(r, 1.0) < cavity_ao(r, 0.25));
    }

    #[test]
    fn effective_roughness_recovers_base_at_full_coherence() {
        for &base in &[0.0f32, 0.2, 0.5, 1.0] {
            assert!((effective_roughness(base, 1.0) - base).abs() < 1e-5, "base {}", base);
        }
    }

    #[test]
    fn merge_identity_with_tight_macro_preserves_micro() {
        // Flat micro (r=1, +Z) merged with a tight macro cone along +Z.
        let micro = MicroBentNormal::FLAT;
        let merged = merge_micro_macro(micro, Vec3::Z, 0.0, 1.0);
        assert!(merged.direction.dot(Vec3::Z) > 0.9999, "dir {:?}", merged.direction);
        assert!(merged.aperture < 1e-3, "aperture {}", merged.aperture);
        assert!((merged.visibility - 1.0).abs() < 1e-6, "vis {}", merged.visibility);
    }

    #[test]
    fn merge_disagreement_widens_and_multiplies_visibility() {
        let micro = MicroBentNormal {
            direction: Vec3::Z,
            aperture: 0.3,
            roughness: 0.2,
            visibility: 0.8,
        };
        // Macro leans toward +X with a medium cone and partial visibility.
        let merged = merge_micro_macro(micro, Vec3::X, 0.3, 0.5);
        // Axis sits between the two directions, in the upper hemisphere.
        assert!(merged.direction.x > 0.0 && merged.direction.z > 0.0, "dir {:?}", merged.direction);
        assert!((merged.direction.length() - 1.0).abs() < 1e-5);
        // Visibility is the product of the two.
        assert!((merged.visibility - 0.4).abs() < 1e-6, "vis {}", merged.visibility);
        // Wider than either input cone because the directions disagree.
        assert!(merged.aperture > 0.3, "aperture {}", merged.aperture);
    }

    #[test]
    fn merge_opposite_directions_open_fully() {
        // Two equally tight, opposite cones cancel -> fully open.
        let micro = MicroBentNormal {
            direction: Vec3::Z,
            aperture: 0.0,
            roughness: 0.0,
            visibility: 1.0,
        };
        let merged = merge_micro_macro(micro, Vec3::NEG_Z, 0.0, 1.0);
        assert!((merged.aperture - PI).abs() < 1e-4, "aperture {}", merged.aperture);
        assert!((merged.direction.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_input_falls_back_to_flat() {
        let mb = micro_bent_normal(&[], 0.4, 1.0);
        assert_eq!(mb.direction, Vec3::Z);
        assert_eq!(mb.aperture, 0.0);
        assert!((mb.roughness - 0.4).abs() < 1e-6);
        assert!((mb.visibility - 1.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_normals_are_ignored() {
        // A mix of zero / non-finite normals plus two good ones along +Z.
        let normals = [
            Vec3::ZERO,
            Vec3::new(f32::NAN, 0.0, 1.0),
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
        ];
        let mb = micro_bent_normal(&normals, 0.1, 1.0);
        assert!(mb.direction.dot(Vec3::Z) > 0.9999, "dir {:?}", mb.direction);
        assert!(mb.aperture < 1e-3, "aperture {}", mb.aperture);
        assert!(mb.visibility.is_finite() && (mb.visibility - 1.0).abs() < 1e-5);
    }

    #[test]
    fn non_finite_scalars_never_produce_nan() {
        assert_eq!(cavity_ao(f32::NAN, 1.0), 1.0);
        assert_eq!(cavity_ao(0.5, f32::INFINITY), 1.0);
        assert!(effective_roughness(f32::NAN, f32::NAN).is_finite());
        let mb = micro_bent_normal_from_mean(Vec3::ZERO, f32::NAN, f32::NAN, f32::NAN);
        assert_eq!(mb.direction, Vec3::Z);
        assert!(mb.aperture.is_finite() && mb.roughness.is_finite() && mb.visibility.is_finite());
    }

    #[test]
    fn results_are_deterministic() {
        let normals = spread_normals(Vec3::new(0.2, 0.4, 0.9), 0.6, 128);
        let a = micro_bent_normal(&normals, 0.3, 1.5);
        let b = micro_bent_normal(&normals, 0.3, 1.5);
        assert_eq!(a, b);
        let m1 = merge_micro_macro(a, Vec3::new(0.1, 0.2, 1.0), 0.4, 0.7);
        let m2 = merge_micro_macro(b, Vec3::new(0.1, 0.2, 1.0), 0.4, 0.7);
        assert_eq!(m1, m2);
    }
}
