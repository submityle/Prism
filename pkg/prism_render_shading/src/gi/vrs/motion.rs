//! Motion-magnitude VRS classifier and cross-signal rate combination.
//!
//! This is the CPU golden reference for the *temporal* VRS signal.  Fast screen
//! motion smears a tile across many pixels within a single frame's exposure, so
//! the eye (and any downstream motion-blur reconstruction) cannot resolve
//! fine shading detail there: shading error is **masked by motion blur**.  A
//! fast-moving tile may therefore be shaded coarsely with no perceptible loss,
//! while a static tile must keep full detail because nothing hides a sloppy
//! sample.
//!
//! Two complementary entry points are provided:
//!
//! * [`classify_motion`] takes a scalar speed (the tile's mean motion-vector
//!   length in pixels/frame) and returns an **isotropic** rate — larger speed →
//!   coarser rate — capped at a configurable maximum.
//! * [`classify_motion_vec`] takes the motion vector itself and returns an
//!   **anisotropic** rate: blur is strongest *along* the direction of travel, so
//!   a tile moving mostly horizontally may coarsen more in `x` than in `y`
//!   (and vice-versa).  Per-axis speeds are mapped to per-axis coarsening
//!   factors and assembled with [`ShadingRate::from_factors`].
//!
//! Finally [`combine_rates`] / [`combine_all`] fold several classifiers'
//! proposals into one by taking the **finest** (lowest-coarseness) rate.  Under
//! this conservative rule the motion signal acts as a *permission* to coarsen:
//! a tile is only coarsened where every signal — luma flatness, edge absence,
//! and sufficient motion — independently agrees, so no classifier can force a
//! tile coarser than another wants it fine.
//!
//! # Conventions
//! * Speeds are in **pixels per frame**, the natural output of a tile-velocity
//!   / NeighborMax pass.
//! * Pure, deterministic, `no_std`-friendly: no RNG, IO, GPU, or `unsafe`, and
//!   no allocation (vectors come in as [`bevy_math::Vec2`], proposals as
//!   `&[ShadingRate]`).
//! * [`bevy_math::Vec2`] provides the vector length; no transcendental from
//!   [`bevy_math::ops`] is required.
//! * Defensive clamping is pervasive.  A non-finite speed or vector component
//!   makes the classifier fall back to the finest rate [`ShadingRate::X1x1`] —
//!   shading more is never a visible error — and all thresholds are sanitized
//!   before use.

use bevy_math::Vec2;

use super::ShadingRate;

/// Thresholds for the motion-magnitude → [`ShadingRate`] mapping.
///
/// `still_speed` and `blur_speed` are the two speed boundaries (in
/// pixels/frame) separating the three coarsening levels; they must satisfy
/// `0 <= still_speed <= blur_speed`, which [`MotionThresholds::sanitized`]
/// restores defensively.  `max_rate` caps how coarse motion alone is ever
/// allowed to go.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MotionThresholds {
    /// At or below this speed the tile is effectively static and keeps full
    /// rate (`X1x1`).  Defaults to `0.5` px/frame (sub-pixel motion rounds to
    /// "not moving", matching the classic half-pixel motion-blur cutoff).
    pub still_speed: f32,
    /// At or below this speed the tile is moderately moving and earns a medium
    /// per-axis coarsening (factor `2`); above it the fast-motion coarsening
    /// (factor `4`) applies.  Defaults to `8.0` px/frame.
    pub blur_speed: f32,
    /// Upper bound on the coarseness motion alone may select.  The classifier
    /// returns `result.finer_of(max_rate)`, so a smaller cap keeps fast tiles
    /// finer.  Defaults to [`ShadingRate::X4x4`] (no extra cap).
    pub max_rate: ShadingRate,
}

impl Default for MotionThresholds {
    #[inline]
    fn default() -> Self {
        Self {
            still_speed: 0.5,
            blur_speed: 8.0,
            max_rate: ShadingRate::X4x4,
        }
    }
}

impl MotionThresholds {
    /// Returns a copy with non-finite / negative speeds repaired and
    /// `still_speed <= blur_speed` re-established.
    #[inline]
    pub fn sanitized(self) -> Self {
        let still = sanitize_nonneg(self.still_speed);
        let blur = sanitize_nonneg(self.blur_speed);
        Self {
            still_speed: still.min(blur),
            blur_speed: still.max(blur),
            max_rate: self.max_rate,
        }
    }
}

/// Replaces a non-finite scalar with `0.0`, otherwise returns it unchanged.
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Sanitizes a speed/threshold: non-finite becomes `0.0`, negatives clamp to
/// `0.0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    finite_or_zero(x).max(0.0)
}

/// Maps a per-axis speed to a per-axis coarsening factor in `{1, 2, 4}`.
///
/// `speed <= still_speed` → `1` (no coarsening on this axis); `speed <=
/// blur_speed` → `2`; faster → `4`.  Thresholds are taken already-sanitized.
#[inline]
fn speed_to_factor(speed: f32, t: &MotionThresholds) -> u32 {
    let s = sanitize_nonneg(speed);
    if s <= t.still_speed {
        1
    } else if s <= t.blur_speed {
        2
    } else {
        4
    }
}

/// Classifies a tile from a scalar motion speed (pixels/frame), isotropically.
///
/// Larger speed → coarser rate (`X1x1` → `X2x2` → `X4x4`), then clamped to the
/// configured `max_rate`.  A non-finite speed falls back to the finest rate.
pub fn classify_motion(speed: f32, thresholds: &MotionThresholds) -> ShadingRate {
    if !speed.is_finite() {
        return ShadingRate::X1x1;
    }
    let t = thresholds.sanitized();
    let s = speed.max(0.0);
    let rate = if s <= t.still_speed {
        ShadingRate::X1x1
    } else if s <= t.blur_speed {
        ShadingRate::X2x2
    } else {
        ShadingRate::X4x4
    };
    // Cap: never coarser than the configured maximum.
    rate.finer_of(t.max_rate)
}

/// Classifies a tile from its motion *vector*, anisotropically.
///
/// The per-axis absolute speeds `|vx|` / `|vy|` are mapped to per-axis
/// coarsening factors (via the same ladder as [`classify_motion`]) and
/// assembled with [`ShadingRate::from_factors`], so a tile travelling mostly
/// along one axis coarsens more along that axis — where motion blur is
/// strongest — than across it.  The result is capped at `max_rate`.
///
/// A vector with any non-finite component falls back to the finest rate.
pub fn classify_motion_vec(velocity: Vec2, thresholds: &MotionThresholds) -> ShadingRate {
    if !velocity.x.is_finite() || !velocity.y.is_finite() {
        return ShadingRate::X1x1;
    }
    let t = thresholds.sanitized();
    let fx = speed_to_factor(velocity.x.abs(), &t);
    let fy = speed_to_factor(velocity.y.abs(), &t);
    ShadingRate::from_factors(fx, fy).finer_of(t.max_rate)
}

/// Combines two shading-rate proposals by keeping the **finer** of the two.
///
/// This is the conservative cross-signal merge: whichever classifier demands
/// the most detail wins, so coarsening only happens where all signals concur.
/// It is associative and commutative up to the deterministic equal-rank
/// tie-break of [`ShadingRate::finer_of`].
#[inline]
pub fn combine_rates(a: ShadingRate, b: ShadingRate) -> ShadingRate {
    a.finer_of(b)
}

/// Folds any number of proposals into the single finest rate.
///
/// Starts from [`ShadingRate::COARSEST`] (the identity for a "take the finer"
/// fold) so an **empty** proposal list conservatively yields the coarsest rate;
/// callers that treat "no signal" as "shade fully" should guard emptiness
/// themselves.
#[inline]
pub fn combine_all(rates: &[ShadingRate]) -> ShadingRate {
    rates
        .iter()
        .copied()
        .fold(ShadingRate::COARSEST, combine_rates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_tile_keeps_full_rate() {
        let t = MotionThresholds::default();
        assert_eq!(classify_motion(0.0, &t), ShadingRate::X1x1);
        assert_eq!(classify_motion(0.4, &t), ShadingRate::X1x1);
        assert_eq!(classify_motion_vec(Vec2::ZERO, &t), ShadingRate::X1x1);
    }

    #[test]
    fn moderate_and_fast_motion_coarsen() {
        let t = MotionThresholds::default();
        // Between still (0.5) and blur (8.0) → medium.
        assert_eq!(classify_motion(4.0, &t), ShadingRate::X2x2);
        // Above blur → coarsest.
        assert_eq!(classify_motion(32.0, &t), ShadingRate::X4x4);
    }

    #[test]
    fn directional_motion_is_anisotropic() {
        let t = MotionThresholds::default();
        // Fast purely-horizontal motion: coarsen along x, keep some y detail.
        // fx = 4, fy = 1 → from_factors repairs 4x1 to X4x2.
        let horiz = classify_motion_vec(Vec2::new(32.0, 0.0), &t);
        assert_eq!(horiz, ShadingRate::X4x2);
        // Fast purely-vertical motion is the transpose.
        let vert = classify_motion_vec(Vec2::new(0.0, 32.0), &t);
        assert_eq!(vert, ShadingRate::X2x4);
        // Equal diagonal motion is isotropic.
        let diag = classify_motion_vec(Vec2::new(32.0, 32.0), &t);
        assert_eq!(diag, ShadingRate::X4x4);
    }

    #[test]
    fn max_rate_caps_coarseness() {
        let t = MotionThresholds {
            max_rate: ShadingRate::X2x2,
            ..MotionThresholds::default()
        };
        // Would be X4x4 uncapped; the cap keeps it at X2x2.
        assert_eq!(classify_motion(100.0, &t), ShadingRate::X2x2);
        // Anisotropic fast motion is also capped to the finer of result/cap.
        let capped = classify_motion_vec(Vec2::new(100.0, 0.0), &t);
        assert_eq!(capped, capped.finer_of(ShadingRate::X2x2));
        assert!(capped.rank() <= ShadingRate::X2x2.rank());
    }

    #[test]
    fn non_finite_motion_falls_back_to_full_rate() {
        let t = MotionThresholds::default();
        assert_eq!(classify_motion(f32::NAN, &t), ShadingRate::X1x1);
        assert_eq!(classify_motion(f32::INFINITY, &t), ShadingRate::X1x1);
        assert_eq!(
            classify_motion_vec(Vec2::new(f32::NAN, 1.0), &t),
            ShadingRate::X1x1
        );
        assert_eq!(
            classify_motion_vec(Vec2::new(1.0, f32::INFINITY), &t),
            ShadingRate::X1x1
        );
    }

    #[test]
    fn classify_motion_is_monotonic_in_speed() {
        // As speed rises the isotropic rate must get *coarser* (rank
        // non-decreasing): more blur can only ever permit more coarsening.
        let t = MotionThresholds::default();
        let mut prev_rank = classify_motion(0.0, &t).rank();
        let mut s = 0.0_f32;
        while s <= 64.0 {
            let rank = classify_motion(s, &t).rank();
            assert!(rank >= prev_rank, "coarseness decreased as speed rose at {s}");
            prev_rank = rank;
            s += 0.5;
        }
    }

    #[test]
    fn classify_motion_vec_is_monotonic_along_a_ray() {
        // Scaling a fixed direction up only ever coarsens.
        let t = MotionThresholds::default();
        let dir = Vec2::new(1.0, 0.3).normalize();
        let mut prev_rank = classify_motion_vec(Vec2::ZERO, &t).rank();
        let mut mag = 0.0_f32;
        while mag <= 64.0 {
            let rank = classify_motion_vec(dir * mag, &t).rank();
            assert!(rank >= prev_rank, "coarseness decreased at magnitude {mag}");
            prev_rank = rank;
            mag += 0.5;
        }
    }

    #[test]
    fn combine_takes_the_finer() {
        assert_eq!(
            combine_rates(ShadingRate::X4x4, ShadingRate::X2x2),
            ShadingRate::X2x2
        );
        assert_eq!(
            combine_rates(ShadingRate::X1x1, ShadingRate::X4x4),
            ShadingRate::X1x1
        );
    }

    #[test]
    fn combine_all_folds_to_the_finest() {
        let rates = [
            ShadingRate::X4x4,
            ShadingRate::X2x4,
            ShadingRate::X1x2,
            ShadingRate::X2x2,
        ];
        assert_eq!(combine_all(&rates), ShadingRate::X1x2);
        // Empty list is the fold identity: coarsest.
        assert_eq!(combine_all(&[]), ShadingRate::COARSEST);
        // Single element passes through.
        assert_eq!(combine_all(&[ShadingRate::X2x1]), ShadingRate::X2x1);
    }

    #[test]
    fn combine_is_commutative_and_associative() {
        let a = ShadingRate::X2x4;
        let b = ShadingRate::X1x2;
        let c = ShadingRate::X4x2;
        assert_eq!(combine_rates(a, b), combine_rates(b, a));
        assert_eq!(
            combine_rates(combine_rates(a, b), c),
            combine_rates(a, combine_rates(b, c))
        );
    }
}
