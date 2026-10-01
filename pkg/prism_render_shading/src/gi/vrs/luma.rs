//! Luminance-variance / just-noticeable-difference (JND) VRS classifier.
//!
//! This is the CPU golden reference for the *perceptual* VRS signal: it decides
//! how coarsely a tile may be shaded based on how much its luminance varies
//! relative to what the human visual system can actually resolve.
//!
//! The guiding principle is **Weber's law**: the smallest luminance change a
//! viewer can notice, `ΔL`, is roughly proportional to the background
//! luminance `L` — their ratio `ΔL / L` (the *Weber fraction*) is approximately
//! constant across a wide luminance range (empirically ≈ 1–2 % for the
//! foveal luminance channel).  A tile whose internal luminance spread, measured
//! as a Weber-style relative contrast, falls below that just-noticeable
//! threshold carries no perceptible detail, so it may be shaded at a coarse
//! rate without any visible artifact.  As the relative contrast climbs above
//! the threshold the tile carries progressively more visible structure and must
//! be shaded more finely.
//!
//! The classifier therefore maps a single scalar — the tile's Weber contrast —
//! onto an **isotropic** rung of the [`ShadingRate`] lattice:
//! `X4x4` (imperceptible) → `X2x2` (mild) → `X1x1` (clearly visible).  Direction
//! (anisotropy) is intentionally *not* inferred here; that is the job of the
//! [`edge`](super::edge) and [`motion`](super::motion) classifiers, which have
//! the 2-D / vector information needed to justify an anisotropic choice.  Keeping
//! this stage isotropic keeps its monotonicity guarantee clean: contrast only
//! ever moves the result in the "finer" direction.
//!
//! # Conventions
//! * Input is a borrowed slice of per-pixel luminances (`&[f32]`); the tile's
//!   2-D shape is irrelevant to a variance measure, so none is required and no
//!   allocation occurs.
//! * Pure, deterministic, `no_std`-friendly: no RNG, IO, GPU, or `unsafe`.
//! * Only `x.sqrt()` is needed for the standard deviation; no
//!   [`bevy_math::ops`] transcendental is required.
//! * Defensive clamping is pervasive.  An empty tile, any non-finite sample, or
//!   a non-positive mean (which would make the Weber ratio ill-defined) all
//!   fall back to the finest rate [`ShadingRate::X1x1`] — shading more is never
//!   a visible error.

use super::ShadingRate;

/// Luminance floor used to keep the Weber ratio `stddev / mean` well-defined for
/// very dark tiles.
///
/// Deep shadow regions have a mean luminance near zero, which would make the
/// relative contrast explode.  Clamping the denominator to this floor both
/// avoids the divide-by-zero and reflects the physiological *dark-light* /
/// absolute threshold: below a certain luminance the eye's contrast sensitivity
/// collapses, so near-black tiles are treated as low-contrast.
const LUMINANCE_FLOOR: f32 = 1.0e-3;

/// Thresholds for the Weber-contrast → [`ShadingRate`] mapping.
///
/// Both thresholds are expressed as **Weber fractions** (dimensionless relative
/// contrast, `stddev / mean`).  They must satisfy `0 <= jnd_fraction <=
/// detail_fraction`; [`LumaThresholds::sanitized`] re-establishes that ordering
/// defensively if a caller supplies something out of order.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LumaThresholds {
    /// At or below this relative contrast the tile is perceptually flat and may
    /// be shaded at the coarsest rate (`X4x4`).  This is the just-noticeable
    /// difference (Weber fraction); the default of `0.02` matches the classic
    /// ≈ 2 % foveal luminance JND.
    pub jnd_fraction: f32,
    /// At or below this relative contrast the tile carries mild structure and
    /// is shaded at a medium rate (`X2x2`); above it the tile is shaded at full
    /// rate (`X1x1`).  Defaults to `0.10` (≈ five JNDs).
    pub detail_fraction: f32,
}

impl Default for LumaThresholds {
    #[inline]
    fn default() -> Self {
        Self {
            jnd_fraction: 0.02,
            detail_fraction: 0.10,
        }
    }
}

impl LumaThresholds {
    /// Returns a copy with non-finite / negative fields replaced by sane values
    /// and the two thresholds re-ordered so `jnd_fraction <= detail_fraction`.
    ///
    /// This makes every public entry point robust to adversarial configuration
    /// without sprinkling guards through the classifier body.
    #[inline]
    pub fn sanitized(self) -> Self {
        let jnd = sanitize_nonneg(self.jnd_fraction);
        let detail = sanitize_nonneg(self.detail_fraction);
        Self {
            jnd_fraction: jnd.min(detail),
            detail_fraction: jnd.max(detail),
        }
    }
}

/// First- and second-moment statistics of a tile's luminance.
///
/// Produced by [`tile_statistics`]; all fields are finite for a valid
/// (non-empty, all-finite) tile.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LumaStats {
    /// Number of samples that contributed.
    pub count: usize,
    /// Arithmetic mean luminance.
    pub mean: f32,
    /// Population variance (divided by `count`, not `count - 1`).
    pub variance: f32,
    /// Minimum sample.
    pub min: f32,
    /// Maximum sample.
    pub max: f32,
}

impl LumaStats {
    /// Population standard deviation, `sqrt(variance)`.
    #[inline]
    pub fn std_dev(&self) -> f32 {
        self.variance.max(0.0).sqrt()
    }
}

/// Replaces a non-finite scalar with `0.0`, otherwise returns it unchanged.
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Sanitizes a threshold: non-finite becomes `0.0`, negatives clamp up to
/// `0.0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    finite_or_zero(x).max(0.0)
}

/// Computes mean / population-variance / min / max of a tile's luminances in a
/// single numerically careful pass.
///
/// Returns `None` for a degenerate tile — one that is empty or contains any
/// non-finite sample — so callers can route straight to the safe fallback.  The
/// variance uses the shifted-data (two-pass) formulation to avoid the
/// catastrophic cancellation of the naive "mean of squares minus square of
/// mean" estimator for tiles with a large DC offset.
pub fn tile_statistics(samples: &[f32]) -> Option<LumaStats> {
    if samples.is_empty() {
        return None;
    }
    let mut sum = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &s in samples {
        if !s.is_finite() {
            return None;
        }
        sum += s;
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }
    let count = samples.len();
    let mean = sum / count as f32;
    // Second pass about the mean for numerical stability.
    let mut sq = 0.0_f32;
    for &s in samples {
        let d = s - mean;
        sq += d * d;
    }
    let variance = (sq / count as f32).max(0.0);
    Some(LumaStats {
        count,
        mean,
        variance,
        min,
        max,
    })
}

/// Weber-style relative contrast of a tile, `stddev / max(mean, LUMINANCE_FLOOR)`.
///
/// This is the perceptual quantity the classifier thresholds against: a
/// dimensionless ratio that is (approximately) constant at the detection
/// threshold across luminance levels, per Weber's law.  The denominator is
/// floored to keep very dark tiles well-behaved.
///
/// Returns `0.0` for a degenerate tile, which maps to "perceptually flat".
pub fn weber_contrast(samples: &[f32]) -> f32 {
    match tile_statistics(samples) {
        Some(stats) => {
            // `mean` can be negative for signed inputs; the floor keeps the
            // denominator strictly positive either way.
            let denom = stats.mean.max(LUMINANCE_FLOOR);
            let c = stats.std_dev() / denom;
            if c.is_finite() { c.max(0.0) } else { 0.0 }
        }
        None => 0.0,
    }
}

/// Michelson contrast of a tile, `(max - min) / (max + min)`.
///
/// Offered as an alternative perceptual measure (classic for periodic / grating
/// stimuli).  Not used by [`classify_luma`] itself but handy for callers and
/// tests; returns `0.0` for a degenerate or non-positive-sum tile.
pub fn michelson_contrast(samples: &[f32]) -> f32 {
    match tile_statistics(samples) {
        Some(stats) => {
            let denom = stats.max + stats.min;
            if denom <= LUMINANCE_FLOOR {
                return 0.0;
            }
            let c = (stats.max - stats.min) / denom;
            if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 }
        }
        None => 0.0,
    }
}

/// Maps a Weber contrast value onto the isotropic shading-rate ladder.
///
/// This is the pure numeric core shared by [`classify_luma`]; exposed so tests
/// (and callers that already hold a contrast) can exercise the mapping
/// directly.  Thresholds are sanitized before use.
///
/// * `contrast <= jnd_fraction`        → [`ShadingRate::X4x4`] (imperceptible)
/// * `contrast <= detail_fraction`     → [`ShadingRate::X2x2`] (mild structure)
/// * otherwise                         → [`ShadingRate::X1x1`] (visible detail)
#[inline]
pub fn rate_from_contrast(contrast: f32, thresholds: &LumaThresholds) -> ShadingRate {
    let t = thresholds.sanitized();
    let c = if contrast.is_finite() { contrast.max(0.0) } else {
        // A non-finite contrast is treated as "maximally detailed": shade fully.
        return ShadingRate::X1x1;
    };
    if c <= t.jnd_fraction {
        ShadingRate::X4x4
    } else if c <= t.detail_fraction {
        ShadingRate::X2x2
    } else {
        ShadingRate::X1x1
    }
}

/// Classifies a tile from its luminance samples using the Weber-contrast JND
/// model.
///
/// Low relative contrast → coarse rate; high relative contrast → fine rate.
/// Any degenerate input (empty tile or a non-finite sample, detected by
/// [`tile_statistics`] inside [`weber_contrast`]) yields a contrast of `0.0`,
/// which the mapping would turn into `X4x4`; to stay strictly safe, degenerate
/// tiles are instead forced to the finest rate here.
pub fn classify_luma(samples: &[f32], thresholds: &LumaThresholds) -> ShadingRate {
    // Degenerate tiles must shade fully, so detect them explicitly rather than
    // trusting the `0.0` contrast fallback (which would coarsen).
    let stats = match tile_statistics(samples) {
        Some(s) => s,
        None => return ShadingRate::X1x1,
    };
    let denom = stats.mean.max(LUMINANCE_FLOOR);
    let contrast = stats.std_dev() / denom;
    rate_from_contrast(contrast, thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn statistics_match_hand_computed() {
        let s = [1.0_f32, 2.0, 3.0, 4.0];
        let stats = tile_statistics(&s).expect("finite tile");
        assert_eq!(stats.count, 4);
        assert!((stats.mean - 2.5).abs() < EPS);
        // Population variance of 1..4 is 1.25.
        assert!((stats.variance - 1.25).abs() < 1.0e-5);
        assert!((stats.min - 1.0).abs() < EPS);
        assert!((stats.max - 4.0).abs() < EPS);
        assert!((stats.std_dev() - 1.25_f32.sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn degenerate_tiles_return_none_and_full_rate() {
        assert!(tile_statistics(&[]).is_none());
        assert!(tile_statistics(&[1.0, f32::NAN]).is_none());
        assert!(tile_statistics(&[f32::INFINITY]).is_none());
        let t = LumaThresholds::default();
        assert_eq!(classify_luma(&[], &t), ShadingRate::X1x1);
        assert_eq!(classify_luma(&[1.0, f32::NAN, 0.5], &t), ShadingRate::X1x1);
    }

    #[test]
    fn flat_tile_is_coarsest() {
        let t = LumaThresholds::default();
        let flat = [0.5_f32; 16];
        assert!((weber_contrast(&flat)).abs() < EPS);
        assert_eq!(classify_luma(&flat, &t), ShadingRate::X4x4);
    }

    #[test]
    fn high_contrast_tile_is_finest() {
        let t = LumaThresholds::default();
        // Half black, half white around a mid mean: huge relative contrast.
        let hi = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        assert!(weber_contrast(&hi) > t.detail_fraction);
        assert_eq!(classify_luma(&hi, &t), ShadingRate::X1x1);
    }

    #[test]
    fn mild_contrast_tile_is_medium() {
        let t = LumaThresholds::default();
        // Mean 1.0, stddev chosen to land between the two Weber thresholds
        // (0.02 .. 0.10). Deviations of +/-0.05 give stddev 0.05 → contrast 0.05.
        let mild = [0.95_f32, 1.05, 0.95, 1.05, 0.95, 1.05, 0.95, 1.05];
        let c = weber_contrast(&mild);
        assert!(c > t.jnd_fraction && c <= t.detail_fraction, "contrast {c}");
        assert_eq!(classify_luma(&mild, &t), ShadingRate::X2x2);
    }

    #[test]
    fn rate_is_monotonic_in_contrast() {
        // As the Weber contrast rises, the chosen rate must get *finer* (its
        // coarseness rank must be non-increasing): detail is never discarded by
        // adding more detail.
        let t = LumaThresholds::default();
        let mut prev_rank = rate_from_contrast(0.0, &t).rank();
        let mut c = 0.0_f32;
        while c <= 0.5 {
            let rank = rate_from_contrast(c, &t).rank();
            assert!(
                rank <= prev_rank,
                "coarseness increased with contrast at c={c}"
            );
            prev_rank = rank;
            c += 0.005;
        }
    }

    #[test]
    fn classify_is_monotonic_as_variance_grows() {
        // Fix the mean and widen the spread; the realized rate must never get
        // coarser as variance increases.
        let t = LumaThresholds::default();
        let mean = 1.0_f32;
        let mut prev_rank = ShadingRate::X4x4.rank();
        let mut spread = 0.0_f32;
        while spread <= 0.4 {
            let tile = [
                mean - spread,
                mean + spread,
                mean - spread,
                mean + spread,
            ];
            let rank = classify_luma(&tile, &t).rank();
            assert!(rank <= prev_rank, "coarsened as spread grew at {spread}");
            prev_rank = rank;
            spread += 0.01;
        }
    }

    #[test]
    fn dark_tiles_do_not_explode() {
        // A near-black flat tile has mean ~ 0; the luminance floor must keep the
        // contrast finite and small (coarse), not blow up to "fine".
        let t = LumaThresholds::default();
        let dark = [1.0e-5_f32; 9];
        let c = weber_contrast(&dark);
        assert!(c.is_finite());
        assert_eq!(classify_luma(&dark, &t), ShadingRate::X4x4);
    }

    #[test]
    fn thresholds_sanitize_reorders_and_clamps() {
        let bad = LumaThresholds {
            jnd_fraction: 0.3,
            detail_fraction: -1.0,
        };
        let s = bad.sanitized();
        assert!(s.jnd_fraction <= s.detail_fraction);
        assert!(s.jnd_fraction >= 0.0 && s.detail_fraction >= 0.0);
        // Non-finite config does not panic and still orders correctly.
        let nan = LumaThresholds {
            jnd_fraction: f32::NAN,
            detail_fraction: f32::INFINITY,
        };
        let s2 = nan.sanitized();
        assert!(s2.jnd_fraction.is_finite());
        assert!(s2.jnd_fraction <= s2.detail_fraction);
    }

    #[test]
    fn michelson_matches_definition() {
        let s = [0.2_f32, 0.6];
        // (0.6 - 0.2) / (0.6 + 0.2) = 0.5.
        assert!((michelson_contrast(&s) - 0.5).abs() < 1.0e-5);
        assert!(michelson_contrast(&[]).abs() < EPS);
    }
}
