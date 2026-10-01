//! Reservoir-based spatiotemporal resampling for screen-probe GI — CPU golden.
//!
//! Screen-probe global illumination can only afford a tiny ray budget per
//! probe per frame, so it leans on *reservoir resampling* to recycle good
//! samples across neighbouring probes (space) and across frames (time).  This
//! module is the backend-neutral numerical reference for that machinery:
//!
//! * [`Reservoir`] is a weighted-reservoir-sampling (WRS) container that keeps
//!   a single surviving candidate drawn with probability proportional to its
//!   resampling weight, together with the running weight sum `w_sum`, the
//!   *confidence weight* `m` (how many candidate samples it summarises), and
//!   the finalised *unbiased contribution weight* `W`.
//! * [`Reservoir::update`] is the streaming resampled-importance-sampling (RIS)
//!   step: it folds one candidate in using an externally supplied uniform in
//!   `[0, 1]`, so the structure stays a deterministic pure function with no
//!   internal RNG.
//! * [`Reservoir::merge`] combines a second reservoir (a temporal predecessor
//!   or a spatial neighbour) following the generalized resampled importance
//!   sampling (`GRIS`) rules: confidence weights `m` act as the pairing counts
//!   of the balance heuristic, so re-dividing by the total `m` in
//!   [`Reservoir::finalize_weight`] yields the multiple-importance-sampling
//!   (MIS) weighted, (nearly) unbiased estimator.
//! * [`Reservoir::finalize_weight`] converts the accumulated state into the
//!   contribution weight `W = (1 / p_hat) * (w_sum / m)`.
//! * [`GiSample`] is the sample payload — a visible point and a secondary
//!   sample point with their normals plus the sampled RGB radiance — and
//!   [`target_function`] is the scalar target `p_hat` (luminance times the
//!   geometric term) that drives resampling.
//!
//! The algorithm follows Bitterli et al. 2020 (*Spatiotemporal reservoir
//! resampling for real-time ray tracing*), Ouyang et al. 2021 (*`ReSTIR` GI*)
//! and Lin et al. 2022 (*Generalized resampled importance sampling*, `GRIS`).
//!
//! # Conventions
//! * All state is stored as `f32` to mirror the GPU reservoir buffer twin: a
//!   reservoir packs into `w_sum`, `m`, `W` scalars plus the sample payload,
//!   exactly as the WESL/GPU structure will.
//! * Resampling weights are *non-negative*.  A candidate with a non-finite or
//!   non-positive weight is a degenerate sample and is discarded without
//!   perturbing the reservoir (it does not even raise the confidence `m`),
//!   because such a sample carries no information and must never be selected.
//! * The replacement test is `u * w_sum <= weight`, with `u` the supplied
//!   uniform clamped to `[0, 1]`.  Using `<=` guarantees the first valid
//!   candidate is always accepted even when `u == 1`, so a reservoir is never
//!   left empty after a successful [`update`](Reservoir::update).
//! * `m` is capped by [`Reservoir::cap_confidence`] to bound temporal history
//!   (the standard `ReSTIR` M-cap), which prevents stale samples from
//!   dominating and keeps the estimator responsive to lighting changes.
//! * [`finalize_weight`](Reservoir::finalize_weight) and
//!   [`target_function`] clamp against zero/`NaN`: every division guards its
//!   denominator and every returned weight is finite and non-negative, so the
//!   reference can never inject `NaN` energy into the lighting integral.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.

use bevy_math::Vec3;

/// Rec. 709 luminance coefficient for the red channel.
const LUMA_R: f32 = 0.212_639;
/// Rec. 709 luminance coefficient for the green channel.
const LUMA_G: f32 = 0.715_169;
/// Rec. 709 luminance coefficient for the blue channel.
const LUMA_B: f32 = 0.072_192;

/// Scalar Rec. 709 relative luminance of a linear RGB radiance triple.
///
/// Negative channels (which a biased upstream estimator can produce) are
/// clamped to zero before weighting so the luminance is always non-negative.
#[inline]
pub fn luminance(rgb: Vec3) -> f32 {
    LUMA_R * rgb.x.max(0.0) + LUMA_G * rgb.y.max(0.0) + LUMA_B * rgb.z.max(0.0)
}

/// Geometric coupling term between a visible point and a sample point.
///
/// This is the standard surface-to-surface geometry factor
/// `cos_v * cos_s / dist^2`, where `cos_v` is the cosine at the visible point's
/// normal towards the sample point, `cos_s` is the cosine at the sample point's
/// normal back towards the visible point, and `dist^2` is their squared
/// separation.  It is the geometry part of the path throughput that `ReSTIR` GI
/// folds into its scalar target.
///
/// The term is `0` for a degenerate (coincident) pair, for back-facing
/// configurations, and whenever the result would be non-finite, giving a
/// deterministic, non-negative fallback.
#[inline]
pub fn geometric_term(
    visible_point: Vec3,
    visible_normal: Vec3,
    sample_point: Vec3,
    sample_normal: Vec3,
) -> f32 {
    let delta = sample_point - visible_point;
    let dist_sq = delta.length_squared();
    // A `NaN` or non-positive squared distance (coincident points) fails this
    // test and falls through to the deterministic zero fallback.
    if dist_sq.is_finite() && dist_sq > f32::MIN_POSITIVE {
        let inv_dist = dist_sq.sqrt().recip();
        let dir = delta * inv_dist;
        let cos_v = visible_normal.dot(dir).max(0.0);
        let cos_s = sample_normal.dot(-dir).max(0.0);
        let term = cos_v * cos_s / dist_sq;
        if term.is_finite() {
            return term.max(0.0);
        }
    }
    0.0
}

/// A single screen-probe GI path sample (the reservoir payload).
///
/// It records the primary *visible point* `x_v` (where shading happens) and the
/// secondary *sample point* `x_s` (where incoming radiance was gathered), each
/// with its surface normal, plus the RGB radiance leaving `x_s` towards `x_v`.
/// All fields are `f32`-backed [`Vec3`]s to match the GPU sample buffer twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiSample {
    /// Visible-point (shading-point) world position `x_v`.
    pub visible_point: Vec3,
    /// Unit surface normal at the visible point.
    pub visible_normal: Vec3,
    /// Secondary sample-point world position `x_s`.
    pub sample_point: Vec3,
    /// Unit surface normal at the sample point.
    pub sample_normal: Vec3,
    /// Linear RGB radiance leaving `x_s` towards `x_v`.
    pub radiance: Vec3,
}

impl Default for GiSample {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl GiSample {
    /// A fully-zeroed sample that carries no energy.
    pub const ZERO: Self = Self {
        visible_point: Vec3::ZERO,
        visible_normal: Vec3::ZERO,
        sample_point: Vec3::ZERO,
        sample_normal: Vec3::ZERO,
        radiance: Vec3::ZERO,
    };
}

/// Scalar resampling target `p_hat` for a GI sample: luminance of the sampled
/// radiance weighted by the surface-to-surface geometric term.
///
/// This is the unnormalised target function `ReSTIR` resamples towards.  It is
/// always finite and non-negative; a degenerate geometry or dark sample yields
/// `0`, which the reservoir then discards.
#[inline]
pub fn target_function(sample: &GiSample) -> f32 {
    let g = geometric_term(
        sample.visible_point,
        sample.visible_normal,
        sample.sample_point,
        sample.sample_normal,
    );
    let t = luminance(sample.radiance) * g;
    if t.is_finite() {
        t.max(0.0)
    } else {
        0.0
    }
}

/// Generalized balance-heuristic MIS weight for technique `index`.
///
/// Given per-technique target densities `pdfs` and confidence counts `counts`,
/// returns the balance-heuristic weight
/// `(c_i * p_i) / sum_j (c_j * p_j)` clamped to `[0, 1]`.  With unit counts
/// this is Veach's classic balance heuristic; with `ReSTIR` confidence weights
/// it is the pairing-count weight used by `GRIS` to keep spatiotemporal reuse
/// (nearly) unbiased.
///
/// Returns `0` for an out-of-range `index`, a non-positive numerator, or a
/// degenerate (zero / non-finite) denominator.
#[inline]
pub fn balance_heuristic(pdfs: &[f32], counts: &[f32], index: usize) -> f32 {
    let n = pdfs.len().min(counts.len());
    if index >= n {
        return 0.0;
    }
    // `f32::max(_, 0.0)` maps `NaN` to `0.0`, so both the numerator and the
    // accumulated denominator are non-negative and `NaN`-free below.
    let num = counts[index].max(0.0) * pdfs[index].max(0.0);
    if num <= 0.0 {
        return 0.0;
    }
    let mut denom = 0.0f32;
    for i in 0..n {
        denom += counts[i].max(0.0) * pdfs[i].max(0.0);
    }
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (num / denom).clamp(0.0, 1.0)
}

/// A weighted-reservoir-sampling container holding one surviving candidate.
///
/// The generic payload `S` is `Copy` so the reservoir itself is a plain,
/// trivially-copyable value matching the GPU buffer twin.  Use [`GiSample`] for
/// screen-probe GI; the container is otherwise payload-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservoir<S: Copy> {
    /// The currently selected candidate, or `None` while the reservoir is
    /// empty (no valid candidate has been folded in yet).
    sample: Option<S>,
    /// Running sum of resampling weights of every candidate considered.
    w_sum: f32,
    /// Confidence weight: the number of candidate samples this reservoir
    /// summarises (fractional after merges / capping).
    m: f32,
    /// Finalised unbiased contribution weight `W`, valid only after
    /// [`finalize_weight`](Self::finalize_weight).
    w: f32,
}

impl<S: Copy> Default for Reservoir<S> {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<S: Copy> Reservoir<S> {
    /// An empty reservoir: no sample, zero weight sum, zero confidence.
    pub const EMPTY: Self = Self {
        sample: None,
        w_sum: 0.0,
        m: 0.0,
        w: 0.0,
    };

    /// Creates an empty reservoir (alias for [`Reservoir::EMPTY`]).
    #[inline]
    pub fn new() -> Self {
        Self::EMPTY
    }

    /// The currently selected candidate, if any.
    #[inline]
    pub fn sample(&self) -> Option<S> {
        self.sample
    }

    /// Running sum of resampling weights (`w_sum`).
    #[inline]
    pub fn weight_sum(&self) -> f32 {
        self.w_sum
    }

    /// Confidence weight `m` (effective candidate count).
    #[inline]
    pub fn confidence(&self) -> f32 {
        self.m
    }

    /// Finalised unbiased contribution weight `W`.
    ///
    /// Meaningful only after [`finalize_weight`](Self::finalize_weight); it is
    /// `0` on a fresh or empty reservoir.
    #[inline]
    pub fn contribution_weight(&self) -> f32 {
        self.w
    }

    /// Whether the reservoir currently holds no selected candidate.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sample.is_none()
    }

    /// Streaming RIS update: fold one candidate `sample` with resampling
    /// `weight` into the reservoir using the external uniform `rng_uniform`.
    ///
    /// Returns `true` when the candidate replaced the surviving sample.  The
    /// candidate survives with probability `weight / w_sum`; the first valid
    /// candidate always survives.  Degenerate candidates (non-finite or
    /// non-positive `weight`) are discarded without touching any state,
    /// including the confidence `m`.
    #[inline]
    pub fn update(&mut self, sample: S, weight: f32, rng_uniform: f32) -> bool {
        if !weight.is_finite() || weight <= 0.0 {
            return false;
        }
        self.w_sum += weight;
        self.m += 1.0;
        let u = rng_uniform.clamp(0.0, 1.0);
        if u * self.w_sum <= weight {
            self.sample = Some(sample);
            true
        } else {
            false
        }
    }

    /// Merges another reservoir into this one following `GRIS` reuse.
    ///
    /// `other_target_pdf` is the target density `p_hat` of `other`'s selected
    /// sample *re-evaluated in this reservoir's domain* (shift-mapped for a
    /// spatial neighbour, identical for a temporal predecessor).  The induced
    /// resampling weight is `other.m * other_target_pdf * other.W`.
    ///
    /// The confidence `m` always accumulates `other.m`, even when the induced
    /// weight is zero, so the division by the total confidence in
    /// [`finalize_weight`](Self::finalize_weight) realises the balance-heuristic
    /// MIS correction.  Returns `true` when `other`'s sample was selected.
    #[inline]
    pub fn merge(&mut self, other: &Reservoir<S>, other_target_pdf: f32, rng_uniform: f32) -> bool {
        if !other.m.is_finite() || other.m <= 0.0 {
            return false;
        }
        // Confidence always accumulates to keep the balance heuristic correct.
        self.m += other.m;
        let rw = other.m * other_target_pdf.max(0.0) * other.w.max(0.0);
        if !rw.is_finite() || rw <= 0.0 {
            return false;
        }
        self.w_sum += rw;
        let u = rng_uniform.clamp(0.0, 1.0);
        if u * self.w_sum <= rw {
            self.sample = other.sample;
            true
        } else {
            false
        }
    }

    /// Caps the confidence weight `m` to `max_m` (the `ReSTIR` M-cap).
    ///
    /// This bounds how much temporal history a reservoir may accumulate so a
    /// long-lived chain of frames cannot overwhelm fresh samples and lag behind
    /// lighting changes.  A negative `max_m` is ignored (no cap applied).
    #[inline]
    pub fn cap_confidence(&mut self, max_m: f32) {
        if max_m >= 0.0 && self.m > max_m {
            self.m = max_m;
        }
    }

    /// Computes the unbiased contribution weight `W = (1 / p_hat) * (w_sum / m)`
    /// from the selected sample's target density `target_pdf` (`p_hat`).
    ///
    /// `W` is stored and returned by [`contribution_weight`](Self::contribution_weight).
    /// It is forced to `0` whenever the reservoir is empty, the confidence is
    /// non-positive, the target density is non-positive / non-finite, or the
    /// result would be non-finite — so `W` is always finite and non-negative.
    #[inline]
    pub fn finalize_weight(&mut self, target_pdf: f32) {
        if self.sample.is_none()
            || self.m <= 0.0
            || !self.w_sum.is_finite()
            || !target_pdf.is_finite()
            || target_pdf <= 0.0
        {
            self.w = 0.0;
            return;
        }
        let w = (self.w_sum / self.m) / target_pdf;
        self.w = if w.is_finite() && w >= 0.0 { w } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial payload used to exercise the generic reservoir machinery.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Tagged(u32);

    #[test]
    fn luminance_is_non_negative_and_weighted() {
        assert_eq!(luminance(Vec3::ZERO), 0.0);
        // Negative channels clamp to zero.
        assert_eq!(luminance(Vec3::new(-1.0, -1.0, -1.0)), 0.0);
        // Green is weighted the most.
        let r = luminance(Vec3::new(1.0, 0.0, 0.0));
        let g = luminance(Vec3::new(0.0, 1.0, 0.0));
        let b = luminance(Vec3::new(0.0, 0.0, 1.0));
        assert!(g > r && r > b);
        assert!((r + g + b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn geometric_term_matches_analytic_facing_pair() {
        // Two unit-separated planes facing each other head-on: cos_v = cos_s =
        // 1, dist^2 = 1, so G = 1.
        let g = geometric_term(Vec3::ZERO, Vec3::Z, Vec3::Z, Vec3::NEG_Z);
        assert!((g - 1.0).abs() < 1e-6, "g={g}");
    }

    #[test]
    fn geometric_term_falls_off_with_inverse_square_distance() {
        let g1 = geometric_term(Vec3::ZERO, Vec3::Z, Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z);
        let g2 = geometric_term(Vec3::ZERO, Vec3::Z, Vec3::new(0.0, 0.0, 2.0), Vec3::NEG_Z);
        // Doubling distance quarters the geometry term.
        assert!((g1 - 1.0).abs() < 1e-6);
        assert!((g2 - 0.25).abs() < 1e-6, "g2={g2}");
    }

    #[test]
    fn geometric_term_degenerate_and_backfacing_are_zero() {
        // Coincident points.
        assert_eq!(geometric_term(Vec3::ZERO, Vec3::Z, Vec3::ZERO, Vec3::Z), 0.0);
        // Visible normal points away from the sample point.
        assert_eq!(
            geometric_term(Vec3::ZERO, Vec3::NEG_Z, Vec3::Z, Vec3::NEG_Z),
            0.0
        );
        // Sample normal points away from the visible point.
        assert_eq!(geometric_term(Vec3::ZERO, Vec3::Z, Vec3::Z, Vec3::Z), 0.0);
    }

    #[test]
    fn target_function_is_product_of_luminance_and_geometry() {
        let s = GiSample {
            visible_point: Vec3::ZERO,
            visible_normal: Vec3::Z,
            sample_point: Vec3::Z,
            sample_normal: Vec3::NEG_Z,
            radiance: Vec3::new(0.0, 1.0, 0.0),
        };
        // G = 1, luminance(green) = LUMA_G.
        assert!((target_function(&s) - LUMA_G).abs() < 1e-6);
    }

    #[test]
    fn target_function_zero_for_dark_or_degenerate() {
        // A geometrically valid pair but zero radiance -> zero target.
        let dark = GiSample {
            visible_point: Vec3::ZERO,
            visible_normal: Vec3::Z,
            sample_point: Vec3::Z,
            sample_normal: Vec3::NEG_Z,
            radiance: Vec3::ZERO,
        };
        assert_eq!(target_function(&dark), 0.0);
        // A fully-zeroed sample is degenerate (coincident points) -> zero.
        assert_eq!(target_function(&GiSample::ZERO), 0.0);
    }

    #[test]
    fn update_accepts_first_valid_candidate_even_at_u_one() {
        let mut r = Reservoir::<Tagged>::new();
        assert!(r.is_empty());
        // u == 1 must still accept the very first candidate.
        assert!(r.update(Tagged(7), 2.5, 1.0));
        assert_eq!(r.sample(), Some(Tagged(7)));
        assert_eq!(r.confidence(), 1.0);
        assert_eq!(r.weight_sum(), 2.5);
    }

    #[test]
    fn update_discards_degenerate_weights_without_touching_state() {
        let mut r = Reservoir::<Tagged>::new();
        assert!(!r.update(Tagged(1), 0.0, 0.0));
        assert!(!r.update(Tagged(2), -3.0, 0.0));
        assert!(!r.update(Tagged(3), f32::NAN, 0.0));
        assert!(!r.update(Tagged(4), f32::INFINITY, 0.0));
        assert!(r.is_empty());
        assert_eq!(r.confidence(), 0.0);
        assert_eq!(r.weight_sum(), 0.0);
    }

    #[test]
    fn update_selection_frequency_tracks_weight_ratio() {
        // Two candidates with weights 1 and 3: sweeping the uniform over a dense
        // deterministic grid, candidate B should win ~3x as often as A.
        let n = 100_000usize;
        let mut wins_b = 0usize;
        for i in 0..n {
            let u0 = (i as f32 + 0.5) / n as f32;
            // Decorrelate the second uniform with a golden-ratio stride.
            let u1 = ((i as f32 * 0.618_034) + 0.5).fract();
            let mut r = Reservoir::<Tagged>::new();
            r.update(Tagged(0), 1.0, u0); // A
            if r.update(Tagged(1), 3.0, u1) {
                // B accepted at the second step...
            }
            if r.sample() == Some(Tagged(1)) {
                wins_b += 1;
            }
        }
        let frac_b = wins_b as f32 / n as f32;
        // Expected 3 / (1 + 3) = 0.75.
        assert!((frac_b - 0.75).abs() < 0.01, "frac_b={frac_b}");
    }

    #[test]
    fn single_sample_finalize_matches_closed_form() {
        // One candidate, confidence m = 1, so W = (w_sum / m) / p_hat.
        // When the resampling weight equals the target p_hat, W collapses to 1.
        let p_hat = 4.0;
        let mut r = Reservoir::<Tagged>::new();
        r.update(Tagged(0), p_hat, 0.3);
        r.finalize_weight(p_hat);
        assert!((r.contribution_weight() - 1.0).abs() < 1e-6);

        // A resampling weight twice the target doubles W: W = (2 p_hat) / p_hat.
        let mut r2 = Reservoir::<Tagged>::new();
        r2.update(Tagged(0), 2.0 * p_hat, 0.3);
        r2.finalize_weight(p_hat);
        assert!((r2.contribution_weight() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn empty_or_degenerate_finalize_is_zero() {
        let mut empty = Reservoir::<Tagged>::new();
        empty.finalize_weight(1.0);
        assert_eq!(empty.contribution_weight(), 0.0);

        let mut r = Reservoir::<Tagged>::new();
        r.update(Tagged(0), 1.0, 0.0);
        // Non-positive / non-finite target density -> W = 0.
        r.finalize_weight(0.0);
        assert_eq!(r.contribution_weight(), 0.0);
        r.finalize_weight(f32::NAN);
        assert_eq!(r.contribution_weight(), 0.0);
    }

    #[test]
    fn cap_confidence_bounds_history() {
        let mut r = Reservoir::<Tagged>::new();
        for _ in 0..50 {
            r.update(Tagged(0), 1.0, 0.5);
        }
        assert_eq!(r.confidence(), 50.0);
        r.cap_confidence(20.0);
        assert_eq!(r.confidence(), 20.0);
        // A negative cap is ignored.
        r.cap_confidence(-1.0);
        assert_eq!(r.confidence(), 20.0);
        // A cap above the current value is a no-op.
        r.cap_confidence(1000.0);
        assert_eq!(r.confidence(), 20.0);
    }

    #[test]
    fn merge_accumulates_confidence_and_can_select_neighbour() {
        // Canonical reservoir with one sample.
        let mut canonical = Reservoir::<Tagged>::new();
        canonical.update(Tagged(0), 1.0, 0.5);
        canonical.finalize_weight(1.0);

        // Neighbour reservoir with a strong finalised weight.
        let mut neighbour = Reservoir::<Tagged>::new();
        neighbour.update(Tagged(9), 1.0, 0.5);
        neighbour.m = 8.0;
        neighbour.finalize_weight(1.0);

        // Merge with a large induced weight (u = 0 forces selection).
        let selected = canonical.merge(&neighbour, 10.0, 0.0);
        assert!(selected);
        assert_eq!(canonical.sample(), Some(Tagged(9)));
        // Confidence is the sum of both histories.
        assert_eq!(canonical.confidence(), 1.0 + 8.0);
    }

    #[test]
    fn merge_still_accumulates_confidence_when_weight_is_zero() {
        let mut canonical = Reservoir::<Tagged>::new();
        canonical.update(Tagged(0), 2.0, 0.5);

        let mut neighbour = Reservoir::<Tagged>::new();
        neighbour.update(Tagged(9), 1.0, 0.5);
        neighbour.m = 5.0;
        // Zero target pdf in this domain -> no selection, but confidence grows.
        let selected = canonical.merge(&neighbour, 0.0, 0.0);
        assert!(!selected);
        assert_eq!(canonical.sample(), Some(Tagged(0)));
        assert_eq!(canonical.confidence(), 1.0 + 5.0);
    }

    #[test]
    fn merge_ignores_empty_history() {
        let mut canonical = Reservoir::<Tagged>::new();
        canonical.update(Tagged(0), 1.0, 0.5);
        let empty = Reservoir::<Tagged>::new();
        assert!(!canonical.merge(&empty, 10.0, 0.0));
        assert_eq!(canonical.confidence(), 1.0);
        assert_eq!(canonical.sample(), Some(Tagged(0)));
    }

    #[test]
    fn balance_heuristic_partition_of_unity() {
        let pdfs = [1.0f32, 3.0, 6.0];
        let counts = [1.0f32, 1.0, 1.0];
        let w: f32 = (0..3).map(|i| balance_heuristic(&pdfs, &counts, i)).sum();
        assert!((w - 1.0).abs() < 1e-6, "sum={w}");
        // Each weight is in [0, 1] and proportional to its pdf here.
        assert!((balance_heuristic(&pdfs, &counts, 0) - 0.1).abs() < 1e-6);
        assert!((balance_heuristic(&pdfs, &counts, 2) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn balance_heuristic_respects_confidence_counts() {
        // Equal pdfs but unequal counts -> weight proportional to counts.
        let pdfs = [2.0f32, 2.0];
        let counts = [1.0f32, 3.0];
        assert!((balance_heuristic(&pdfs, &counts, 0) - 0.25).abs() < 1e-6);
        assert!((balance_heuristic(&pdfs, &counts, 1) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn balance_heuristic_degenerate_inputs_are_zero() {
        assert_eq!(balance_heuristic(&[], &[], 0), 0.0);
        assert_eq!(balance_heuristic(&[0.0, 0.0], &[1.0, 1.0], 0), 0.0);
        assert_eq!(balance_heuristic(&[1.0], &[1.0], 5), 0.0);
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut r = Reservoir::<Tagged>::new();
            r.update(Tagged(1), 1.0, 0.2);
            r.update(Tagged(2), 2.0, 0.7);
            r.update(Tagged(3), 0.5, 0.9);
            r.cap_confidence(10.0);
            r.finalize_weight(1.5);
            r
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a.sample(), b.sample());
        assert_eq!(a.contribution_weight(), b.contribution_weight());
    }
}
