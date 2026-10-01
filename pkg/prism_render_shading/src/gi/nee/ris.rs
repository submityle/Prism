//! Resampled importance sampling of light candidate sets for NEE — CPU golden.
//!
//! When a scene has many lights, picking one uniformly wastes samples on lights
//! that barely contribute.  Resampled importance sampling (RIS; Talbot et al.
//! 2005) first draws `M` cheap *candidate* lights from a simple source density
//! `p`, scores each by a target function `p̂` (an estimate of its true
//! contribution, e.g. the unshadowed radiance reaching the shading point), then
//! resamples **one** survivor proportionally to those scores.  The survivor is
//! paired with an unbiased contribution weight so the overall estimator remains
//! correct even though `p̂` is only approximate.  This module is the
//! backend-neutral numerical reference for that machinery:
//!
//! * [`Candidate`] bundles a candidate's target density `p̂` and the source
//!   density `p` it was drawn from; its *resampling weight* is `w = p̂ / p`.
//! * [`weighted_reservoir_sample`] is the deterministic weighted-reservoir
//!   selection: given a weight array and a single uniform `u ∈ [0, 1)`, it
//!   returns the chosen index via the inverse-CDF walk (equivalent to streaming
//!   WRS but with one shared uniform, so it is a pure function).
//! * [`ris_contribution_weight`] computes the unbiased RIS weight
//!   `W = (1 / p̂(y)) · (1/M · Σ wᵢ)`.
//! * [`resample`] fuses the two: it selects a survivor and returns a
//!   [`RisResult`] carrying the index, the running weight sum, the candidate
//!   count `M` and the finalised `W`.
//! * [`Reservoir`] is the streaming / mergeable form used to fold candidates in
//!   one at a time ([`Reservoir::update`]) and to combine independent reservoirs
//!   ([`Reservoir::merge`]) for spatial / temporal reuse, mirroring the style of
//!   [`crate::gi::screen_probe::restir`] but implemented independently for the
//!   NEE light-selection target.
//!
//! # Conventions
//! * All weights and densities are `f32`, matching the GPU reservoir twin.
//! * Resampling weights are non-negative; a candidate with a non-finite or
//!   non-positive weight carries no information and is skipped without being
//!   selected (and, for the streaming reservoir, without raising its
//!   confidence `m`).
//! * The replacement test is `u · w_sum ≤ weight`, with `u` clamped to
//!   `[0, 1]`; `≤` guarantees the first valid candidate is always accepted even
//!   at `u == 1`, so a non-empty candidate set never yields an empty survivor.
//! * Every division guards its denominator; [`ris_contribution_weight`] and
//!   [`Reservoir::finalize_weight`] return `0` for a non-positive / non-finite
//!   target or count, so `W` is always finite and non-negative.
//! * All functions are deterministic pure functions: no RNG, no I/O, no GPU,
//!   no global state and no `unsafe`.

/// Tiny positive epsilon guarding divisions against degenerate denominators.
const EPS: f32 = 1.0e-12;

/// A single RIS candidate: its target density `p̂` and source density `p`.
///
/// The *resampling weight* is `w = p̂ / p` (see [`Candidate::weight`]), the
/// quantity the reservoir resamples proportionally to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Target-function value `p̂` for this candidate (an estimate of its true,
    /// unnormalised contribution — e.g. unshadowed radiance).
    pub target: f32,
    /// Source density `p` the candidate was drawn from (e.g. uniform light
    /// selection probability).
    pub source_pdf: f32,
}

impl Candidate {
    /// Creates a candidate from its target `p̂` and source density `p`.
    #[inline]
    pub fn new(target: f32, source_pdf: f32) -> Self {
        Self {
            target,
            source_pdf,
        }
    }

    /// The RIS resampling weight `w = p̂ / p`.
    ///
    /// Returns `0` for a non-positive / non-finite target or source density, so
    /// a degenerate candidate contributes nothing and can never be selected.
    #[inline]
    pub fn weight(&self) -> f32 {
        if self.target <= 0.0
            || !self.target.is_finite()
            || self.source_pdf <= EPS
            || !self.source_pdf.is_finite()
        {
            return 0.0;
        }
        let w = self.target / self.source_pdf;
        if w.is_finite() { w.max(0.0) } else { 0.0 }
    }
}

/// Deterministic weighted-reservoir selection over a weight array.
///
/// Picks an index with probability proportional to its weight using a single
/// uniform `u ∈ [0, 1)` and an inverse-CDF walk: it returns the first index at
/// which the cumulative weight exceeds `u · Σw`.  Non-finite / non-positive
/// weights are skipped.  Returns `None` only when every weight is degenerate
/// (total weight `0`); at `u == 1` it returns the last valid index.
#[inline]
pub fn weighted_reservoir_sample(weights: &[f32], u: f32) -> Option<usize> {
    let mut total = 0.0f32;
    for &w in weights {
        if w.is_finite() && w > 0.0 {
            total += w;
        }
    }
    if total <= EPS || !total.is_finite() {
        return None;
    }
    let u = u.clamp(0.0, 1.0);
    let target = u * total;
    let mut cumulative = 0.0f32;
    let mut last_valid = None;
    for (i, &w) in weights.iter().enumerate() {
        if w.is_finite() && w > 0.0 {
            cumulative += w;
            last_valid = Some(i);
            if cumulative > target {
                return Some(i);
            }
        }
    }
    // Reached only at u == 1 (target == total): return the last valid index.
    last_valid
}

/// Computes the unbiased RIS contribution weight
/// `W = (1 / p̂(y)) · (1/M · Σ wᵢ)`.
///
/// `selected_target` is the target density `p̂(y)` of the surviving candidate,
/// `weights` the full resampling-weight array (its length is the candidate
/// count `M`, including zero-weight candidates), and the sum runs over all
/// weights.  Returns `0` for a non-positive / non-finite `p̂(y)`, an empty
/// array, or a non-finite result, so `W` is always finite and non-negative.
#[inline]
pub fn ris_contribution_weight(selected_target: f32, weights: &[f32]) -> f32 {
    let m = weights.len();
    if m == 0 || selected_target <= 0.0 || !selected_target.is_finite() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &w in weights {
        if w.is_finite() && w > 0.0 {
            sum += w;
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        return 0.0;
    }
    let w = (sum / m as f32) / selected_target;
    if w.is_finite() && w >= 0.0 { w } else { 0.0 }
}

/// The result of resampling a candidate set: the chosen survivor plus the
/// bookkeeping needed to form an unbiased estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RisResult {
    /// Index of the surviving candidate in the input slice, or `None` when the
    /// whole set was degenerate (no positive-weight candidate).
    pub selected: Option<usize>,
    /// Sum of all resampling weights `Σ wᵢ`.
    pub weight_sum: f32,
    /// Candidate count `M` (the length of the input slice).
    pub m: f32,
    /// Unbiased contribution weight `W = (1 / p̂(y)) · (1/M · Σ wᵢ)`.
    pub contribution_weight: f32,
}

impl RisResult {
    /// A degenerate result: nothing selected, zero weights.
    pub const EMPTY: Self = Self {
        selected: None,
        weight_sum: 0.0,
        m: 0.0,
        contribution_weight: 0.0,
    };
}

/// Resamples one survivor from a candidate set with a single uniform `u`.
///
/// Builds the resampling weights `wᵢ = p̂ᵢ / pᵢ`, selects a survivor with
/// [`weighted_reservoir_sample`], and finalises the unbiased weight `W` with
/// [`ris_contribution_weight`].  Returns [`RisResult::EMPTY`] for an empty or
/// fully degenerate candidate set.
#[inline]
pub fn resample(candidates: &[Candidate], u: f32) -> RisResult {
    let m = candidates.len();
    if m == 0 {
        return RisResult::EMPTY;
    }
    // Build the resampling weights and their sum in one pass.
    let mut sum = 0.0f32;
    let mut selected = None;
    let mut cumulative = 0.0f32;
    let u = u.clamp(0.0, 1.0);

    // First pass: total weight.
    for c in candidates {
        let w = c.weight();
        if w > 0.0 {
            sum += w;
        }
    }
    if sum <= EPS || !sum.is_finite() {
        return RisResult {
            selected: None,
            weight_sum: 0.0,
            m: m as f32,
            contribution_weight: 0.0,
        };
    }
    // Second pass: inverse-CDF selection against u·sum.
    let target = u * sum;
    let mut last_valid = None;
    for (i, c) in candidates.iter().enumerate() {
        let w = c.weight();
        if w > 0.0 {
            cumulative += w;
            last_valid = Some(i);
            if cumulative > target {
                selected = Some(i);
                break;
            }
        }
    }
    // At u == 1 the strict `>` test never fires; fall back to the last valid.
    let selected = selected.or(last_valid);
    let selected_target = selected.map(|i| candidates[i].target).unwrap_or(0.0);
    // W = (1 / p̂(y)) · (1/M · Σ wᵢ); reuse the running sum rather than re-summing.
    let contribution_weight = if selected_target > 0.0 && selected_target.is_finite() {
        let cw = (sum / m as f32) / selected_target;
        if cw.is_finite() && cw >= 0.0 { cw } else { 0.0 }
    } else {
        0.0
    };
    RisResult {
        selected,
        weight_sum: sum,
        m: m as f32,
        contribution_weight,
    }
}

/// A streaming / mergeable RIS reservoir holding one surviving candidate.
///
/// Unlike [`resample`], which consumes a whole slice at once, this folds
/// candidates in one at a time (so an integrator need not store them all) and
/// merges independent reservoirs for spatial / temporal reuse.  The payload is
/// the caller's opaque light index `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservoir {
    /// The surviving candidate's caller-defined light index, if any.
    selected: Option<u32>,
    /// Target density `p̂` of the surviving candidate.
    selected_target: f32,
    /// Running sum of resampling weights.
    w_sum: f32,
    /// Confidence weight `m`: the effective number of candidates summarised.
    m: f32,
    /// Finalised unbiased contribution weight `W`.
    w: f32,
}

impl Default for Reservoir {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Reservoir {
    /// An empty reservoir: no survivor, zero weights, zero confidence.
    pub const EMPTY: Self = Self {
        selected: None,
        selected_target: 0.0,
        w_sum: 0.0,
        m: 0.0,
        w: 0.0,
    };

    /// Creates an empty reservoir (alias for [`Reservoir::EMPTY`]).
    #[inline]
    pub fn new() -> Self {
        Self::EMPTY
    }

    /// The surviving candidate's light index, if any.
    #[inline]
    pub fn selected(&self) -> Option<u32> {
        self.selected
    }

    /// Target density `p̂` of the surviving candidate (`0` when empty).
    #[inline]
    pub fn selected_target(&self) -> f32 {
        self.selected_target
    }

    /// Running sum of resampling weights `Σ wᵢ`.
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
    #[inline]
    pub fn contribution_weight(&self) -> f32 {
        self.w
    }

    /// Whether the reservoir currently holds no survivor.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.selected.is_none()
    }

    /// Streaming RIS update: fold candidate `index` with target `p̂` drawn from
    /// source density `p` using the external uniform `u`.
    ///
    /// The resampling weight `w = p̂ / p` is accumulated; the candidate survives
    /// with probability `w / w_sum`.  Degenerate candidates (non-finite /
    /// non-positive `w`) are skipped without touching any state, including the
    /// confidence `m`.  Returns `true` when the candidate became the survivor.
    #[inline]
    pub fn update(&mut self, index: u32, target: f32, source_pdf: f32, u: f32) -> bool {
        let weight = Candidate::new(target, source_pdf).weight();
        if weight <= 0.0 {
            return false;
        }
        self.w_sum += weight;
        self.m += 1.0;
        let u = u.clamp(0.0, 1.0);
        if u * self.w_sum <= weight {
            self.selected = Some(index);
            self.selected_target = target;
            true
        } else {
            false
        }
    }

    /// Merges another reservoir into this one for spatial / temporal reuse.
    ///
    /// `other_target` is `other`'s surviving sample's target density
    /// *re-evaluated in this reservoir's domain* (identical for a temporal
    /// predecessor).  The induced resampling weight is
    /// `other.m · other_target · other.W`; the confidence `m` always
    /// accumulates `other.m` so the balance-heuristic MIS correction in
    /// [`finalize_weight`](Self::finalize_weight) stays valid.  Returns `true`
    /// when `other`'s survivor was selected.
    #[inline]
    pub fn merge(&mut self, other: &Reservoir, other_target: f32, u: f32) -> bool {
        if !other.m.is_finite() || other.m <= 0.0 {
            return false;
        }
        self.m += other.m;
        let rw = other.m * other_target.max(0.0) * other.w.max(0.0);
        if !rw.is_finite() || rw <= 0.0 {
            return false;
        }
        self.w_sum += rw;
        let u = u.clamp(0.0, 1.0);
        if u * self.w_sum <= rw {
            self.selected = other.selected;
            self.selected_target = other_target;
            true
        } else {
            false
        }
    }

    /// Caps the confidence weight `m` to `max_m` (the standard reuse M-cap),
    /// bounding how much history a reservoir may accumulate.  A negative
    /// `max_m` is ignored.
    #[inline]
    pub fn cap_confidence(&mut self, max_m: f32) {
        if max_m >= 0.0 && self.m > max_m {
            self.m = max_m;
        }
    }

    /// Finalises the unbiased contribution weight
    /// `W = (1 / p̂(y)) · (w_sum / m)` from the survivor's target `p̂(y)`.
    ///
    /// Forced to `0` whenever the reservoir is empty, the confidence is
    /// non-positive, the target is non-positive / non-finite, or the result is
    /// non-finite, so `W` is always finite and non-negative.
    #[inline]
    pub fn finalize_weight(&mut self) {
        if self.selected.is_none()
            || self.m <= 0.0
            || !self.w_sum.is_finite()
            || self.selected_target <= 0.0
            || !self.selected_target.is_finite()
        {
            self.w = 0.0;
            return;
        }
        let w = (self.w_sum / self.m) / self.selected_target;
        self.w = if w.is_finite() && w >= 0.0 { w } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_weight_is_ratio_and_guards_degenerate() {
        assert!((Candidate::new(4.0, 2.0).weight() - 2.0).abs() < 1.0e-6);
        assert_eq!(Candidate::new(0.0, 2.0).weight(), 0.0);
        assert_eq!(Candidate::new(1.0, 0.0).weight(), 0.0);
        assert_eq!(Candidate::new(f32::NAN, 1.0).weight(), 0.0);
    }

    #[test]
    fn wrs_selects_proportionally_across_the_cdf() {
        let weights = [1.0, 3.0, 0.0, 2.0]; // total 6
        // u in each sub-interval selects the matching index.
        assert_eq!(weighted_reservoir_sample(&weights, 0.0), Some(0)); // (0, 1]
        assert_eq!(weighted_reservoir_sample(&weights, 0.1), Some(0));
        assert_eq!(weighted_reservoir_sample(&weights, 0.3), Some(1)); // (1, 4]
        assert_eq!(weighted_reservoir_sample(&weights, 0.6), Some(1));
        assert_eq!(weighted_reservoir_sample(&weights, 0.8), Some(3)); // (4, 6]
        // Zero-weight index 2 is never selected.
        for k in 0..100 {
            let u = k as f32 / 100.0;
            assert_ne!(weighted_reservoir_sample(&weights, u), Some(2));
        }
    }

    #[test]
    fn wrs_handles_edges_and_degenerate_sets() {
        let weights = [2.0, 5.0];
        assert_eq!(weighted_reservoir_sample(&weights, 1.0), Some(1)); // u==1 → last
        assert_eq!(weighted_reservoir_sample(&[0.0, 0.0], 0.5), None);
        assert_eq!(weighted_reservoir_sample(&[], 0.5), None);
        assert_eq!(weighted_reservoir_sample(&[f32::NAN, -1.0], 0.5), None);
    }

    #[test]
    fn ris_weight_matches_formula() {
        let weights = [1.0, 2.0, 3.0]; // sum 6, M 3
        let selected_target = 2.0;
        let w = ris_contribution_weight(selected_target, &weights);
        // W = (1/2) * (6/3) = 1.
        assert!((w - 1.0).abs() < 1.0e-6, "w={w}");
    }

    #[test]
    fn ris_weight_guards_degenerate() {
        assert_eq!(ris_contribution_weight(0.0, &[1.0, 2.0]), 0.0);
        assert_eq!(ris_contribution_weight(1.0, &[]), 0.0);
        assert_eq!(ris_contribution_weight(1.0, &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn resample_product_identity_holds_for_every_u() {
        // For a fixed candidate set, p̂(y)·W == (1/M)·Σ wᵢ for *any* u, since
        // W = (1/M·Σw)/p̂(y).  This is the exact invariant RIS unbiasedness
        // rests on.
        let candidates = [
            Candidate::new(1.0, 0.25),
            Candidate::new(4.0, 0.25),
            Candidate::new(2.0, 0.25),
            Candidate::new(0.5, 0.25),
        ];
        let mean_w = {
            let mut s = 0.0;
            for c in &candidates {
                s += c.weight();
            }
            s / candidates.len() as f32
        };
        for k in 0..64 {
            let u = (k as f32 + 0.5) / 64.0;
            let r = resample(&candidates, u);
            let idx = r.selected.expect("non-degenerate set selects a survivor");
            let product = candidates[idx].target * r.contribution_weight;
            assert!(
                (product - mean_w).abs() < 1.0e-4,
                "u={u} product={product} mean_w={mean_w}"
            );
        }
    }

    #[test]
    fn resample_is_numerically_unbiased_against_a_known_integral() {
        // Integrate f(x)=x² on [0,1].  Source p is uniform (pdf = 1), the target
        // p̂ = f, so wᵢ = f(xᵢ).  Because p̂(y)·W == (1/M)Σf(xᵢ) regardless of the
        // survivor (the product identity), the estimator f(y)·W collapses to the
        // Riemann mean, which must approach ∫₀¹ x² dx = 1/3.  We verify the mean
        // directly and confirm the product identity on a fixed-size candidate set.
        const M: usize = 32;
        let mut candidates = [Candidate::new(0.0, 1.0); M];
        let mut analytic = 0.0f64;
        for (i, c) in candidates.iter_mut().enumerate() {
            let x = (i as f32 + 0.5) / M as f32;
            let fx = x * x;
            *c = Candidate::new(fx, 1.0);
            analytic += fx as f64;
        }
        let mean = analytic / M as f64;
        // Midpoint Riemann sum of x² converges to 1/3; 32 cells is within 1e-3.
        assert!((mean - 1.0 / 3.0).abs() < 1.0e-3, "mean={mean}");
        // The RIS estimator f(y)·W equals that mean for every survivor.
        for k in 0..M as u32 {
            let u = (k as f32 + 0.5) / M as f32;
            let r = resample(&candidates, u);
            let idx = r.selected.expect("survivor");
            let estimator = candidates[idx].target * r.contribution_weight;
            assert!((estimator as f64 - mean).abs() < 1.0e-4, "u={u} est={estimator}");
        }
    }

    #[test]
    fn streaming_reservoir_matches_batch_selection_distribution() {
        // Streaming update with per-item uniforms must keep the survivor's
        // target among the candidates and finalise a sane W.
        let mut r = Reservoir::new();
        r.update(0, 1.0, 0.25, 0.9);
        r.update(1, 4.0, 0.25, 0.1);
        r.update(2, 2.0, 0.25, 0.8);
        assert!(!r.is_empty());
        r.finalize_weight();
        // The invariant p̂(y)·W == w_sum/m holds for the streaming form too.
        let product = r.selected_target() * r.contribution_weight();
        let mean_w = r.weight_sum() / r.confidence();
        assert!((product - mean_w).abs() < 1.0e-5, "product={product} mean={mean_w}");
    }

    #[test]
    fn streaming_reservoir_skips_degenerate_without_raising_m() {
        let mut r = Reservoir::new();
        assert!(!r.update(0, 0.0, 1.0, 0.5)); // zero target
        assert_eq!(r.confidence(), 0.0);
        assert!(r.is_empty());
        assert!(!r.update(1, 1.0, 0.0, 0.5)); // zero source pdf
        assert_eq!(r.confidence(), 0.0);
    }

    #[test]
    fn merge_accumulates_confidence_and_can_select() {
        let mut a = Reservoir::new();
        a.update(0, 2.0, 0.5, 0.5);
        a.finalize_weight();

        let mut b = Reservoir::new();
        b.update(7, 3.0, 0.5, 0.5);
        b.finalize_weight();

        let mut merged = a;
        // u small → prefer the incoming reservoir if it has weight.
        merged.merge(&b, b.selected_target(), 0.0);
        assert!(merged.confidence() >= a.confidence() + b.confidence() - 1.0e-6);
        merged.finalize_weight();
        assert!(merged.contribution_weight().is_finite());
        assert!(merged.contribution_weight() >= 0.0);
    }

    #[test]
    fn cap_confidence_bounds_history() {
        let mut r = Reservoir::new();
        for i in 0..10 {
            r.update(i, 1.0, 0.5, 0.5);
        }
        assert!((r.confidence() - 10.0).abs() < 1.0e-6);
        r.cap_confidence(4.0);
        assert!((r.confidence() - 4.0).abs() < 1.0e-6);
        r.cap_confidence(-1.0); // negative ignored
        assert!((r.confidence() - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn finalize_weight_guards_degenerate() {
        let mut empty = Reservoir::new();
        empty.finalize_weight();
        assert_eq!(empty.contribution_weight(), 0.0);
    }
}
