//! Near-/far-field scatter adaptation + hair BSDF importance-sampling weights.
//!
//! A production hair renderer evaluates the three `Marschner`/`Chiang` lobes
//! (`R` surface reflection, `TT` transmission, `TRT` internal reflection) under
//! two regimes depending on how large a fibre is on screen (`d'Eon` 2011
//! energy-conserving near-/far-field split):
//!
//! * **Near field** (close-up, projected fibre width `>= near_px`): per-fibre
//!   scattering with the full lobe separation preserved — maximum fidelity.
//! * **Far field** (distant, projected fibre width `<= far_px`): an analytic
//!   aggregate that widens the lobes to integrate many sub-pixel fibres at once,
//!   which removes variance/aliasing and keeps the colour un-biased.
//!
//! Between the two thresholds this module produces a *continuous* blend factor so
//! the §4 LOD ladder crosses the near->far boundary without a pop. The blend
//! also drives a far-field roughness gain (lobes widen as the footprint shrinks).
//!
//! The second job of this module is **BSDF importance sampling** for the RT /
//! path-traced path (and its denoiser): given per-fibre optical inputs it maps
//! the lobes to a normalised sampling pdf (`R`/`TT`/`TRT` weights that sum to
//! `1`), picks a lobe from a canonical uniform sample, and allocates stratified
//! samples across the lobes with the matching Monte-Carlo weights. Importance
//! sampling the lobes in proportion to their energy is what keeps the estimator
//! low-variance at a fixed sample budget.
//!
//! Like [`crate::hair::melanin`] this is a *material-independent, deterministic*
//! architecture-side mapping: array in, array out, golden-comparable,
//! panic-free. It performs **no** transcendental math. In particular the real
//! Beer-Lambert transmittance `exp(-sigma_a)` lives in the shading closure (the
//! only place the exponential belongs); here we only need a *monotone,
//! energy-ordered* proxy to rank the lobes, so we use the rational attenuation
//! `1 / (1 + sigma_a)` (equal to `1` at zero absorption, decreasing, in `[0,1]`).
//! The absolute lobe energies are never shaded from these numbers — only their
//! *relative ordering* feeds the sampling pdf — so the rational proxy is exact
//! for the purpose it serves and needs no `libm` determinism shim.

use alloc::vec::Vec;

/// Number of hair BSDF lobes: `R` (0), `TT` (1), `TRT` (2).
pub const LOBE_COUNT: usize = 3;

/// Index of the `R` (surface reflection) lobe.
pub const LOBE_R: usize = 0;
/// Index of the `TT` (transmission) lobe.
pub const LOBE_TT: usize = 1;
/// Index of the `TRT` (internal reflection) lobe.
pub const LOBE_TRT: usize = 2;

const EPS: f32 = 1e-6;

/// Which scatter regime a fibre falls into for its current screen footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterRegime {
    /// Pure per-fibre near-field (footprint at or above the near threshold).
    Near,
    /// Pure analytic far-field (footprint at or below the far threshold).
    Far,
    /// Continuous blend in the transition band (anti-pop).
    Blended,
}

/// Screen-space thresholds (projected fibre width, in pixels) that bracket the
/// near->far scatter transition. Invariant after [`ScatterLodThresholds::sanitized`]:
/// `near_px >= far_px >= 0`, both finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterLodThresholds {
    /// At or above this width the fibre is pure near-field.
    pub near_px: f32,
    /// At or below this width the fibre is pure far-field.
    pub far_px: f32,
}

impl Default for ScatterLodThresholds {
    fn default() -> Self {
        Self {
            near_px: 1.0,
            far_px: 0.25,
        }
    }
}

impl ScatterLodThresholds {
    /// Thresholds from explicit near/far pixel widths.
    #[must_use]
    pub const fn new(near_px: f32, far_px: f32) -> Self {
        Self { near_px, far_px }
    }

    /// Clamp non-finite/negative widths to `0` and enforce `near_px >= far_px`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let far = sanitize_nonneg(self.far_px);
        let near_raw = sanitize_nonneg(self.near_px);
        let near = near_raw.max(far);
        Self {
            near_px: near,
            far_px: far,
        }
    }
}

/// Continuous far-field blend for a projected fibre width: `0` = pure near-field,
/// `1` = pure far-field, linearly ramped across the transition band. Monotone
/// non-increasing in `fiber_width_px`, so wider (closer) fibres read as more
/// near-field. A degenerate band (`near_px == far_px`) is a hard step at that
/// width (still panic-free). Non-finite widths sanitise to `0` (treated as the
/// finest / most far-field footprint, i.e. blend `1`).
#[must_use]
pub fn scatter_blend(fiber_width_px: f32, thresholds: ScatterLodThresholds) -> f32 {
    let t = thresholds.sanitized();
    let w = if fiber_width_px.is_finite() {
        fiber_width_px.max(0.0)
    } else {
        0.0
    };
    if w >= t.near_px {
        return 0.0;
    }
    if w <= t.far_px {
        return 1.0;
    }
    let span = t.near_px - t.far_px;
    if span <= EPS {
        // Degenerate band: hard step (we are strictly between equal bounds only
        // when span > 0, so this is a defensive fall-through).
        return 1.0;
    }
    // w in (far_px, near_px): blend 1 at far edge down to 0 at near edge.
    ((t.near_px - w) / span).clamp(0.0, 1.0)
}

/// Classify the scatter regime from the blend factor of a fibre width.
#[must_use]
pub fn scatter_regime(fiber_width_px: f32, thresholds: ScatterLodThresholds) -> ScatterRegime {
    let b = scatter_blend(fiber_width_px, thresholds);
    if b <= EPS {
        ScatterRegime::Near
    } else if b >= 1.0 - EPS {
        ScatterRegime::Far
    } else {
        ScatterRegime::Blended
    }
}

/// Far-field roughness gain: lobes widen by `1 + blend * max_gain` as the
/// footprint shrinks, which integrates sub-pixel fibres and cuts aliasing. The
/// blend is clamped to `[0,1]` and a negative/non-finite `max_gain` is treated as
/// `0` (no widening). Always `>= 1`.
#[must_use]
pub fn far_field_roughness_gain(blend: f32, max_gain: f32) -> f32 {
    let b = clamp01(blend);
    let g = sanitize_nonneg(max_gain);
    1.0 + b * g
}

/// Per-fibre optical inputs used to rank the three lobes by energy. `fresnel` is
/// the surface reflectance proxy in `[0,1]`; `absorption` is the (non-negative)
/// `sigma_a * path-length` proxy, e.g. derived from [`crate::hair::melanin`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobeOpticalInput {
    /// Surface reflectance (drives the `R` lobe); clamped to `[0,1]`.
    pub fresnel: f32,
    /// Absorption-path proxy (attenuates transmission lobes); clamped `>= 0`.
    pub absorption: f32,
}

impl LobeOpticalInput {
    /// Optical inputs from explicit fresnel / absorption values.
    #[must_use]
    pub const fn new(fresnel: f32, absorption: f32) -> Self {
        Self {
            fresnel,
            absorption,
        }
    }

    /// Clamp `fresnel` to `[0,1]` and `absorption` to a finite `>= 0` value.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            fresnel: clamp01(self.fresnel),
            absorption: sanitize_nonneg(self.absorption),
        }
    }
}

/// Monotone rational stand-in for Beer-Lambert transmittance `exp(-absorption)`:
/// `1 / (1 + absorption)`. Equals `1` at zero absorption, decreases, stays in
/// `(0,1]`. Used only to *order* the lobes, never shaded directly.
#[must_use]
fn transmittance_proxy(absorption: f32) -> f32 {
    1.0 / (1.0 + sanitize_nonneg(absorption))
}

/// Un-normalised, non-negative lobe energies for `R`/`TT`/`TRT`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobeWeights {
    /// `R` surface-reflection energy.
    pub r: f32,
    /// `TT` single-transmission energy.
    pub tt: f32,
    /// `TRT` transmit-reflect-transmit energy.
    pub trt: f32,
}

impl LobeWeights {
    /// Weights from explicit per-lobe energies.
    #[must_use]
    pub const fn new(r: f32, tt: f32, trt: f32) -> Self {
        Self { r, tt, trt }
    }

    /// `[R, TT, TRT]` as an array.
    #[must_use]
    pub fn as_array(&self) -> [f32; LOBE_COUNT] {
        [self.r, self.tt, self.trt]
    }

    /// Sum of the (sanitised, non-negative) lobe energies.
    #[must_use]
    pub fn sum(&self) -> f32 {
        sanitize_nonneg(self.r) + sanitize_nonneg(self.tt) + sanitize_nonneg(self.trt)
    }
}

/// Derive un-normalised lobe energies from per-fibre optical inputs.
///
/// * `R   = F`                       (surface reflection)
/// * `TT  = (1-F)^2 * T`             (in, out — two interface transmissions)
/// * `TRT = (1-F)^2 * F * T^2`       (in, internal reflect, out)
///
/// where `F` is the fresnel proxy and `T` the rational transmittance proxy. All
/// terms are products of non-negative factors, so every energy is `>= 0`.
#[must_use]
pub fn lobe_weights(input: LobeOpticalInput) -> LobeWeights {
    let i = input.sanitized();
    let f = i.fresnel;
    let one_minus_f = 1.0 - f;
    let two_transmit = one_minus_f * one_minus_f;
    let t = transmittance_proxy(i.absorption);
    LobeWeights {
        r: f,
        tt: two_transmit * t,
        trt: two_transmit * f * (t * t),
    }
}

/// Normalised lobe sampling pdf: non-negative, sums to `1`. An all-zero energy
/// stack falls back to a uniform `1/3` per lobe (so sampling never divides by
/// zero and never biases toward a dead lobe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobePdf {
    /// Probability of sampling the `R` lobe.
    pub r: f32,
    /// Probability of sampling the `TT` lobe.
    pub tt: f32,
    /// Probability of sampling the `TRT` lobe.
    pub trt: f32,
}

impl LobePdf {
    /// `[P(R), P(TT), P(TRT)]` as an array.
    #[must_use]
    pub fn as_array(&self) -> [f32; LOBE_COUNT] {
        [self.r, self.tt, self.trt]
    }

    /// Probability of lobe `i` (`0..LOBE_COUNT`); out-of-range yields `0`.
    #[must_use]
    pub fn get(&self, i: usize) -> f32 {
        match i {
            LOBE_R => self.r,
            LOBE_TT => self.tt,
            LOBE_TRT => self.trt,
            _ => 0.0,
        }
    }
}

/// Normalise lobe energies into a sampling pdf (uniform fallback for an all-zero
/// stack). Negative/non-finite energies are clamped to `0` first.
#[must_use]
pub fn lobe_pdf(weights: LobeWeights) -> LobePdf {
    let r = sanitize_nonneg(weights.r);
    let tt = sanitize_nonneg(weights.tt);
    let trt = sanitize_nonneg(weights.trt);
    let sum = r + tt + trt;
    if sum <= EPS {
        let third = 1.0 / (LOBE_COUNT as f32);
        return LobePdf {
            r: third,
            tt: third,
            trt: third,
        };
    }
    let inv = 1.0 / sum;
    LobePdf {
        r: r * inv,
        tt: tt * inv,
        trt: trt * inv,
    }
}

/// A lobe chosen by importance sampling plus the canonical sample remapped back
/// into `[0,1)` inside the chosen lobe's cdf interval (so the caller can reuse it
/// as a fresh stratified sample for the lobe's own azimuthal/longitudinal term).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobeSample {
    /// Chosen lobe index (`0..LOBE_COUNT`).
    pub lobe: usize,
    /// The input `u` rescaled to `[0,1)` within the chosen lobe's cdf slice.
    pub remapped_u: f32,
    /// The chosen lobe's sampling probability (the pdf value).
    pub pdf: f32,
}

/// Importance-sample a lobe from a canonical uniform `u`. `u` is clamped to
/// `[0,1]`; the cdf is walked `R -> TT -> TRT` and the final lobe absorbs the
/// upper end (including `u == 1`), so a valid lobe is always returned.
#[must_use]
pub fn sample_lobe(pdf: LobePdf, u: f32) -> LobeSample {
    let uu = clamp01(if u.is_finite() { u } else { 0.0 });
    let p_r = sanitize_nonneg(pdf.r);
    let p_tt = sanitize_nonneg(pdf.tt);
    let p_trt = sanitize_nonneg(pdf.trt);

    let c_r = p_r;
    let c_tt = p_r + p_tt;

    let (lobe, lo, p) = if uu < c_r {
        (LOBE_R, 0.0, p_r)
    } else if uu < c_tt {
        (LOBE_TT, c_r, p_tt)
    } else {
        (LOBE_TRT, c_tt, p_trt)
    };

    let remapped_u = if p <= EPS {
        0.0
    } else {
        ((uu - lo) / p).clamp(0.0, 1.0)
    };

    LobeSample {
        lobe,
        remapped_u,
        pdf: p,
    }
}

/// Deterministically allocate `total_samples` across the three lobes in
/// proportion to the pdf (largest-remainder apportionment: floor each quota, then
/// hand the leftover samples to the largest fractional remainders, ties broken by
/// ascending lobe index). The returned counts always sum to `total_samples`.
#[must_use]
pub fn stratified_allocation(pdf: LobePdf, total_samples: usize) -> [usize; LOBE_COUNT] {
    let mut counts = [0usize; LOBE_COUNT];
    if total_samples == 0 {
        return counts;
    }
    let probs = [
        sanitize_nonneg(pdf.r),
        sanitize_nonneg(pdf.tt),
        sanitize_nonneg(pdf.trt),
    ];
    let total_f = total_samples as f32;
    let mut quotas = [0.0f32; LOBE_COUNT];
    let mut assigned = 0usize;
    let mut i = 0;
    while i < LOBE_COUNT {
        let q = probs[i] * total_f;
        let base = q.floor();
        quotas[i] = q - base; // fractional remainder in [0,1)
        let base_count = base.max(0.0) as usize;
        counts[i] = base_count;
        assigned += base_count;
        i += 1;
    }
    // Guard against over/under-assignment from fp rounding of a near-1 sum.
    if assigned > total_samples {
        // Trim from the smallest-remainder lobes first (reverse of top-up order).
        let mut over = assigned - total_samples;
        while over > 0 {
            let mut victim = usize::MAX;
            let mut worst = f32::INFINITY;
            let mut j = 0;
            while j < LOBE_COUNT {
                if counts[j] > 0 && quotas[j] < worst {
                    worst = quotas[j];
                    victim = j;
                }
                j += 1;
            }
            if victim == usize::MAX {
                break;
            }
            counts[victim] -= 1;
            over -= 1;
        }
        return counts;
    }
    let mut leftover = total_samples - assigned;
    while leftover > 0 {
        let mut best = usize::MAX;
        let mut best_rem = -1.0f32;
        let mut j = 0;
        while j < LOBE_COUNT {
            if quotas[j] > best_rem {
                best_rem = quotas[j];
                best = j;
            }
            j += 1;
        }
        if best == usize::MAX {
            break;
        }
        counts[best] += 1;
        // Deplete so the next leftover prefers a different lobe (round-robin by
        // remaining remainder), keeping ties resolved by ascending index.
        quotas[best] -= 1.0;
        leftover -= 1;
    }
    counts
}

/// Per-sample Monte-Carlo weight for lobe `i` under a stratified allocation:
/// `pdf_i * total_samples / count_i`, i.e. the reciprocal of the per-sample
/// probability so the stratified estimator stays unbiased. Returns `0` when no
/// samples were allocated to the lobe (`count_i == 0`) or `total_samples == 0`.
#[must_use]
pub fn stratified_sample_weight(pdf_i: f32, count_i: usize, total_samples: usize) -> f32 {
    if count_i == 0 || total_samples == 0 {
        return 0.0;
    }
    sanitize_nonneg(pdf_i) * (total_samples as f32) / (count_i as f32)
}

/// Full per-lobe stratified sample-weight vector (length [`LOBE_COUNT`]) for a
/// pdf and total sample budget. Pairs with [`stratified_allocation`].
#[must_use]
pub fn stratified_weights(pdf: LobePdf, total_samples: usize) -> Vec<f32> {
    let counts = stratified_allocation(pdf, total_samples);
    let probs = pdf.as_array();
    let mut out = Vec::with_capacity(LOBE_COUNT);
    let mut i = 0;
    while i < LOBE_COUNT {
        out.push(stratified_sample_weight(probs[i], counts[i], total_samples));
        i += 1;
    }
    out
}

/// Clamp to `[0,1]`, mapping non-finite inputs to `0`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamp to a finite `>= 0` value (non-finite -> `0`).
#[must_use]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < T_EPS
    }

    #[test]
    fn blend_is_zero_at_or_above_near_and_one_at_or_below_far() {
        let th = ScatterLodThresholds::new(1.0, 0.25);
        assert!(close(scatter_blend(2.0, th), 0.0));
        assert!(close(scatter_blend(1.0, th), 0.0));
        assert!(close(scatter_blend(0.25, th), 1.0));
        assert!(close(scatter_blend(0.1, th), 1.0));
    }

    #[test]
    fn blend_is_monotone_non_increasing_in_width() {
        let th = ScatterLodThresholds::new(1.0, 0.25);
        let mut prev = scatter_blend(0.0, th);
        let mut w = 0.0;
        while w <= 1.5 {
            let b = scatter_blend(w, th);
            assert!(b <= prev + T_EPS, "w={w} b={b} prev={prev}");
            assert!((0.0..=1.0).contains(&b));
            prev = b;
            w += 0.05;
        }
    }

    #[test]
    fn blend_midpoint_is_half() {
        let th = ScatterLodThresholds::new(1.0, 0.0);
        assert!(close(scatter_blend(0.5, th), 0.5));
    }

    #[test]
    fn regime_classification() {
        let th = ScatterLodThresholds::new(1.0, 0.25);
        assert_eq!(scatter_regime(2.0, th), ScatterRegime::Near);
        assert_eq!(scatter_regime(0.1, th), ScatterRegime::Far);
        assert_eq!(scatter_regime(0.625, th), ScatterRegime::Blended);
    }

    #[test]
    fn thresholds_sanitize_order_and_sign() {
        let th = ScatterLodThresholds::new(f32::NAN, -3.0).sanitized();
        assert!(close(th.far_px, 0.0));
        assert!(th.near_px >= th.far_px);
        let th2 = ScatterLodThresholds::new(0.2, 0.9).sanitized();
        assert!(th2.near_px >= th2.far_px);
    }

    #[test]
    fn roughness_gain_widens_with_blend() {
        assert!(close(far_field_roughness_gain(0.0, 2.0), 1.0));
        assert!(close(far_field_roughness_gain(1.0, 2.0), 3.0));
        assert!(close(far_field_roughness_gain(0.5, 2.0), 2.0));
        // negative/non-finite gain -> no widening.
        assert!(close(far_field_roughness_gain(1.0, -5.0), 1.0));
        assert!(close(far_field_roughness_gain(2.0, 1.0), 2.0));
    }

    #[test]
    fn lobe_weights_are_non_negative_and_ordered_for_dark_hair() {
        // Strongly absorbing fibre: under the rational transmittance proxy
        // `T = 1/(1+a)`, `R >= TT` only holds once `T <= F/(1-F)^2`
        // (here `a >= ~17`), so pick a genuinely dark absorption.
        let w = lobe_weights(LobeOpticalInput::new(0.05, 40.0));
        for v in w.as_array() {
            assert!(v >= 0.0);
        }
        assert!(w.r >= w.tt);
        assert!(w.tt >= w.trt);
    }

    #[test]
    fn pdf_sums_to_one() {
        let w = lobe_weights(LobeOpticalInput::new(0.1, 0.5));
        let p = lobe_pdf(w);
        assert!(close(p.r + p.tt + p.trt, 1.0));
        for v in p.as_array() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn pdf_uniform_fallback_for_zero_energy() {
        let p = lobe_pdf(LobeWeights::new(0.0, 0.0, 0.0));
        let third = 1.0 / 3.0;
        assert!(close(p.r, third));
        assert!(close(p.tt, third));
        assert!(close(p.trt, third));
    }

    #[test]
    fn pdf_sanitizes_negative_and_non_finite_energies() {
        let p = lobe_pdf(LobeWeights::new(-1.0, f32::NAN, 2.0));
        assert!(close(p.r + p.tt + p.trt, 1.0));
        assert!(close(p.r, 0.0));
        assert!(close(p.tt, 0.0));
        assert!(close(p.trt, 1.0));
    }

    #[test]
    fn sample_lobe_picks_each_lobe_in_its_cdf_slice() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        assert_eq!(sample_lobe(p, 0.0).lobe, LOBE_R);
        assert_eq!(sample_lobe(p, 0.25).lobe, LOBE_R);
        assert_eq!(sample_lobe(p, 0.6).lobe, LOBE_TT);
        assert_eq!(sample_lobe(p, 0.9).lobe, LOBE_TRT);
        assert_eq!(sample_lobe(p, 1.0).lobe, LOBE_TRT);
    }

    #[test]
    fn sample_lobe_remaps_u_into_unit_interval() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        // u=0.25 is the midpoint of the R slice [0,0.5) -> remapped 0.5.
        let s = sample_lobe(p, 0.25);
        assert_eq!(s.lobe, LOBE_R);
        assert!(close(s.remapped_u, 0.5));
        assert!(close(s.pdf, 0.5));
        // Mid of TT slice [0.5,0.8).
        let s2 = sample_lobe(p, 0.65);
        assert_eq!(s2.lobe, LOBE_TT);
        assert!(close(s2.remapped_u, 0.5));
    }

    #[test]
    fn sample_lobe_handles_non_finite_u() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        let s = sample_lobe(p, f32::NAN);
        assert_eq!(s.lobe, LOBE_R);
        assert!((0.0..=1.0).contains(&s.remapped_u));
    }

    #[test]
    fn allocation_sums_to_total() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        for total in [0usize, 1, 7, 10, 100, 999] {
            let c = stratified_allocation(p, total);
            assert_eq!(c[0] + c[1] + c[2], total, "total={total}");
        }
    }

    #[test]
    fn allocation_is_proportional() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        let c = stratified_allocation(p, 100);
        assert_eq!(c, [50, 30, 20]);
    }

    #[test]
    fn allocation_largest_remainder_tie_breaks_by_index() {
        // Uniform thirds of 10 -> base [3,3,3], remainder 1/3 each, leftover 1
        // goes to lowest index.
        let p = LobePdf {
            r: 1.0 / 3.0,
            tt: 1.0 / 3.0,
            trt: 1.0 / 3.0,
        };
        let c = stratified_allocation(p, 10);
        assert_eq!(c[0] + c[1] + c[2], 10);
        assert_eq!(c, [4, 3, 3]);
    }

    #[test]
    fn stratified_weights_match_reciprocal_probability() {
        let p = LobePdf {
            r: 0.5,
            tt: 0.3,
            trt: 0.2,
        };
        let w = stratified_weights(p, 100);
        assert_eq!(w.len(), LOBE_COUNT);
        // count = [50,30,20]; weight = pdf*total/count = [1,1,1].
        for v in &w {
            assert!(close(*v, 1.0));
        }
    }

    #[test]
    fn stratified_weight_zero_when_no_samples() {
        assert!(close(stratified_sample_weight(0.5, 0, 10), 0.0));
        assert!(close(stratified_sample_weight(0.5, 4, 0), 0.0));
    }

    #[test]
    fn empty_budget_allocates_nothing() {
        let p = lobe_pdf(lobe_weights(LobeOpticalInput::new(0.2, 1.0)));
        assert_eq!(stratified_allocation(p, 0), [0, 0, 0]);
        let w = stratified_weights(p, 0);
        for v in &w {
            assert!(close(*v, 0.0));
        }
    }
}
