//! Multiple-importance-sampling weights for next-event estimation — CPU golden.
//!
//! A direct-lighting integral can be estimated by sampling the *light* (great
//! for small bright emitters) or by sampling the *BSDF* (great for sharp
//! glossy lobes).  Multiple importance sampling (MIS; Veach & Guibas 1995)
//! combines both without double counting by weighting each strategy's
//! contribution by a heuristic that favours whichever strategy sampled the
//! shared point more densely.  This module is the backend-neutral reference for
//! those weights:
//!
//! * [`balance_heuristic`] — Veach's balance heuristic
//!   `n_f p_f / (n_f p_f + n_g p_g)`.
//! * [`power_heuristic`] — the power heuristic with exponent `β = 2`, which
//!   suppresses the higher-variance strategy more aggressively and is the
//!   production default.
//! * [`mis_weight_light`] / [`mis_weight_bsdf`] — convenience wrappers that
//!   return the power-heuristic weight for the light- and BSDF-sampling
//!   strategies, so a NEE estimator can weight each term directly.
//! * [`area_to_solid_angle_pdf`] / [`solid_angle_to_area_pdf`] /
//!   [`bsdf_vs_light_same_domain`] — pdf conversions that bring an area-measure
//!   light pdf and a solid-angle-measure BSDF pdf into the **same domain**
//!   before they are compared, which is mandatory for a correct heuristic.
//!
//! # Delta lights
//! A Dirac-delta emitter (directional / point / spot) has no finite density: it
//! can only be hit by light sampling, never by BSDF sampling.  Such a light is
//! represented here by the sentinel pdf [`DELTA_PDF`] (`f32::INFINITY`).  The
//! heuristics detect it and return weight `1` for the light strategy and `0`
//! for the BSDF strategy, so the MIS combination degrades gracefully to pure
//! NEE for delta lights.
//!
//! # Conventions
//! * All densities and counts are `f32`, matching the GPU twin.
//! * Negative or `NaN` inputs are clamped to `0` (via `f32::max`, which maps
//!   `NaN → 0`), so a denominator is always well defined.
//! * Every weight is clamped to `[0, 1]`; a degenerate (zero / non-finite)
//!   denominator yields `0`.  No function ever returns a `NaN`.
//! * All functions are deterministic pure functions: no RNG, no I/O, no GPU,
//!   no global state and no `unsafe`.

/// Sentinel pdf marking a Dirac-delta light (directional / point / spot): it
/// has no finite solid-angle density, so BSDF sampling can never reproduce it.
pub const DELTA_PDF: f32 = f32::INFINITY;

/// Tiny positive epsilon guarding divisions against degenerate denominators.
const EPS: f32 = 1.0e-12;

/// Whether `pdf` is the delta sentinel (an infinite density).
#[inline]
pub fn is_delta_pdf(pdf: f32) -> bool {
    pdf.is_infinite() && pdf > 0.0
}

/// Veach's **balance heuristic** weight for the first strategy.
///
/// Returns `n_f p_f / (n_f p_f + n_g p_g)`, the provably good MIS weight whose
/// combined estimator variance is close to optimal.  `nf` / `ng` are the sample
/// counts of each strategy and `fpdf` / `gpdf` their densities **evaluated in
/// the same domain** for the shared sample.
///
/// Delta handling: if `fpdf` is the [`DELTA_PDF`] sentinel the first strategy is
/// the only one that can sample the point, so the weight is `1` (and `0` if only
/// `gpdf` is delta).  Non-finite / non-positive products clamp to `0`; the
/// result is always in `[0, 1]`.
#[inline]
pub fn balance_heuristic(nf: f32, fpdf: f32, ng: f32, gpdf: f32) -> f32 {
    let f_delta = is_delta_pdf(fpdf);
    let g_delta = is_delta_pdf(gpdf);
    match (f_delta, g_delta) {
        (true, true) => return 0.5,
        (true, false) => return 1.0,
        (false, true) => return 0.0,
        (false, false) => {}
    }
    let f = nf.max(0.0) * fpdf.max(0.0);
    let g = ng.max(0.0) * gpdf.max(0.0);
    let denom = f + g;
    if denom <= EPS || !denom.is_finite() {
        return 0.0;
    }
    (f / denom).clamp(0.0, 1.0)
}

/// The **power heuristic** (exponent `β = 2`) weight for the first strategy.
///
/// Returns `(n_f p_f)² / ((n_f p_f)² + (n_g p_g)²)`.  Squaring the per-strategy
/// densities pushes the weight more sharply towards whichever strategy sampled
/// the point best, which lowers variance relative to the balance heuristic in
/// most scenes and is the production default.
///
/// Delta handling, clamping and the `[0, 1]` range match [`balance_heuristic`].
#[inline]
pub fn power_heuristic(nf: f32, fpdf: f32, ng: f32, gpdf: f32) -> f32 {
    let f_delta = is_delta_pdf(fpdf);
    let g_delta = is_delta_pdf(gpdf);
    match (f_delta, g_delta) {
        (true, true) => return 0.5,
        (true, false) => return 1.0,
        (false, true) => return 0.0,
        (false, false) => {}
    }
    let f = nf.max(0.0) * fpdf.max(0.0);
    let g = ng.max(0.0) * gpdf.max(0.0);
    let f2 = f * f;
    let g2 = g * g;
    let denom = f2 + g2;
    if denom <= EPS || !denom.is_finite() {
        return 0.0;
    }
    (f2 / denom).clamp(0.0, 1.0)
}

/// Power-heuristic MIS weight for a sample drawn by **light sampling**.
///
/// `light_pdf` and `bsdf_pdf` must both be solid-angle densities for the shared
/// direction; `n_light` / `n_bsdf` are the respective sample counts (use `1`
/// each for the classic one-sample-per-strategy estimator).  For a delta light
/// (`light_pdf == DELTA_PDF`) this returns `1`.
#[inline]
pub fn mis_weight_light(light_pdf: f32, bsdf_pdf: f32, n_light: f32, n_bsdf: f32) -> f32 {
    power_heuristic(n_light, light_pdf, n_bsdf, bsdf_pdf)
}

/// Power-heuristic MIS weight for a sample drawn by **BSDF sampling** that
/// happened to hit a light.
///
/// Mirror of [`mis_weight_light`] with the strategies swapped.  For a delta
/// light the light pdf is [`DELTA_PDF`], so a BSDF sample can never reproduce it
/// and this returns `0`.
#[inline]
pub fn mis_weight_bsdf(bsdf_pdf: f32, light_pdf: f32, n_bsdf: f32, n_light: f32) -> f32 {
    power_heuristic(n_bsdf, bsdf_pdf, n_light, light_pdf)
}

/// Converts an **area** pdf on a light surface to a **solid-angle** pdf at the
/// shading point, so it can be compared against a BSDF solid-angle pdf.
///
/// `p_ω = p_A · dist² / cos θ_l`, with `θ_l` the angle between the light normal
/// and the connection direction.  Returns `0` for a degenerate configuration.
#[inline]
pub fn area_to_solid_angle_pdf(area_pdf: f32, distance: f32, cos_light: f32) -> f32 {
    let cos_light = cos_light.max(0.0);
    if area_pdf <= 0.0
        || !area_pdf.is_finite()
        || cos_light <= EPS
        || distance <= EPS
        || !distance.is_finite()
    {
        return 0.0;
    }
    let pdf = area_pdf * distance * distance / cos_light;
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Converts a **solid-angle** pdf back to an **area** pdf: the inverse of
/// [`area_to_solid_angle_pdf`] (`p_A = p_ω · cos θ_l / dist²`).
///
/// Returns `0` for a degenerate configuration.
#[inline]
pub fn solid_angle_to_area_pdf(solid_angle_pdf: f32, distance: f32, cos_light: f32) -> f32 {
    let cos_light = cos_light.max(0.0);
    if solid_angle_pdf <= 0.0
        || !solid_angle_pdf.is_finite()
        || distance <= EPS
        || !distance.is_finite()
    {
        return 0.0;
    }
    let pdf = solid_angle_pdf * cos_light / (distance * distance);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Brings a light's area pdf and a BSDF solid-angle pdf into the **same**
/// (solid-angle) domain, returning `(light_pdf_sa, bsdf_pdf_sa)` ready for a
/// heuristic.
///
/// The light pdf is converted from area to solid angle; the BSDF pdf is already
/// a solid-angle density and is passed through unchanged (clamped non-negative).
/// A delta light's sentinel [`DELTA_PDF`] is preserved so the heuristics can
/// detect it.  This exists to make the "same-domain comparison" requirement of
/// MIS explicit and hard to get wrong at the call site.
#[inline]
pub fn bsdf_vs_light_same_domain(
    light_area_pdf: f32,
    distance: f32,
    cos_light: f32,
    bsdf_solid_angle_pdf: f32,
    light_is_delta: bool,
) -> (f32, f32) {
    let light_sa = if light_is_delta {
        DELTA_PDF
    } else {
        area_to_solid_angle_pdf(light_area_pdf, distance, cos_light)
    };
    let bsdf_sa = if bsdf_solid_angle_pdf.is_finite() {
        bsdf_solid_angle_pdf.max(0.0)
    } else {
        0.0
    };
    (light_sa, bsdf_sa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_heuristic_matches_closed_form() {
        // n_f = n_g = 1: w = p_f / (p_f + p_g).
        let w = balance_heuristic(1.0, 2.0, 1.0, 6.0);
        assert!((w - 2.0 / 8.0).abs() < 1.0e-6, "w={w}");
    }

    #[test]
    fn balance_heuristic_pair_sums_to_one() {
        let (pf, pg) = (3.0_f32, 5.0_f32);
        let wf = balance_heuristic(1.0, pf, 1.0, pg);
        let wg = balance_heuristic(1.0, pg, 1.0, pf);
        assert!((wf + wg - 1.0).abs() < 1.0e-6, "wf={wf} wg={wg}");
    }

    #[test]
    fn balance_heuristic_respects_sample_counts() {
        // Doubling n_f doubles the numerator weight.
        let w = balance_heuristic(2.0, 1.0, 1.0, 1.0);
        assert!((w - 2.0 / 3.0).abs() < 1.0e-6, "w={w}");
    }

    #[test]
    fn power_heuristic_matches_closed_form() {
        // (p_f)² / ((p_f)² + (p_g)²).
        let w = power_heuristic(1.0, 2.0, 1.0, 4.0);
        let expected = 4.0 / (4.0 + 16.0);
        assert!((w - expected).abs() < 1.0e-6, "w={w}");
    }

    #[test]
    fn power_heuristic_pair_sums_to_one() {
        let (pf, pg) = (1.5_f32, 4.0_f32);
        let wf = power_heuristic(1.0, pf, 1.0, pg);
        let wg = power_heuristic(1.0, pg, 1.0, pf);
        assert!((wf + wg - 1.0).abs() < 1.0e-6, "wf={wf} wg={wg}");
    }

    #[test]
    fn power_heuristic_is_sharper_than_balance() {
        // When f dominates, the power heuristic pushes the weight higher.
        let b = balance_heuristic(1.0, 4.0, 1.0, 1.0);
        let p = power_heuristic(1.0, 4.0, 1.0, 1.0);
        assert!(p > b, "power={p} balance={b}");
    }

    #[test]
    fn heuristics_handle_zero_and_negative_pdfs() {
        assert_eq!(balance_heuristic(1.0, 0.0, 1.0, 0.0), 0.0);
        assert_eq!(power_heuristic(1.0, 0.0, 1.0, 0.0), 0.0);
        // Negative pdfs clamp to zero → degenerate denominator → 0.
        assert_eq!(balance_heuristic(1.0, -1.0, 1.0, -2.0), 0.0);
        // Only f positive → weight 1.
        assert!((balance_heuristic(1.0, 5.0, 1.0, 0.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn heuristics_never_produce_nan() {
        let w1 = balance_heuristic(f32::NAN, f32::NAN, 1.0, 1.0);
        let w2 = power_heuristic(1.0, f32::NAN, f32::NAN, 1.0);
        assert!(!w1.is_nan());
        assert!(!w2.is_nan());
    }

    #[test]
    fn delta_light_degenerates_to_weight_one() {
        // Light sample on a delta light: weight 1.
        let wl = mis_weight_light(DELTA_PDF, 3.0, 1.0, 1.0);
        assert_eq!(wl, 1.0);
        // A BSDF sample can never hit a delta light: weight 0.
        let wb = mis_weight_bsdf(3.0, DELTA_PDF, 1.0, 1.0);
        assert_eq!(wb, 0.0);
    }

    #[test]
    fn delta_light_balance_and_power_agree() {
        assert_eq!(balance_heuristic(1.0, DELTA_PDF, 1.0, 2.0), 1.0);
        assert_eq!(power_heuristic(1.0, DELTA_PDF, 1.0, 2.0), 1.0);
        assert_eq!(balance_heuristic(1.0, 2.0, 1.0, DELTA_PDF), 0.0);
        assert_eq!(power_heuristic(1.0, 2.0, 1.0, DELTA_PDF), 0.0);
    }

    #[test]
    fn mis_light_and_bsdf_weights_sum_to_one_for_finite_pdfs() {
        let (lp, bp) = (2.5_f32, 1.0_f32);
        let wl = mis_weight_light(lp, bp, 1.0, 1.0);
        let wb = mis_weight_bsdf(bp, lp, 1.0, 1.0);
        assert!((wl + wb - 1.0).abs() < 1.0e-6, "wl={wl} wb={wb}");
    }

    #[test]
    fn pdf_domain_round_trip() {
        let area_pdf = 0.5_f32;
        let distance = 2.0_f32;
        let cos_light = 0.75_f32;
        let sa = area_to_solid_angle_pdf(area_pdf, distance, cos_light);
        let back = solid_angle_to_area_pdf(sa, distance, cos_light);
        assert!((back - area_pdf).abs() / area_pdf < 1.0e-5, "back={back}");
    }

    #[test]
    fn same_domain_helper_converts_light_and_preserves_bsdf() {
        let (light_sa, bsdf_sa) = bsdf_vs_light_same_domain(0.5, 2.0, 0.75, 1.3, false);
        let expected_light = area_to_solid_angle_pdf(0.5, 2.0, 0.75);
        assert!((light_sa - expected_light).abs() < 1.0e-6);
        assert!((bsdf_sa - 1.3).abs() < 1.0e-6);
    }

    #[test]
    fn same_domain_helper_marks_delta_light() {
        let (light_sa, bsdf_sa) = bsdf_vs_light_same_domain(0.0, 2.0, 0.5, 1.0, true);
        assert!(is_delta_pdf(light_sa));
        assert!((bsdf_sa - 1.0).abs() < 1.0e-6);
        // Feeding into the heuristic yields a unit light weight.
        assert_eq!(mis_weight_light(light_sa, bsdf_sa, 1.0, 1.0), 1.0);
    }

    #[test]
    fn conversions_reject_degenerate_geometry() {
        assert_eq!(area_to_solid_angle_pdf(1.0, 1.0, 0.0), 0.0);
        assert_eq!(area_to_solid_angle_pdf(1.0, 0.0, 1.0), 0.0);
        assert_eq!(solid_angle_to_area_pdf(1.0, 0.0, 1.0), 0.0);
    }
}
