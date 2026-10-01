//! Local (zonal) tone mapping CPU golden reference.
//!
//! Where [`super::post_gi`] ships *global* display operators (one curve, one
//! exposure for the whole frame), this module reproduces **local** tone mapping:
//! the per-pixel adaptation that lets a scene keep both deep shadow detail and
//! bright highlight structure in a single frame — the behaviour film emulsion
//! and the human eye exhibit, and the reason local operators avoid the "flat,
//! washed-out" look global curves produce on extreme dynamic range.
//!
//! The pipeline is split across three submodules:
//!
//! * [`luminance`] — Rec. 709 luminance extraction, guarded log-luminance,
//!   zonal statistics / histograms, and colour-preserving ratio re-application.
//! * [`bilateral`] — Durand-Dorsey edge-preserving base/detail decomposition on
//!   log-luminance (spatial × range Gaussian weighting).
//! * [`operator`] — Reinhard-local base compression plus Mertens exposure-fusion
//!   quality weights, recombining a gained detail layer into display luminance.
//!
//! The high-level entry point, [`local_tonemap_pixel`], takes a linear HDR RGB
//! sample and a precomputed local adaptation luminance (what the bilateral base
//! layer estimates for the pixel's neighbourhood), compresses in the luminance
//! domain, and scales RGB back by the clamped `Lout / Lin` ratio so hue and
//! saturation survive. Callers that already have a log-luminance window can feed
//! it through [`bilateral::decompose_base_detail`] and
//! [`operator::compress_base_detail`] directly; this function wraps the common
//! single-adaptation case.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; every output channel is finite and
//!   non-negative. No path can emit `NaN` or `inf`.
//!
//! # References
//! * Durand & Dorsey, "Fast Bilateral Filtering for the Display of HDR Images",
//!   SIGGRAPH 2002.
//! * Reinhard et al., "Photographic Tone Reproduction for Digital Images",
//!   SIGGRAPH 2002 — eq. 9 local dodge-and-burn operator `L / (1 + V(x,y))`.
//! * Mertens, Kautz & Van Reeth, "Exposure Fusion", Pacific Graphics 2007.

pub mod bilateral;
pub mod luminance;
pub mod operator;

pub use bilateral::{BaseDetail, BilateralParams, BilateralWindow};
pub use luminance::{
    LogHistogramRange, ZoneStats, apply_luminance_ratio, luminance as pixel_luminance,
};
pub use operator::{compress_base_detail, fusion_weight, reinhard_local};

use bevy_math::ops;

use luminance::{LUMINANCE_EPSILON, apply_luminance_ratio as apply_ratio, log_luminance, luminance};

/// Tunable parameters for [`local_tonemap_pixel`].
///
/// Every field has an AAA-sane default; all are sanitised at use so out-of-range
/// values degrade gracefully rather than producing `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalTonemapParams {
    /// Linear exposure multiplier applied to RGB before compression.
    pub exposure: f32,
    /// Scene-to-display prescale (Reinhard's `a / L_avg`): larger brightens.
    pub key: f32,
    /// White point: scaled luminance that maps toward display `1.0` and clips.
    pub white_point: f32,
    /// Local-contrast gain for the detail residual (`1.0` neutral, `>1` sharper).
    pub detail_gain: f32,
    /// Locality of the adaptation divisor: blends local adaptation toward the
    /// pixel's own luminance (`0.0` = fully global, `1.0` = fully local).
    pub contrast: f32,
    /// Maximum `Lout / Lin` ratio when re-applying luminance to RGB.
    pub max_ratio: f32,
}

impl Default for LocalTonemapParams {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            key: 1.0,
            white_point: 4.0,
            detail_gain: 1.1,
            contrast: 0.75,
            max_ratio: 8.0,
        }
    }
}

impl LocalTonemapParams {
    /// Return a copy with every field sanitised to a safe, finite value.
    #[must_use]
    fn sanitized(self) -> Self {
        let fix = |v: f32, default: f32, lo: f32, hi: f32| -> f32 {
            if v.is_finite() { v.clamp(lo, hi) } else { default }
        };
        Self {
            exposure: fix(self.exposure, 1.0, 0.0, 1.0e6),
            key: fix(self.key, 1.0, LUMINANCE_EPSILON, 1.0e3),
            white_point: fix(self.white_point, 4.0, LUMINANCE_EPSILON, 1.0e6),
            detail_gain: fix(self.detail_gain, 1.1, 0.0, 16.0),
            contrast: fix(self.contrast, 0.75, 0.0, 1.0),
            max_ratio: fix(self.max_ratio, 8.0, 1.0e-3, 1.0e6),
        }
    }
}

/// Tone-map a single linear HDR pixel using a precomputed local adaptation.
///
/// `rgb` is linear scene-referred radiance; `local_adaptation` is the local
/// (base-layer) luminance estimated for the pixel's neighbourhood — e.g. the
/// output of [`bilateral::bilateral_filter_window`] exponentiated back to the
/// linear domain. The returned RGB is display-referred, colour-preserving,
/// finite and non-negative, with luminance in `[0, 1]`.
///
/// # Pipeline
/// 1. Apply the linear `exposure` to RGB and compute its luminance `l_in`.
/// 2. Blend the local adaptation toward `l_in` by `contrast`, forming the
///    *effective* adaptation `V` (so `contrast = 0` is fully global, `1` fully
///    local).
/// 3. Boost the pixel's departure from `V` in the log domain by `detail_gain`
///    (micro-contrast); flat regions, where the pixel equals `V`, are untouched.
/// 4. Apply the Reinhard dodge-and-burn operator with a white point:
///    `L' (1 + L'/white^2) / (1 + V')` in the key-scaled domain, clamped to
///    `[0, 1]`. The `1 + V` denominator keeps absolute level (and black) intact.
/// 5. Re-apply the resulting luminance to RGB via a clamped ratio.
#[must_use]
pub fn local_tonemap_pixel(
    rgb: [f32; 3],
    local_adaptation: f32,
    params: LocalTonemapParams,
) -> [f32; 3] {
    let p = params.sanitized();

    // 1. Exposure + input luminance.
    let exposed = [
        rgb[0].max(0.0) * p.exposure,
        rgb[1].max(0.0) * p.exposure,
        rgb[2].max(0.0) * p.exposure,
    ];
    let l_in = luminance(exposed).max(0.0);

    // 2. Effective local adaptation (blend of surround and pixel luminance).
    let adapt_raw = if local_adaptation.is_finite() {
        local_adaptation.max(0.0) * p.exposure
    } else {
        l_in
    };
    let effective_adapt = (p.contrast * adapt_raw + (1.0 - p.contrast) * l_in).max(LUMINANCE_EPSILON);

    // 3. Detail boost in the log domain: amplify the pixel's deviation from its
    //    local adaptation. At a flat region (`l_in == effective_adapt`) the
    //    ratio is one and the boost is a no-op.
    let log_in = log_luminance(l_in);
    let log_adapt = log_luminance(effective_adapt);
    let boosted_log = log_adapt + (log_in - log_adapt) * p.detail_gain;
    let boosted = {
        let v = ops::exp(boosted_log);
        if v.is_finite() { v.max(0.0) } else { l_in }
    };

    // 4. Reinhard dodge-and-burn with white point, in the key-scaled domain.
    let l_scaled = p.key * boosted;
    let v_scaled = p.key * effective_adapt;
    let white_sq = p.white_point * p.white_point;
    let numerator = l_scaled * (1.0 + l_scaled / white_sq);
    let denom = 1.0 + v_scaled;
    let l_out = {
        let v = numerator / denom;
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
    };

    // 5. Colour-preserving re-application.
    apply_ratio(exposed, l_in, l_out, p.max_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx_rel(a: f32, b: f32) {
        let tol = 1.0e-3 * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{a} != {b}");
    }

    #[test]
    fn default_params_are_sane() {
        let p = LocalTonemapParams::default();
        assert!(p.key > 0.0);
        assert!(p.white_point > 0.0);
        assert!((0.0..=1.0).contains(&p.contrast));
    }

    #[test]
    fn sanitize_fixes_non_finite() {
        let bad = LocalTonemapParams {
            exposure: f32::NAN,
            key: -1.0,
            white_point: f32::INFINITY,
            detail_gain: f32::NAN,
            contrast: 5.0,
            max_ratio: -2.0,
        };
        let out = local_tonemap_pixel([0.5, 0.5, 0.5], 0.5, bad);
        for c in out {
            assert!(c.is_finite() && c >= 0.0);
        }
    }

    #[test]
    fn black_maps_to_black() {
        let out = local_tonemap_pixel([0.0, 0.0, 0.0], 0.0, LocalTonemapParams::default());
        approx(out[0], 0.0);
        approx(out[1], 0.0);
        approx(out[2], 0.0);
    }

    #[test]
    fn output_luminance_in_unit_range() {
        for &v in &[0.01_f32, 0.18, 1.0, 10.0, 1000.0] {
            let out = local_tonemap_pixel([v, v, v], v, LocalTonemapParams::default());
            let l = pixel_luminance(out);
            assert!((0.0..=1.0 + 1.0e-4).contains(&l), "v={v} l={l}");
        }
    }

    #[test]
    fn preserves_hue() {
        let rgb = [0.8, 0.4, 0.2];
        let out = local_tonemap_pixel(rgb, 0.4, LocalTonemapParams::default());
        // Channel ordering preserved (red brightest, blue darkest).
        assert!(out[0] > out[1] && out[1] > out[2]);
    }

    #[test]
    fn monotonic_in_input_brightness() {
        let p = LocalTonemapParams::default();
        let a = pixel_luminance(local_tonemap_pixel([0.1, 0.1, 0.1], 0.1, p));
        let b = pixel_luminance(local_tonemap_pixel([1.0, 1.0, 1.0], 1.0, p));
        let c = pixel_luminance(local_tonemap_pixel([10.0, 10.0, 10.0], 10.0, p));
        assert!(a < b && b < c, "{a} {b} {c}");
    }

    #[test]
    fn local_contrast_dodges_bright_surround() {
        // Identical pixel, differing local adaptation. A brighter surround (more
        // local compression) should darken the output when contrast > 0.
        let p = LocalTonemapParams::default();
        let dark_surround = pixel_luminance(local_tonemap_pixel([1.0, 1.0, 1.0], 0.1, p));
        let bright_surround = pixel_luminance(local_tonemap_pixel([1.0, 1.0, 1.0], 10.0, p));
        assert!(bright_surround < dark_surround, "{bright_surround} !< {dark_surround}");
    }

    #[test]
    fn zero_contrast_is_adaptation_independent() {
        // With contrast = 0 the operator is fully global: the local adaptation
        // argument must not change the result.
        let p = LocalTonemapParams {
            contrast: 0.0,
            ..LocalTonemapParams::default()
        };
        let a = local_tonemap_pixel([1.0, 1.0, 1.0], 0.1, p);
        let b = local_tonemap_pixel([1.0, 1.0, 1.0], 100.0, p);
        approx(a[0], b[0]);
    }

    #[test]
    fn exposure_brightens() {
        let dim = LocalTonemapParams {
            exposure: 0.5,
            ..LocalTonemapParams::default()
        };
        let bright = LocalTonemapParams {
            exposure: 4.0,
            ..LocalTonemapParams::default()
        };
        let a = pixel_luminance(local_tonemap_pixel([0.3, 0.3, 0.3], 0.3, dim));
        let b = pixel_luminance(local_tonemap_pixel([0.3, 0.3, 0.3], 0.3, bright));
        assert!(b > a);
    }

    #[test]
    fn detail_gain_boosts_micro_contrast() {
        // A pixel brighter than its surround: higher detail gain should push its
        // output luminance up relative to a neutral gain.
        let neutral = LocalTonemapParams {
            detail_gain: 1.0,
            ..LocalTonemapParams::default()
        };
        let sharp = LocalTonemapParams {
            detail_gain: 2.0,
            ..LocalTonemapParams::default()
        };
        let a = pixel_luminance(local_tonemap_pixel([0.3, 0.3, 0.3], 0.1, neutral));
        let b = pixel_luminance(local_tonemap_pixel([0.3, 0.3, 0.3], 0.1, sharp));
        assert!(b > a, "{b} !> {a}");
    }

    #[test]
    fn never_emits_non_finite_on_extremes() {
        let p = LocalTonemapParams::default();
        let out = local_tonemap_pixel([f32::INFINITY, 1.0e30, -5.0], f32::NAN, p);
        for c in out {
            assert!(c.is_finite() && c >= 0.0, "{c}");
        }
    }

    #[test]
    fn round_trip_tolerance_helper_is_relative() {
        // Guards the relative helper used by precision-sensitive sibling tests.
        approx_rel(5000.002, 5000.0);
    }

    #[test]
    fn reexports_are_reachable() {
        // Smoke test that the re-exported surface compiles and runs.
        let w = [1.0_f32; 9];
        let geom = BilateralWindow::centered(3, 3);
        let bd: BaseDetail = bilateral::decompose_base_detail(&w, geom, BilateralParams::default());
        let _ = compress_base_detail(bd, 1.0, 0.18, 6.0, 1.0);
        let _ = reinhard_local(1.0, 1.0, 1.0);
        let _ = fusion_weight(0.5, 0.5, 0.5, [1.0, 1.0, 1.0]);
        let _ = apply_luminance_ratio([1.0, 1.0, 1.0], 1.0, 0.5, 4.0);
        let _ = ZoneStats::default();
        let _ = LogHistogramRange::default();
    }
}
