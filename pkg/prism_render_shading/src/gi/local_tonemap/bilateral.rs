//! Edge-preserving local adaptation via bilateral filtering and base/detail
//! decomposition — backend-neutral CPU golden.
//!
//! Durand & Dorsey's 2002 HDR display operator rests on a single idea: split the
//! log-luminance image into a slowly varying **base** layer (the large-scale
//! lighting envelope) and a **detail** layer (fine texture and edges), compress
//! only the base, then re-add the detail at (near) full strength. The split must
//! be *edge-aware*, otherwise haloes appear along high-contrast boundaries — so
//! the base layer is a bilateral filter rather than a plain Gaussian blur.
//!
//! A bilateral filter weights each neighbour by the product of two Gaussians:
//! a **spatial** term that falls off with pixel distance, and a **range** term
//! that falls off with the luminance *difference* to the window centre. Pixels
//! on the far side of an edge contribute almost nothing, so edges are preserved
//! while smooth regions are averaged.
//!
//! Everything here is a pure function over a borrowed window slice plus an
//! explicit [`BilateralWindow`] geometry; the caller owns gather and storage, so
//! this module never allocates and never touches a 2-D image buffer directly.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Windows are row-major `&[f32]` of length `width * height`; the centre is an
//!   explicit `(cx, cy)` cell. Degenerate geometry falls back to the centre
//!   sample (identity), never to a divide-by-zero.
//! * Sigmas are floored to a tiny positive value; all weights and outputs are
//!   finite. No path can emit `NaN` or `inf`.
//!
//! # References
//! * Durand & Dorsey, "Fast Bilateral Filtering for the Display of
//!   High-Dynamic-Range Images", SIGGRAPH 2002 — base/detail split in log space.
//! * Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images",
//!   ICCV 1998 — spatial × range Gaussian weighting.

use bevy_math::ops;

/// Smallest sigma permitted for either Gaussian; keeps weights finite.
pub const MIN_SIGMA: f32 = 1.0e-4;

/// Rectangular window geometry describing a row-major luminance tile and the
/// cell treated as its centre.
///
/// The backing slice is expected to hold `width * height` samples in row-major
/// order (`index = y * width + x`). The centre `(cx, cy)` is where the bilateral
/// range weight is anchored and is also the sample returned by the identity
/// fallbacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BilateralWindow {
    /// Window width in cells.
    pub width: usize,
    /// Window height in cells.
    pub height: usize,
    /// Centre column (clamped into `[0, width)` on use).
    pub cx: usize,
    /// Centre row (clamped into `[0, height)` on use).
    pub cy: usize,
}

impl BilateralWindow {
    /// Build a window with the centre at the geometric middle.
    #[must_use]
    pub fn centered(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cx: width / 2,
            cy: height / 2,
        }
    }

    /// Number of cells the window spans.
    #[must_use]
    pub fn len(self) -> usize {
        self.width * self.height
    }

    /// Whether the window has no cells.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Clamped linear index of the centre cell.
    #[must_use]
    fn center_index(self) -> usize {
        let cx = self.cx.min(self.width.saturating_sub(1));
        let cy = self.cy.min(self.height.saturating_sub(1));
        cy * self.width + cx
    }
}

/// Parameters for the bilateral base-layer estimator.
///
/// `sigma_spatial` is measured in cells; `sigma_range` in log-luminance units
/// (the filter is intended to run on [`super::luminance::log_luminance`] data).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilateralParams {
    /// Spatial Gaussian standard deviation, in cells.
    pub sigma_spatial: f32,
    /// Range Gaussian standard deviation, in value (log-luminance) units.
    pub sigma_range: f32,
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self {
            sigma_spatial: 1.5,
            sigma_range: 0.4,
        }
    }
}

impl BilateralParams {
    /// Return sanitised, strictly-positive sigmas.
    #[must_use]
    fn sanitized(self) -> (f32, f32) {
        let s = if self.sigma_spatial.is_finite() {
            self.sigma_spatial.max(MIN_SIGMA)
        } else {
            MIN_SIGMA
        };
        let r = if self.sigma_range.is_finite() {
            self.sigma_range.max(MIN_SIGMA)
        } else {
            MIN_SIGMA
        };
        (s, r)
    }
}

/// Unnormalised Gaussian `exp(-x^2 / (2 sigma^2))`.
///
/// `sigma` is floored to [`MIN_SIGMA`]; the result lies in `(0, 1]` and is
/// always finite.
#[must_use]
pub fn gaussian(x: f32, sigma: f32) -> f32 {
    let s = if sigma.is_finite() {
        sigma.max(MIN_SIGMA)
    } else {
        MIN_SIGMA
    };
    let e = -(x * x) / (2.0 * s * s);
    // Guard against -inf exponent for pathological inputs.
    if e.is_finite() { ops::exp(e) } else { 0.0 }
}

/// Edge-preserving bilateral estimate of the centre cell from its window.
///
/// For every cell the weight is `spatial(distance) * range(value - center)`;
/// the return value is the weight-normalised mean. Degenerate input (empty
/// window, size mismatch, or all-zero weights) falls back to the centre sample,
/// i.e. the filter acts as the identity rather than failing.
#[must_use]
pub fn bilateral_filter_window(
    window: &[f32],
    geom: BilateralWindow,
    params: BilateralParams,
) -> f32 {
    if geom.is_empty() || window.len() != geom.len() {
        // Fall back to whatever centre sample we can address.
        return window
            .get(geom.center_index().min(window.len().saturating_sub(1)))
            .copied()
            .map(sanitize)
            .unwrap_or(0.0);
    }
    let (sigma_spatial, sigma_range) = params.sanitized();
    let cx = geom.cx.min(geom.width - 1);
    let cy = geom.cy.min(geom.height - 1);
    let center = sanitize(window[cy * geom.width + cx]);

    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for y in 0..geom.height {
        let dy = y as f32 - cy as f32;
        for x in 0..geom.width {
            let dx = x as f32 - cx as f32;
            let value = sanitize(window[y * geom.width + x]);
            let dist = (dx * dx + dy * dy).sqrt();
            let w = gaussian(dist, sigma_spatial) * gaussian(value - center, sigma_range);
            weighted_sum += w * value;
            weight_sum += w;
        }
    }
    if weight_sum > 0.0 {
        let out = weighted_sum / weight_sum;
        if out.is_finite() { out } else { center }
    } else {
        center
    }
}

/// A log-luminance pixel split into base (coarse) and detail (fine) layers.
///
/// By construction `base + detail` reproduces the original log-luminance, so the
/// decomposition is exactly invertible up to floating-point rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseDetail {
    /// Coarse, edge-aware base layer (log-luminance units).
    pub base: f32,
    /// Fine detail residual, `original - base` (log-luminance units).
    pub detail: f32,
}

impl BaseDetail {
    /// Recombine the layers, applying a `detail_gain` to the detail residual.
    ///
    /// `detail_gain == 1.0` is a lossless reconstruction; `> 1.0` sharpens local
    /// contrast, `< 1.0` softens it. The gain is clamped to a sane non-negative
    /// range so the result stays finite.
    #[must_use]
    pub fn recombine(self, detail_gain: f32) -> f32 {
        let gain = if detail_gain.is_finite() {
            detail_gain.clamp(0.0, 16.0)
        } else {
            1.0
        };
        let v = self.base + self.detail * gain;
        if v.is_finite() { v } else { self.base }
    }
}

/// Split a log-luminance window into base/detail at its centre cell.
///
/// The base is [`bilateral_filter_window`]; the detail is the centre sample
/// minus that base. Operates purely in whatever (log) domain the caller feeds.
#[must_use]
pub fn decompose_base_detail(
    log_window: &[f32],
    geom: BilateralWindow,
    params: BilateralParams,
) -> BaseDetail {
    let base = bilateral_filter_window(log_window, geom, params);
    let center = if geom.is_empty() || log_window.len() != geom.len() {
        log_window
            .get(geom.center_index().min(log_window.len().saturating_sub(1)))
            .copied()
            .map(sanitize)
            .unwrap_or(0.0)
    } else {
        let cx = geom.cx.min(geom.width - 1);
        let cy = geom.cy.min(geom.height - 1);
        sanitize(log_window[cy * geom.width + cx])
    };
    let detail = center - base;
    BaseDetail {
        base,
        detail: if detail.is_finite() { detail } else { 0.0 },
    }
}

/// Replace non-finite samples with zero and leave finite ones untouched.
#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    #[test]
    fn gaussian_peak_is_one() {
        approx(gaussian(0.0, 1.0), 1.0);
    }

    #[test]
    fn gaussian_is_monotone_decreasing() {
        let a = gaussian(0.5, 1.0);
        let b = gaussian(1.0, 1.0);
        let c = gaussian(2.0, 1.0);
        assert!(a > b && b > c);
    }

    #[test]
    fn gaussian_handles_bad_sigma() {
        assert!(gaussian(1.0, 0.0).is_finite());
        assert!(gaussian(1.0, f32::NAN).is_finite());
    }

    #[test]
    fn constant_window_returns_constant() {
        let w = [5.0_f32; 25];
        let geom = BilateralWindow::centered(5, 5);
        approx(bilateral_filter_window(&w, geom, BilateralParams::default()), 5.0);
    }

    #[test]
    fn empty_geometry_falls_back_to_center() {
        let w = [3.0_f32];
        let geom = BilateralWindow {
            width: 0,
            height: 0,
            cx: 0,
            cy: 0,
        };
        approx(bilateral_filter_window(&w, geom, BilateralParams::default()), 3.0);
    }

    #[test]
    fn size_mismatch_falls_back_to_center() {
        let w = [1.0_f32, 2.0, 3.0];
        let geom = BilateralWindow::centered(5, 5); // expects 25
        let out = bilateral_filter_window(&w, geom, BilateralParams::default());
        assert!(out.is_finite());
    }

    #[test]
    fn preserves_edge_better_than_box() {
        // A sharp step edge: left half low, right half high; centre on the high
        // side. A tight range sigma should keep the centre near the high value
        // rather than averaging across the edge.
        let width = 5;
        let height = 1;
        let mut w = [0.0_f32; 5];
        for (x, cell) in w.iter_mut().enumerate() {
            *cell = if x < 2 { 0.0 } else { 10.0 };
        }
        let geom = BilateralWindow {
            width,
            height,
            cx: 3,
            cy: 0,
        };
        let params = BilateralParams {
            sigma_spatial: 2.0,
            sigma_range: 0.5,
        };
        let bilateral = bilateral_filter_window(&w, geom, params);
        // A plain mean would be (0+0+10+10+10)/5 = 6.0; bilateral stays near 10.
        assert!(bilateral > 9.0, "edge not preserved: {bilateral}");
    }

    #[test]
    fn large_range_sigma_approaches_mean() {
        // With a huge range sigma the range term is ~1 everywhere, so the
        // bilateral collapses toward a spatial-weighted blur.
        let width = 3;
        let height = 1;
        let w = [0.0_f32, 10.0, 0.0];
        let geom = BilateralWindow {
            width,
            height,
            cx: 1,
            cy: 0,
        };
        let params = BilateralParams {
            sigma_spatial: 100.0,
            sigma_range: 1.0e6,
        };
        let out = bilateral_filter_window(&w, geom, params);
        // Spatial weights near-equal -> close to the arithmetic mean 10/3.
        assert!((out - 10.0 / 3.0).abs() < 1.0e-3, "{out}");
    }

    #[test]
    fn base_detail_reconstructs_original() {
        let width = 3;
        let height = 3;
        let mut w = [0.0_f32; 9];
        for (i, c) in w.iter_mut().enumerate() {
            *c = i as f32 * 0.5;
        }
        let geom = BilateralWindow::centered(width, height);
        let bd = decompose_base_detail(&w, geom, BilateralParams::default());
        let center = w[geom.center_index()];
        approx(bd.base + bd.detail, center);
        // Lossless recombine at unit gain.
        approx(bd.recombine(1.0), center);
    }

    #[test]
    fn detail_gain_scales_residual() {
        let bd = BaseDetail {
            base: 2.0,
            detail: 0.5,
        };
        approx(bd.recombine(0.0), 2.0);
        approx(bd.recombine(2.0), 3.0);
    }

    #[test]
    fn detail_gain_clamps_bad_input() {
        let bd = BaseDetail {
            base: 1.0,
            detail: 1.0,
        };
        assert!(bd.recombine(f32::NAN).is_finite());
        assert!(bd.recombine(1.0e9).is_finite());
    }

    #[test]
    fn window_handles_non_finite_samples() {
        let mut w = [1.0_f32; 9];
        w[4] = f32::NAN;
        let geom = BilateralWindow::centered(3, 3);
        let out = bilateral_filter_window(&w, geom, BilateralParams::default());
        assert!(out.is_finite());
    }

    #[test]
    fn centered_helper_picks_middle() {
        let g = BilateralWindow::centered(7, 3);
        assert_eq!(g.cx, 3);
        assert_eq!(g.cy, 1);
        assert_eq!(g.len(), 21);
        assert!(!g.is_empty());
    }
}
