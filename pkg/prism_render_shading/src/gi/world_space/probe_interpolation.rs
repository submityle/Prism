//! Bilinear + geometry-aware interpolation of screen probes.
//!
//! When shading a pixel we gather the four screen probes surrounding it on the
//! coarse probe grid and blend their SH irradiance.  A naive bilinear blend
//! leaks light across depth discontinuities and around corners, so each
//! bilinear weight is additionally gated by geometric similarity: the probe's
//! normal must agree with the shading normal, and its depth must be close to
//! the shading point's depth.  Probes that fail either test are dropped, and
//! the surviving weights are renormalised.
//!
//! # Conventions
//! * The four neighbours are ordered `[(0,0), (1,0), (0,1), (1,1)]` — top-left,
//!   top-right, bottom-left, bottom-right — matching the row-major probe grid.
//! * `frac` is the shading point's fractional position inside the 2x2 probe
//!   quad, with `(0, 0)` at the top-left probe.  See
//!   [`probe_bilinear_coords`] for how to derive it from a pixel coordinate.
//! * Normal similarity uses `dot(n_probe, n_point)`; a probe is accepted when
//!   the dot exceeds `normal_threshold`.  Depth similarity uses the relative
//!   difference `|d_probe - d_point| / max(d_point, eps)`, accepted when below
//!   `depth_rel_threshold`.
//! * When every weight vanishes (all neighbours rejected, or a fully
//!   degenerate quad) the blend falls back to the neighbour with the largest
//!   bilinear weight, guaranteeing a defined, non-black result.

use bevy_math::Vec3;

use super::radiance_cache::{evaluate_irradiance, ShL1Rgb};

/// A neighbouring screen probe as seen during interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeNeighbor {
    /// The probe's stored L1 irradiance probe.
    pub sh: ShL1Rgb,
    /// World-space surface normal the probe was captured on.
    pub normal: Vec3,
    /// Linear view-space depth (or any monotal positive depth metric) the probe
    /// was captured at.
    pub depth: f32,
    /// Whether the probe holds valid data at all (e.g. it hit geometry).
    pub valid: bool,
}

impl ProbeNeighbor {
    /// Builds a valid neighbour.
    #[inline]
    pub fn new(sh: ShL1Rgb, normal: Vec3, depth: f32) -> Self {
        Self {
            sh,
            normal,
            depth,
            valid: true,
        }
    }

    /// An invalid neighbour that contributes nothing.
    pub const INVALID: Self = Self {
        sh: ShL1Rgb::ZERO,
        normal: Vec3::ZERO,
        depth: 0.0,
        valid: false,
    };
}

/// Thresholds controlling how aggressively dissimilar probes are rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolationConfig {
    /// Minimum `dot(n_probe, n_point)` for a probe to be accepted.
    pub normal_threshold: f32,
    /// Maximum relative depth difference for a probe to be accepted.
    pub depth_rel_threshold: f32,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            // ~26 degrees of normal deviation, and 10% relative depth slack.
            normal_threshold: 0.9,
            depth_rel_threshold: 0.1,
        }
    }
}

/// Bilinear weights for the 2x2 quad in `[(0,0),(1,0),(0,1),(1,1)]` order.
///
/// `frac` is clamped to `[0, 1]^2` so out-of-quad inputs stay well defined.
#[inline]
pub fn bilinear_weights(frac_x: f32, frac_y: f32) -> [f32; 4] {
    let fx = frac_x.clamp(0.0, 1.0);
    let fy = frac_y.clamp(0.0, 1.0);
    [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ]
}

/// Geometric acceptance weight (`0` or `1`) for a single probe.
///
/// Returns `0` when the probe is invalid, faces away beyond
/// `normal_threshold`, or its depth differs by more than
/// `depth_rel_threshold` relatively.
#[inline]
pub fn similarity_weight(
    probe: &ProbeNeighbor,
    point_normal: Vec3,
    point_depth: f32,
    cfg: &InterpolationConfig,
) -> f32 {
    if !probe.valid {
        return 0.0;
    }
    let n_dot = probe.normal.dot(point_normal);
    if n_dot < cfg.normal_threshold {
        return 0.0;
    }
    let denom = point_depth.abs().max(1e-4);
    let depth_rel = (probe.depth - point_depth).abs() / denom;
    if depth_rel > cfg.depth_rel_threshold {
        return 0.0;
    }
    1.0
}

/// Combined bilinear * geometric weights for the four neighbours, normalised to
/// sum to one.
///
/// When no neighbour survives the geometric test, all-zero combined weights are
/// replaced by a one-hot vector selecting the neighbour with the largest
/// bilinear weight (nearest-probe fallback).
#[inline]
pub fn resolve_weights(
    neighbors: &[ProbeNeighbor; 4],
    frac_x: f32,
    frac_y: f32,
    point_normal: Vec3,
    point_depth: f32,
    cfg: &InterpolationConfig,
) -> [f32; 4] {
    let bw = bilinear_weights(frac_x, frac_y);
    let mut w = [0.0f32; 4];
    let mut sum = 0.0f32;
    for i in 0..4 {
        let s = similarity_weight(&neighbors[i], point_normal, point_depth, cfg);
        w[i] = bw[i] * s;
        sum += w[i];
    }
    if sum > f32::MIN_POSITIVE {
        let inv = sum.recip();
        for wi in w.iter_mut() {
            *wi *= inv;
        }
        return w;
    }
    // Fallback: pick the largest bilinear weight, preferring a valid probe.
    let mut best = usize::MAX;
    let mut best_w = -1.0f32;
    for i in 0..4 {
        if neighbors[i].valid && bw[i] > best_w {
            best_w = bw[i];
            best = i;
        }
    }
    if best == usize::MAX {
        // No valid neighbour at all: fall back to the largest bilinear weight
        // regardless of validity so the result is still defined (all zero SH).
        for (i, &b) in bw.iter().enumerate() {
            if b > best_w {
                best_w = b;
                best = i;
            }
        }
    }
    let mut out = [0.0f32; 4];
    if best != usize::MAX {
        out[best] = 1.0;
    }
    out
}

/// Blends the four neighbours' SH probes with `weights` (assumed to sum to
/// one) into a single probe.
#[inline]
pub fn blend_sh(neighbors: &[ProbeNeighbor; 4], weights: &[f32; 4]) -> ShL1Rgb {
    let mut out = ShL1Rgb::ZERO;
    for i in 0..4 {
        if weights[i] != 0.0 {
            out.add_scaled(&neighbors[i].sh, weights[i]);
        }
    }
    out
}

/// Full interpolation: resolve weights, blend the SH probes, and evaluate the
/// clamped-cosine irradiance along `point_normal`.
#[inline]
pub fn interpolate_irradiance(
    neighbors: &[ProbeNeighbor; 4],
    frac_x: f32,
    frac_y: f32,
    point_normal: Vec3,
    point_depth: f32,
    cfg: &InterpolationConfig,
) -> Vec3 {
    let weights = resolve_weights(neighbors, frac_x, frac_y, point_normal, point_depth, cfg);
    let sh = blend_sh(neighbors, &weights);
    evaluate_irradiance(&sh, point_normal)
}

/// Maps a shading pixel to its base probe coordinate and fractional position
/// inside the surrounding 2x2 probe quad.
///
/// Probe `i` on an axis is centred at pixel `(i + 0.5) * tile`, so the
/// continuous probe-space coordinate is `pixel / tile - 0.5`.  The integer
/// floor is clamped to `[0, dim - 2]` (when at least two probes exist) so the
/// returned quad always references in-range probes; the fraction absorbs the
/// clamp.
///
/// Returns `(base_x, base_y, frac_x, frac_y)`.
#[inline]
pub fn probe_bilinear_coords(
    pixel_x: f32,
    pixel_y: f32,
    tile: u32,
    dims_x: u32,
    dims_y: u32,
) -> (u32, u32, f32, f32) {
    let tile = tile.max(1) as f32;
    let coord_x = pixel_x / tile - 0.5;
    let coord_y = pixel_y / tile - 0.5;
    let (base_x, frac_x) = clamp_base(coord_x, dims_x);
    let (base_y, frac_y) = clamp_base(coord_y, dims_y);
    (base_x, base_y, frac_x, frac_y)
}

/// Splits a continuous probe coordinate into a clamped integer base and the
/// fraction toward the next probe.
#[inline]
fn clamp_base(coord: f32, dim: u32) -> (u32, f32) {
    if dim == 0 {
        return (0, 0.0);
    }
    if dim == 1 {
        // Only one probe on this axis: always reference it with zero fraction.
        return (0, 0.0);
    }
    let max_base = dim - 2;
    let floor = coord.floor();
    if floor < 0.0 {
        // Left of the first probe centre: reference probe 0 with zero fraction.
        return (0, 0.0);
    }
    let base = floor as u32;
    if base >= max_base {
        // Clamp to the last quad; push the fraction to keep continuity.
        let frac = (coord - max_base as f32).clamp(0.0, 1.0);
        return (max_base, frac);
    }
    (base, coord - floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(sh_dc: [f32; 3], normal: Vec3, depth: f32) -> ProbeNeighbor {
        let mut sh = ShL1Rgb::ZERO;
        sh.coefficients[0] = sh_dc;
        ProbeNeighbor::new(sh, normal, depth)
    }

    #[test]
    fn bilinear_weights_sum_to_one() {
        for &(x, y) in &[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)] {
            let w = bilinear_weights(x, y);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "sum {s} at {x},{y}");
        }
    }

    #[test]
    fn bilinear_weights_corners() {
        assert_eq!(bilinear_weights(0.0, 0.0), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bilinear_weights(1.0, 0.0), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(bilinear_weights(0.0, 1.0), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(bilinear_weights(1.0, 1.0), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn bilinear_weights_clamp_out_of_range() {
        assert_eq!(bilinear_weights(-1.0, -1.0), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bilinear_weights(2.0, 2.0), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn similarity_rejects_dissimilar_normal_and_depth() {
        let cfg = InterpolationConfig::default();
        let good = probe([1.0; 3], Vec3::Y, 10.0);
        assert_eq!(similarity_weight(&good, Vec3::Y, 10.0, &cfg), 1.0);
        // Opposing normal rejected.
        let flipped = probe([1.0; 3], Vec3::NEG_Y, 10.0);
        assert_eq!(similarity_weight(&flipped, Vec3::Y, 10.0, &cfg), 0.0);
        // Far depth rejected.
        let far = probe([1.0; 3], Vec3::Y, 20.0);
        assert_eq!(similarity_weight(&far, Vec3::Y, 10.0, &cfg), 0.0);
        // Invalid rejected.
        assert_eq!(
            similarity_weight(&ProbeNeighbor::INVALID, Vec3::Y, 10.0, &cfg),
            0.0
        );
    }

    #[test]
    fn resolve_weights_normalise_when_all_similar() {
        let cfg = InterpolationConfig::default();
        let neighbors = [
            probe([1.0; 3], Vec3::Y, 10.0),
            probe([1.0; 3], Vec3::Y, 10.0),
            probe([1.0; 3], Vec3::Y, 10.0),
            probe([1.0; 3], Vec3::Y, 10.0),
        ];
        let w = resolve_weights(&neighbors, 0.5, 0.5, Vec3::Y, 10.0, &cfg);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        for wi in w {
            assert!((wi - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn resolve_weights_drops_rejected_and_renormalises() {
        let cfg = InterpolationConfig::default();
        let neighbors = [
            probe([1.0; 3], Vec3::Y, 10.0),  // ok
            probe([1.0; 3], Vec3::NEG_Y, 10.0), // rejected normal
            probe([1.0; 3], Vec3::Y, 999.0), // rejected depth
            probe([1.0; 3], Vec3::Y, 10.0),  // ok
        ];
        // Centre of quad: bilinear all 0.25 -> two survivors share equally.
        let w = resolve_weights(&neighbors, 0.5, 0.5, Vec3::Y, 10.0, &cfg);
        assert!((w[0] - 0.5).abs() < 1e-6, "{w:?}");
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert!((w[3] - 0.5).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn resolve_weights_fallback_to_nearest_when_all_rejected() {
        let cfg = InterpolationConfig::default();
        // All neighbours face away -> all rejected.
        let neighbors = [
            probe([1.0; 3], Vec3::NEG_Y, 10.0),
            probe([2.0; 3], Vec3::NEG_Y, 10.0),
            probe([3.0; 3], Vec3::NEG_Y, 10.0),
            probe([4.0; 3], Vec3::NEG_Y, 10.0),
        ];
        // frac (0.9, 0.9) -> nearest is index 3 (bottom-right).
        let w = resolve_weights(&neighbors, 0.9, 0.9, Vec3::Y, 10.0, &cfg);
        assert_eq!(w, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn fallback_prefers_valid_probe() {
        let cfg = InterpolationConfig::default();
        // Nearest (bottom-right) is invalid; a valid but geometrically-rejected
        // probe elsewhere should win the fallback.
        let mut neighbors = [
            probe([5.0; 3], Vec3::NEG_Y, 10.0), // valid, rejected geom
            ProbeNeighbor::INVALID,
            ProbeNeighbor::INVALID,
            ProbeNeighbor::INVALID,
        ];
        neighbors[3] = ProbeNeighbor::INVALID;
        let w = resolve_weights(&neighbors, 0.9, 0.9, Vec3::Y, 10.0, &cfg);
        assert_eq!(w, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn interpolate_blends_dc_irradiance() {
        let cfg = InterpolationConfig::default();
        // Two equal survivors with different DC brightness -> average.
        let neighbors = [
            probe([1.0, 0.0, 0.0], Vec3::Y, 10.0),
            probe([0.0, 0.0, 0.0], Vec3::NEG_Y, 10.0), // rejected
            probe([0.0, 0.0, 0.0], Vec3::NEG_Y, 10.0), // rejected
            probe([3.0, 0.0, 0.0], Vec3::Y, 10.0),
        ];
        let e = interpolate_irradiance(&neighbors, 0.5, 0.5, Vec3::Y, 10.0, &cfg);
        // DC = (1+3)/2 = 2 (equal weights); irradiance = A0*K0*2 > 0, red only.
        assert!(e.x > 0.0);
        assert_eq!(e.y, 0.0);
        assert_eq!(e.z, 0.0);
        // Compare against directly blended probe.
        let mut blended = ShL1Rgb::ZERO;
        blended.coefficients[0] = [2.0, 0.0, 0.0];
        let expected = evaluate_irradiance(&blended, Vec3::Y);
        assert!((e - expected).length() < 1e-5, "{e:?} vs {expected:?}");
    }

    #[test]
    fn bilinear_coords_center_of_probe_zero() {
        // Pixel at probe 0 centre: (0.5*tile). tile=8 -> pixel 4.
        let (bx, by, fx, fy) = probe_bilinear_coords(4.0, 4.0, 8, 8, 8);
        assert_eq!((bx, by), (0, 0));
        assert!(fx.abs() < 1e-6 && fy.abs() < 1e-6, "{fx},{fy}");
    }

    #[test]
    fn bilinear_coords_between_probes() {
        // Halfway between probe 0 (pixel 4) and probe 1 (pixel 12) -> pixel 8.
        let (bx, _by, fx, _fy) = probe_bilinear_coords(8.0, 4.0, 8, 8, 8);
        assert_eq!(bx, 0);
        assert!((fx - 0.5).abs() < 1e-6, "{fx}");
    }

    #[test]
    fn bilinear_coords_clamped_at_far_edge() {
        // Far right pixel clamps base to dims-2 and frac to <=1.
        let (bx, _by, fx, _fy) = probe_bilinear_coords(1000.0, 4.0, 8, 8, 8);
        assert_eq!(bx, 6); // dims_x - 2
        assert!((0.0..=1.0).contains(&fx), "{fx}");
    }

    #[test]
    fn bilinear_coords_single_probe_axis() {
        let (bx, by, fx, fy) = probe_bilinear_coords(50.0, 4.0, 8, 1, 1);
        assert_eq!((bx, by), (0, 0));
        assert_eq!((fx, fy), (0.0, 0.0));
    }

    #[test]
    fn bilinear_coords_negative_side_clamped() {
        let (bx, _by, fx, _fy) = probe_bilinear_coords(0.0, 4.0, 8, 8, 8);
        assert_eq!(bx, 0);
        assert!((0.0..=1.0).contains(&fx));
    }
}
