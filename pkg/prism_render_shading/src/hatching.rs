//! Backend-neutral CPU golden for layered cross-hatching (Hatching) NPR.
//!
//! Cross-hatching is a stylized (NPR) post-process that reproduces the ink
//! drawing technique of shading with sets of parallel line strokes: the darker
//! a region, the more line directions are overlaid until they cross into a
//! dense mesh. Here the scene's Rec. 709 luminance drives a four-tier ramp —
//! brightest is left blank, then a single +45deg set of lines is added, then a
//! second direction, then a third for the deepest shadows — so the accumulated
//! line coverage tracks tone the way a pen artist builds up value.
//!
//! Each stroke set is a periodic stripe pattern read in a *rotated* screen
//! frame: [`rotate2d`] turns the pixel coordinate into the stroke's local frame
//! and [`line_coverage`] samples the fractional period, using a hand-written
//! `smoothstep` (`t * t * (3 - 2t)`) for anti-aliased edges. The only
//! transcendental is the rotation's sine/cosine, routed through
//! [`bevy_math::ops`] for libm determinism; squares are written `x * x` and the
//! rest is `floor` / `fract` / `clamp` / `min` arithmetic. [`apply_hatching`]
//! composites the ink over paper with `bg.lerp(fg, coverage)`, and is the
//! identity (returns the scene) while disabled. The whole module is mirrored
//! arm-for-arm — same constants, same tier order — by `shaders/hatching.wesl`.

use bevy_math::{ops, Vec2, Vec3};

/// Rec. 709 luma weights (linear `sRGB` primaries) that map tone to line density.
pub const HATCHING_LUMA_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// Half-width of the `smoothstep` anti-aliasing band on each stroke edge, in
/// fractional-period units.
const HATCHING_EDGE_SOFTNESS: f32 = 0.02;

/// Hermite `smoothstep` (`t * t * (3 - 2t)`) over `[edge0, edge1]`.
///
/// Hand-written rather than using a builtin so the CPU golden and the WESL twin
/// stay bit-identical and no disallowed transcendental is touched. `x <= edge0`
/// returns `0`, `x >= edge1` returns `1`, and the interior is the smooth ramp.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Rec. 709 relative luminance of a linear RGB sample; the tone that selects
/// how many stroke directions are drawn.
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    c.x * HATCHING_LUMA_WEIGHTS.x + c.y * HATCHING_LUMA_WEIGHTS.y + c.z * HATCHING_LUMA_WEIGHTS.z
}

/// Rotate a screen-space coordinate by `angle` radians (counter-clockwise) into
/// a stroke set's local frame. The sine/cosine go through [`bevy_math::ops`] for
/// libm determinism; rotation preserves length, so stroke spacing is uniform in
/// every direction.
#[must_use]
pub fn rotate2d(p: Vec2, angle: f32) -> Vec2 {
    let s = ops::sin(angle);
    let c = ops::cos(angle);
    Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y)
}

/// Coverage of a single stroke set at a *rotated* coordinate.
///
/// The stripes lie at integer multiples of `1 / frequency` along the rotated
/// `y` axis; `thickness` is the stroke half-width in fractional-period units.
/// The fractional period is folded to a distance-to-nearest-line and run
/// through [`smoothstep`], so the result is `1` on a stroke, `0` between strokes
/// and an anti-aliased ramp across each edge.
#[must_use]
pub fn line_coverage(rotated: Vec2, frequency: f32, thickness: f32) -> f32 {
    let coord = rotated.y * frequency;
    let f = coord - coord.floor();
    let dist = f.min(1.0 - f);
    1.0 - smoothstep(thickness, thickness + HATCHING_EDGE_SOFTNESS, dist)
}

/// Accumulated stroke coverage for a pixel of the given `luminance`.
///
/// A four-tier ramp: no strokes above `thresholds[0]`, then the `angles[0]` set
/// is added below it, the `angles[1]` set below `thresholds[1]`, and the
/// `angles[2]` set below `thresholds[2]`. Sets combine by `max` (a pixel is inked
/// if *any* active stroke covers it), so coverage is monotonically
/// non-decreasing as the pixel darkens and always lands in `[0, 1]`.
#[must_use]
pub fn hatch_coverage(luminance: f32, screen_pos: Vec2, params: &HatchingParams) -> f32 {
    let mut coverage = 0.0_f32;
    if luminance < params.thresholds[0] {
        let r = rotate2d(screen_pos, params.angles[0]);
        coverage = coverage.max(line_coverage(r, params.frequency, params.thickness));
    }
    if luminance < params.thresholds[1] {
        let r = rotate2d(screen_pos, params.angles[1]);
        coverage = coverage.max(line_coverage(r, params.frequency, params.thickness));
    }
    if luminance < params.thresholds[2] {
        let r = rotate2d(screen_pos, params.angles[2]);
        coverage = coverage.max(line_coverage(r, params.frequency, params.thickness));
    }
    coverage
}

/// Layered cross-hatching controls. `Default` is disabled, so hatching a pixel
/// with the default params returns the scene unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HatchingParams {
    /// Master enable. When `false` [`apply_hatching`] passes the scene through.
    pub enabled: bool,
    /// Stroke sets per unit of rotated coordinate (line density).
    pub frequency: f32,
    /// Stroke half-width in fractional-period units.
    pub thickness: f32,
    /// The three stroke directions (radians), added darkest-last.
    pub angles: [f32; 3],
    /// Descending luminance tiers at which each stroke set switches on.
    pub thresholds: [f32; 3],
}

impl Default for HatchingParams {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency: 40.0,
            thickness: 0.15,
            angles: [
                core::f32::consts::FRAC_PI_4,
                -core::f32::consts::FRAC_PI_4,
                0.0,
            ],
            thresholds: [0.75, 0.5, 0.25],
        }
    }
}

/// Composite the cross-hatch ink over the paper for one pixel.
///
/// While disabled the `scene` colour passes through unchanged (identity).
/// Otherwise the scene's [`luminance`] drives [`hatch_coverage`] and the result
/// is `bg.lerp(fg, coverage)`: paper (`bg`) where uncovered, ink (`fg`) on the
/// strokes, anti-aliased between. Mirrors the WESL twin arm-for-arm.
#[must_use]
pub fn apply_hatching(
    scene: Vec3,
    screen_pos: Vec2,
    params: &HatchingParams,
    fg: Vec3,
    bg: Vec3,
) -> Vec3 {
    if !params.enabled {
        return scene;
    }
    let lum = luminance(scene);
    let coverage = hatch_coverage(lum, screen_pos, params);
    bg.lerp(fg, coverage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn enabled_params() -> HatchingParams {
        HatchingParams {
            enabled: true,
            ..HatchingParams::default()
        }
    }

    // --- luminance ---

    #[test]
    fn luminance_white_is_one() {
        approx(luminance(Vec3::ONE), 1.0);
    }

    #[test]
    fn luminance_isolates_channels() {
        approx(luminance(Vec3::new(1.0, 0.0, 0.0)), 0.2126);
        approx(luminance(Vec3::new(0.0, 1.0, 0.0)), 0.7152);
        approx(luminance(Vec3::new(0.0, 0.0, 1.0)), 0.0722);
    }

    #[test]
    fn luminance_of_grey_is_scalar() {
        approx(luminance(Vec3::splat(0.5)), 0.5);
    }

    // --- rotate2d ---

    #[test]
    fn rotate2d_zero_is_identity() {
        let p = Vec2::new(0.3, -0.7);
        let r = rotate2d(p, 0.0);
        approx(r.x, p.x);
        approx(r.y, p.y);
    }

    #[test]
    fn rotate2d_quarter_turn_maps_x_to_y() {
        // +90deg takes (1, 0) -> (0, 1).
        let r = rotate2d(Vec2::new(1.0, 0.0), core::f32::consts::FRAC_PI_2);
        approx(r.x, 0.0);
        approx(r.y, 1.0);
    }

    #[test]
    fn rotate2d_preserves_length() {
        let p = Vec2::new(1.0, 2.0);
        let r = rotate2d(p, 0.9);
        let len_p = (p.x * p.x + p.y * p.y).sqrt();
        let len_r = (r.x * r.x + r.y * r.y).sqrt();
        approx(len_p, len_r);
    }

    // --- line_coverage ---

    #[test]
    fn line_coverage_on_line_is_full() {
        // coord = 0 sits exactly on a stroke -> full coverage.
        approx(line_coverage(Vec2::new(0.0, 0.0), 1.0, 0.15), 1.0);
    }

    #[test]
    fn line_coverage_between_lines_is_zero() {
        // coord = 0.5 is the midpoint between strokes -> no coverage.
        approx(line_coverage(Vec2::new(0.0, 0.5), 1.0, 0.15), 0.0);
    }

    #[test]
    fn line_coverage_is_periodic() {
        // Integer coords repeat the on-stroke value.
        approx(line_coverage(Vec2::new(0.0, 1.0), 1.0, 0.15), 1.0);
        approx(line_coverage(Vec2::new(0.0, 3.0), 1.0, 0.15), 1.0);
    }

    // --- hatch_coverage ---

    #[test]
    fn hatch_coverage_bright_has_no_lines() {
        // Above the brightest threshold no stroke set is active.
        let p = enabled_params();
        approx(hatch_coverage(0.9, Vec2::new(0.13, 0.27), &p), 0.0);
    }

    #[test]
    fn hatch_coverage_dark_inks_the_origin() {
        // At the origin every rotated coord is 0 (on a stroke), so any active
        // set gives full coverage; luminance 0.1 activates all three.
        let p = enabled_params();
        approx(hatch_coverage(0.1, Vec2::ZERO, &p), 1.0);
    }

    #[test]
    fn hatch_coverage_first_tier_switches_on_below_threshold() {
        // Just below thresholds[0] the first set inks the origin.
        let p = enabled_params();
        approx(hatch_coverage(0.6, Vec2::ZERO, &p), 1.0);
        approx(hatch_coverage(0.8, Vec2::ZERO, &p), 0.0);
    }

    #[test]
    fn hatch_coverage_monotonic_in_darkness() {
        // Darker pixels activate a superset of stroke directions, so coverage
        // never decreases as luminance falls.
        let p = enabled_params();
        let pos = Vec2::new(0.137, 0.291);
        let bright = hatch_coverage(0.9, pos, &p);
        let mid = hatch_coverage(0.6, pos, &p);
        let dark = hatch_coverage(0.1, pos, &p);
        assert!(mid >= bright, "{mid} < {bright}");
        assert!(dark >= mid, "{dark} < {mid}");
    }

    #[test]
    fn hatch_coverage_stays_in_unit_range() {
        let p = enabled_params();
        let cov = hatch_coverage(0.05, Vec2::new(0.41, 0.62), &p);
        assert!((0.0..=1.0).contains(&cov), "coverage {cov} out of range");
    }

    // --- params ---

    #[test]
    fn default_params_are_disabled() {
        let p = HatchingParams::default();
        assert!(!p.enabled);
        approx(p.frequency, 40.0);
        approx(p.thickness, 0.15);
    }

    // --- apply_hatching ---

    #[test]
    fn apply_disabled_is_identity() {
        let scene = Vec3::new(0.2, 0.5, 0.9);
        let out = apply_hatching(
            scene,
            Vec2::new(0.1, 0.2),
            &HatchingParams::default(),
            Vec3::ZERO,
            Vec3::ONE,
        );
        approx(out.x, scene.x);
        approx(out.y, scene.y);
        approx(out.z, scene.z);
    }

    #[test]
    fn apply_bright_scene_returns_background() {
        // White scene -> no strokes -> pure paper (bg).
        let fg = Vec3::ZERO;
        let bg = Vec3::ONE;
        let out = apply_hatching(Vec3::ONE, Vec2::new(0.13, 0.27), &enabled_params(), fg, bg);
        approx(out.x, bg.x);
        approx(out.y, bg.y);
        approx(out.z, bg.z);
    }

    #[test]
    fn apply_dark_origin_returns_foreground() {
        // Black scene at the origin -> full coverage -> pure ink (fg).
        let fg = Vec3::ZERO;
        let bg = Vec3::ONE;
        let out = apply_hatching(Vec3::ZERO, Vec2::ZERO, &enabled_params(), fg, bg);
        approx(out.x, fg.x);
        approx(out.y, fg.y);
        approx(out.z, fg.z);
    }
}
