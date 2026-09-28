//! Backend-neutral CPU golden for the NPR "halftone" (manga screentone) style.
//!
//! Halftone is a stylized post-process that reproduces the printer's dot screen
//! of comics and newsprint: the frame is divided into a regular grid of
//! `cell_size`-pixel cells (optionally rotated by `angle` so the dot lattice
//! runs at the classic 45-degree screen angle), and inside each cell a single
//! dot is grown or shrunk by the local tone. Dark tones grow a large dot and
//! bright tones shrink it, so the average coverage of the foreground ink tracks
//! the original luminance while the image reads as discrete dots up close.
//!
//! The whole module is split into small pure functions so the GPU twin in
//! `prism_render_scene/src/shaders/halftone.wesl` can mirror it arm-for-arm
//! (same function split, same constants, same operation order):
//!
//! * [`luminance`] — Rec. 709 luma of the scene colour.
//! * [`rotate2d`] — rotate the screen position into the dot-screen frame.
//! * [`cell_coord`] — position within a cell, relative to its centre.
//! * [`dot_radius_from_tone`] — dark tones -> large dot, bright -> small.
//! * [`dot_coverage`] — anti-aliased inside/outside test for the dot disc.
//! * [`apply_halftone`] — the full pass, disabled -> identity.
//!
//! Only `sin`/`cos` are transcendental; they go through [`bevy_math::ops`] for
//! libm determinism (the `clippy.toml` disallows the `f32` intrinsics). The
//! smoothstep is written out as `t*t*(3 - 2t)` and squares as `x*x` so the CPU
//! golden and GPU twin stay bit-identical; `sqrt`/`floor`/`fract`/`clamp` are
//! permitted and used directly.

use bevy_math::{ops, Vec2, Vec3};

/// Rec. 709 luma weights, shared with the GPU twin.
pub const HALFTONE_LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Darkest-tone dot radius as a fraction of `cell_size` (the on-tone radius is
/// this times `cell_size`). `0.5` grows the black dot out to the cell's edge
/// midpoint at full ink.
pub const HALFTONE_MAX_DOT_RADIUS: f32 = 0.5;

/// Artist controls for the halftone screen.
///
/// [`Default`] is the identity pass (`enabled = false`): the scene colour is
/// returned untouched, so the committed ABI is inert until a caller enables it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalftoneParams {
    /// Side length of a screen cell, in pixels.
    pub cell_size: f32,
    /// Rotation of the dot lattice, in radians (the classic screen uses ~45deg).
    pub angle: f32,
    /// When `false` the pass is the identity (returns the scene colour).
    pub enabled: bool,
}

impl Default for HalftoneParams {
    fn default() -> Self {
        Self {
            cell_size: 4.0,
            angle: 0.0,
            enabled: false,
        }
    }
}

/// Rec. 709 relative luminance of a linear colour.
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    c.x * HALFTONE_LUMA_WEIGHTS[0] + c.y * HALFTONE_LUMA_WEIGHTS[1] + c.z * HALFTONE_LUMA_WEIGHTS[2]
}

/// Rotate `p` by `angle` radians (counter-clockwise) into the dot-screen frame.
///
/// Uses [`bevy_math::ops::sin`] / [`bevy_math::ops::cos`] for libm determinism.
/// It is a rigid rotation, so it preserves length.
#[must_use]
pub fn rotate2d(p: Vec2, angle: f32) -> Vec2 {
    let s = ops::sin(angle);
    let cs = ops::cos(angle);
    Vec2::new(p.x * cs - p.y * s, p.x * s + p.y * cs)
}

/// Position of `screen_pos` within its cell, measured from the cell centre.
///
/// The result lies in `[-0.5, 0.5) * cell_size` on each axis and is periodic in
/// `cell_size` (shifting `screen_pos` by a whole cell leaves it unchanged). Uses
/// `floor` (fract) which the lint policy permits.
#[must_use]
pub fn cell_coord(screen_pos: Vec2, cell_size: f32) -> Vec2 {
    let cs = cell_size.max(1.0e-6);
    let normalized = screen_pos / cs;
    let frac = normalized - normalized.floor();
    (frac - Vec2::splat(0.5)) * cs
}

/// Dot radius (as a fraction of `cell_size`) for a local tone.
///
/// Dark tones (`luma -> 0`) grow the dot to [`HALFTONE_MAX_DOT_RADIUS`]; bright
/// tones (`luma -> 1`) shrink it to `0`. Monotonically decreasing in `luma`.
/// Uses `sqrt` (permitted) so coverage area is roughly linear in `1 - luma`.
#[must_use]
pub fn dot_radius_from_tone(luma: f32) -> f32 {
    let t = luma.clamp(0.0, 1.0);
    (1.0 - t).sqrt() * HALFTONE_MAX_DOT_RADIUS
}

/// Hermite smoothstep written out as `t*t*(3 - 2t)` (t clamped to `[0, 1]`), so
/// it is bit-identical with the shader twin. The degenerate `edge0 == edge1`
/// case is guarded so the division stays finite.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    let d = if denom.abs() > 1.0e-6 { denom } else { 1.0e-6 };
    let t = ((x - edge0) / d).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Anti-aliased ink coverage of a dot of `radius` at distance `dist` from its
/// centre, with a soft edge of half-width `aa`.
///
/// Returns `1.0` (solid ink) well inside the dot (`dist << radius`), `0.0`
/// (background) well outside (`dist >> radius`), and a smooth transition across
/// `[radius - aa, radius + aa]` (`0.5` at `dist == radius`).
#[must_use]
pub fn dot_coverage(dist: f32, radius: f32, aa: f32) -> f32 {
    let a = aa.max(1.0e-6);
    1.0 - smoothstep(radius - a, radius + a, dist)
}

/// Apply the halftone screen to a scene colour at `screen_pos`.
///
/// When `params.enabled` is `false` this is the identity (returns `scene`).
/// Otherwise it takes `luma = luminance(scene)`, rotates `screen_pos` into the
/// dot-screen frame, finds the position within the cell, grows a dot from the
/// tone and returns `bg.lerp(fg, coverage)` — the foreground ink over the paper.
#[must_use]
pub fn apply_halftone(
    scene: Vec3,
    screen_pos: Vec2,
    params: &HalftoneParams,
    fg: Vec3,
    bg: Vec3,
) -> Vec3 {
    if !params.enabled {
        return scene;
    }
    let luma = luminance(scene);
    let rotated = rotate2d(screen_pos, params.angle);
    let local = cell_coord(rotated, params.cell_size);
    let dist = local.length();
    let radius = dot_radius_from_tone(luma) * params.cell_size;
    let coverage = dot_coverage(dist, radius, 1.0);
    bg.lerp(fg, coverage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "left = {a}, right = {b}");
    }

    #[test]
    fn default_is_disabled_identity() {
        let params = HalftoneParams::default();
        assert!(!params.enabled);
        let scene = Vec3::new(0.2, 0.5, 0.9);
        let out = apply_halftone(
            scene,
            Vec2::new(3.0, 7.0),
            &params,
            Vec3::ZERO,
            Vec3::ONE,
        );
        approx(out.x, scene.x);
        approx(out.y, scene.y);
        approx(out.z, scene.z);
    }

    #[test]
    fn default_param_values() {
        let params = HalftoneParams::default();
        approx(params.cell_size, 4.0);
        approx(params.angle, 0.0);
    }

    #[test]
    fn luminance_rec709() {
        approx(luminance(Vec3::ONE), 1.0);
        approx(luminance(Vec3::new(1.0, 0.0, 0.0)), 0.2126);
        approx(luminance(Vec3::new(0.0, 1.0, 0.0)), 0.7152);
        approx(luminance(Vec3::new(0.0, 0.0, 1.0)), 0.0722);
    }

    #[test]
    fn luminance_weights_sum_to_one() {
        let sum = HALFTONE_LUMA_WEIGHTS[0] + HALFTONE_LUMA_WEIGHTS[1] + HALFTONE_LUMA_WEIGHTS[2];
        approx(sum, 1.0);
    }

    #[test]
    fn rotate2d_zero_is_identity() {
        let p = Vec2::new(1.0, 2.0);
        let r = rotate2d(p, 0.0);
        approx(r.x, 1.0);
        approx(r.y, 2.0);
    }

    #[test]
    fn rotate2d_ninety_degrees() {
        let r = rotate2d(Vec2::new(1.0, 0.0), FRAC_PI_2);
        approx(r.x, 0.0);
        approx(r.y, 1.0);
    }

    #[test]
    fn rotate2d_preserves_length() {
        let p = Vec2::new(3.0, -4.0);
        let r = rotate2d(p, 0.9);
        approx(r.length(), p.length());
    }

    #[test]
    fn cell_coord_within_range() {
        let cs = 8.0;
        for i in 0..64 {
            let x = i as f32 * 0.37;
            let y = i as f32 * -0.71 + 3.0;
            let c = cell_coord(Vec2::new(x, y), cs);
            assert!(c.x >= -0.5 * cs - 1.0e-4 && c.x < 0.5 * cs + 1.0e-4, "x = {}", c.x);
            assert!(c.y >= -0.5 * cs - 1.0e-4 && c.y < 0.5 * cs + 1.0e-4, "y = {}", c.y);
        }
    }

    #[test]
    fn cell_coord_periodic() {
        let cs = 5.0;
        let base = Vec2::new(1.3, 2.7);
        let a = cell_coord(base, cs);
        let b = cell_coord(base + Vec2::new(cs * 3.0, cs * -2.0), cs);
        approx(a.x, b.x);
        approx(a.y, b.y);
    }

    #[test]
    fn cell_coord_center_is_origin() {
        let cs = 10.0;
        // Cell (0,0) centre sits at (5, 5).
        let c = cell_coord(Vec2::new(5.0, 5.0), cs);
        approx(c.x, 0.0);
        approx(c.y, 0.0);
    }

    #[test]
    fn dot_radius_dark_is_max_bright_is_zero() {
        approx(dot_radius_from_tone(0.0), HALFTONE_MAX_DOT_RADIUS);
        approx(dot_radius_from_tone(1.0), 0.0);
    }

    #[test]
    fn dot_radius_monotonic_decreasing() {
        let mut prev = dot_radius_from_tone(0.0);
        for i in 1..=20 {
            let luma = i as f32 / 20.0;
            let r = dot_radius_from_tone(luma);
            assert!(r <= prev + 1.0e-6, "radius increased at luma = {luma}");
            prev = r;
        }
    }

    #[test]
    fn dot_radius_dark_bigger_than_bright() {
        assert!(dot_radius_from_tone(0.1) > dot_radius_from_tone(0.9));
    }

    #[test]
    fn dot_radius_clamps_out_of_range() {
        approx(dot_radius_from_tone(-1.0), HALFTONE_MAX_DOT_RADIUS);
        approx(dot_radius_from_tone(2.0), 0.0);
    }

    #[test]
    fn dot_coverage_inside_is_solid() {
        approx(dot_coverage(0.0, 5.0, 1.0), 1.0);
        approx(dot_coverage(3.0, 5.0, 1.0), 1.0);
    }

    #[test]
    fn dot_coverage_outside_is_background() {
        approx(dot_coverage(10.0, 5.0, 1.0), 0.0);
        approx(dot_coverage(6.5, 5.0, 1.0), 0.0);
    }

    #[test]
    fn dot_coverage_midpoint_is_half() {
        approx(dot_coverage(5.0, 5.0, 1.0), 0.5);
    }

    #[test]
    fn dot_coverage_transition_monotonic() {
        // As distance grows across the edge, coverage falls monotonically.
        let radius = 5.0;
        let aa = 1.0;
        let mut prev = dot_coverage(radius - aa, radius, aa);
        let mut d = radius - aa;
        while d <= radius + aa {
            let cov = dot_coverage(d, radius, aa);
            assert!(cov <= prev + 1.0e-6, "coverage increased at dist = {d}");
            prev = cov;
            d += 0.1;
        }
    }

    #[test]
    fn smoothstep_endpoints() {
        approx(smoothstep(0.0, 1.0, -1.0), 0.0);
        approx(smoothstep(0.0, 1.0, 0.0), 0.0);
        approx(smoothstep(0.0, 1.0, 0.5), 0.5);
        approx(smoothstep(0.0, 1.0, 1.0), 1.0);
        approx(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn apply_dark_covers_more_than_bright() {
        // cell_size 10, angle 0 (identity rotate); point (8, 5) -> local (3, 0),
        // dist = 3. Dark radius = 0.5*10 = 5 (dist inside -> ink); bright
        // (luma 0.9) radius = sqrt(0.1)*0.5*10 ~= 1.58 (dist outside -> paper).
        let params = HalftoneParams {
            cell_size: 10.0,
            angle: 0.0,
            enabled: true,
        };
        let pos = Vec2::new(8.0, 5.0);
        let fg = Vec3::ZERO;
        let bg = Vec3::ONE;

        let dark = apply_halftone(Vec3::splat(0.0), pos, &params, fg, bg);
        let bright = apply_halftone(Vec3::splat(0.9), pos, &params, fg, bg);

        // Dark -> solid ink (near fg=black); bright -> paper (near bg=white).
        assert!(dark.x < 0.1, "dark should be inked: {}", dark.x);
        assert!(bright.x > 0.9, "bright should be paper: {}", bright.x);
    }

    #[test]
    fn apply_enabled_returns_mix_of_fg_bg() {
        // At the cell centre (dist 0) coverage is 1 for any positive radius, so
        // the result is the foreground ink exactly.
        let params = HalftoneParams {
            cell_size: 10.0,
            angle: 0.0,
            enabled: true,
        };
        let fg = Vec3::new(0.1, 0.2, 0.3);
        let bg = Vec3::new(0.9, 0.8, 0.7);
        let out = apply_halftone(Vec3::splat(0.25), Vec2::new(5.0, 5.0), &params, fg, bg);
        approx(out.x, fg.x);
        approx(out.y, fg.y);
        approx(out.z, fg.z);
    }
}
