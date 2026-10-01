//! Screen-space occlusion/emission mask build with depth and sky gating
//! (CPU golden reference).
//!
//! The radial scatter in [`super::radial`] marches an *emission mask*: a scalar
//! field over the screen that says how much light energy leaves each texel
//! toward the camera.  This module builds that mask from the two cheap signals
//! a deferred renderer already has — a depth buffer and the shaded scene colour
//! (or its luminance):
//!
//! * **Sky texels emit.**  Where depth marks the far plane (no geometry), the
//!   light's own glow reaches the camera, so the texel emits its luminance.
//! * **Geometry occludes.**  Where depth marks a surface, the shaft is blocked,
//!   so the texel emits `0` (black) and will carve a silhouette into the march.
//! * **The sun disk injects a core.**  A bright radial falloff is added around
//!   the screen-space sun position, but only through sky texels, so an occluder
//!   in front of the sun correctly swallows the core.
//!
//! # Conventions
//! * Depth is a scalar; its orientation is selected by [`DepthRange`] so both
//!   standard (`far = 1`) and reverse-`Z` (`far = 0`) buffers are supported.
//! * `uv` and the sun position are [`Vec2`] in normalized `[0, 1]` screen space;
//!   colours are linear-RGB [`Vec3`].
//! * The emission is always in `[0, max_emission]`, finite, and non-negative;
//!   every parameter is clamped to its valid range and degenerate inputs fall
//!   back to "fully occluded" (`0`).
//! * Transcendental maths goes through [`bevy_math::ops`]; the sun falloff uses
//!   [`ops::exp`].  Every function is deterministic: no RNG, I/O, GPU, `unsafe`.

use alloc::vec::Vec;

use bevy_math::{ops, Vec2, Vec3};

/// Largest exponent fed to [`ops::exp`] for the sun falloff; `exp(-50)` is far
/// below any meaningful intensity, so clamping keeps the result finite.
const MAX_EXPONENT: f32 = 50.0;

/// Orientation of the depth buffer used to decide what counts as "sky".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthRange {
    /// Standard depth: near `= 0`, far `= 1`; sky is at or beyond the threshold.
    Standard,
    /// Reverse-`Z` depth: near `= 1`, far `= 0`; sky is at or below the threshold.
    ReverseZ,
}

impl Default for DepthRange {
    #[inline]
    fn default() -> Self {
        Self::Standard
    }
}

impl DepthRange {
    /// Signed "sky-ness" of `depth` relative to `threshold`.
    ///
    /// Returns `depth - threshold` for standard depth and `threshold - depth`
    /// for reverse-`Z`, so a positive result always means "further than the
    /// sky threshold" regardless of orientation.
    #[inline]
    fn skyness(self, depth: f32, threshold: f32) -> f32 {
        match self {
            Self::Standard => depth - threshold,
            Self::ReverseZ => threshold - depth,
        }
    }
}

/// Parameters that turn depth + colour into an emission mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionConfig {
    /// Depth value separating geometry from sky.
    pub sky_depth: f32,
    /// Half-width of the soft sky transition band (in depth units); `0` gives a
    /// hard binary gate.
    pub sky_softness: f32,
    /// Depth-buffer orientation.
    pub range: DepthRange,
    /// Screen-space sun position for the injected disk.
    pub sun_uv: Vec2,
    /// Radius of the sun disk in normalized screen units (full falloff distance).
    pub sun_radius: f32,
    /// Peak intensity added at the sun centre (through sky only).
    pub sun_intensity: f32,
    /// Gaussian sharpness of the sun falloff; larger is a tighter core.
    pub sun_sharpness: f32,
    /// Upper clamp applied to the final emission.
    pub max_emission: f32,
}

impl Default for OcclusionConfig {
    #[inline]
    fn default() -> Self {
        Self {
            sky_depth: 0.999,
            sky_softness: 0.0005,
            range: DepthRange::Standard,
            sun_uv: Vec2::new(0.5, 0.2),
            sun_radius: 0.35,
            sun_intensity: 1.0,
            sun_sharpness: 6.0,
            max_emission: 4.0,
        }
    }
}

/// Rec. 709 relative luminance of a linear-RGB colour, clamped non-negative.
///
/// Non-finite channels are treated as `0` so a NaN texel cannot poison the mask.
#[inline]
pub fn luminance(color: Vec3) -> f32 {
    let r = sanitize_scalar(color.x);
    let g = sanitize_scalar(color.y);
    let b = sanitize_scalar(color.z);
    (0.2126 * r + 0.7152 * g + 0.0722 * b).max(0.0)
}

/// Soft sky-visibility gate in `[0, 1]`: `0` for geometry, `1` for sky.
///
/// Uses a smoothstep over `[-softness, +softness]` around the sky threshold so
/// the geometry silhouette is anti-aliased.  With `softness == 0` the gate is a
/// hard binary step.  Non-finite depth is treated as geometry (`0`).
#[inline]
pub fn sky_gate(depth: f32, config: &OcclusionConfig) -> f32 {
    if !depth.is_finite() {
        return 0.0;
    }
    let s = config.range.skyness(depth, config.sky_depth);
    let soft = sanitize_scalar(config.sky_softness).max(0.0);
    if soft <= 0.0 {
        return if s >= 0.0 { 1.0 } else { 0.0 };
    }
    // Map s in [-soft, +soft] -> [0, 1] then smoothstep.
    let t = ((s + soft) / (2.0 * soft)).clamp(0.0, 1.0);
    smoothstep01(t)
}

/// Hard binary occlusion mask: `1.0` for sky, `0.0` for geometry.
///
/// Equivalent to [`sky_gate`] with zero softness; handy when a crisp occluder
/// silhouette is wanted.  Non-finite depth is treated as geometry (`0`).
#[inline]
pub fn occlusion_mask_binary(depth: f32, config: &OcclusionConfig) -> f32 {
    if !depth.is_finite() {
        return 0.0;
    }
    if config.range.skyness(depth, config.sky_depth) >= 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Radial sun-disk intensity in `[0, 1]` at `uv`, before sky gating.
///
/// A normalized Gaussian of the distance from `sun_uv`: `1` at the centre and
/// decaying to (near) `0` at `sun_radius`.  A non-positive radius or non-finite
/// position yields `0`.
#[inline]
pub fn sun_disk(uv: Vec2, config: &OcclusionConfig) -> f32 {
    if !is_finite_vec2(uv) || !is_finite_vec2(config.sun_uv) {
        return 0.0;
    }
    let radius = sanitize_scalar(config.sun_radius);
    if radius <= 0.0 {
        return 0.0;
    }
    let sharp = sanitize_scalar(config.sun_sharpness).max(0.0);
    let d = (uv - config.sun_uv).length() / radius;
    // Gaussian falloff; the exponent is bounded before exp().
    let exponent = (sharp * d * d).min(MAX_EXPONENT);
    ops::exp(-exponent)
}

/// Builds the final emission for one texel from depth, scene colour, and the
/// injected sun disk.
///
/// The result is `sky_gate * (luminance(scene_color) + sun_intensity *
/// sun_disk)`, clamped to `[0, max_emission]`.  Gating by the sky visibility
/// guarantees geometry texels emit `0` even when they sit under the sun, so an
/// occluder in front of the sun carves a clean shadow into the shaft.
#[inline]
pub fn build_emission(depth: f32, scene_color: Vec3, uv: Vec2, config: &OcclusionConfig) -> f32 {
    let gate = sky_gate(depth, config);
    if gate <= 0.0 {
        return 0.0;
    }
    let sky_light = luminance(scene_color);
    let sun = sanitize_scalar(config.sun_intensity).max(0.0) * sun_disk(uv, config);
    let emission = gate * (sky_light + sun);
    clamp_emission(emission, config.max_emission)
}

/// Builds a full-resolution emission buffer from parallel depth and colour
/// buffers (row-major, `width * height`).
///
/// Each texel's `uv` is its centre in normalized screen space.  Mismatched or
/// degenerate buffer sizes yield an empty result.  Every entry is finite and in
/// `[0, max_emission]`.
pub fn build_emission_buffer(
    depth: &[f32],
    color: &[Vec3],
    width: usize,
    height: usize,
    config: &OcclusionConfig,
) -> Vec<f32> {
    let len = width.saturating_mul(height);
    if width == 0 || height == 0 || depth.len() < len || color.len() < len {
        return Vec::new();
    }
    let mut out = alloc::vec![0.0_f32; len];
    let inv_w = 1.0 / width as f32;
    let inv_h = 1.0 / height as f32;
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let uv = Vec2::new((x as f32 + 0.5) * inv_w, (y as f32 + 0.5) * inv_h);
            out[i] = build_emission(depth[i], color[i], uv, config);
        }
    }
    out
}

/// Clamps emission to `[0, max_emission]`, repairing non-finite values to `0`.
#[inline]
fn clamp_emission(v: f32, max_emission: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    let hi = if max_emission.is_finite() {
        max_emission.max(0.0)
    } else {
        f32::INFINITY
    };
    v.clamp(0.0, hi)
}

/// Cubic smoothstep on a value already clamped to `[0, 1]`.
#[inline]
fn smoothstep01(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Returns `v` when finite, else `0`.
#[inline]
fn sanitize_scalar(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Returns `true` when both components of `v` are finite.
#[inline]
fn is_finite_vec2(v: Vec2) -> bool {
    v.x.is_finite() && v.y.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> OcclusionConfig {
        OcclusionConfig::default()
    }

    #[test]
    fn luminance_is_rec709_weighted() {
        assert!((luminance(Vec3::new(1.0, 0.0, 0.0)) - 0.2126).abs() < 1e-6);
        assert!((luminance(Vec3::new(0.0, 1.0, 0.0)) - 0.7152).abs() < 1e-6);
        assert!((luminance(Vec3::new(0.0, 0.0, 1.0)) - 0.0722).abs() < 1e-6);
        // Non-finite channels are ignored.
        assert_eq!(luminance(Vec3::new(f32::NAN, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn standard_depth_sky_vs_geometry_gate() {
        let c = cfg();
        // Far plane -> sky -> gate 1.
        assert!((sky_gate(1.0, &c) - 1.0).abs() < 1e-6);
        // Near geometry -> gate 0.
        assert_eq!(sky_gate(0.5, &c), 0.0);
    }

    #[test]
    fn reverse_z_gate_is_inverted() {
        let c = OcclusionConfig { range: DepthRange::ReverseZ, sky_depth: 0.001, sky_softness: 0.0, ..cfg() };
        // Reverse-Z far plane is 0 -> sky.
        assert_eq!(sky_gate(0.0, &c), 1.0);
        // A near surface at 0.9 -> geometry.
        assert_eq!(sky_gate(0.9, &c), 0.0);
    }

    #[test]
    fn soft_gate_is_monotonic_across_the_band() {
        let c = cfg_soft();
        let mut prev = -1.0;
        let mut d = 0.3;
        while d <= 0.7 {
            let g = sky_gate(d, &c);
            assert!(g >= prev - 1e-6, "non-monotonic at d={d}: {g} < {prev}");
            assert!((0.0..=1.0).contains(&g));
            prev = g;
            d += 0.01;
        }
        // Mid-band is exactly the smoothstep midpoint.
        assert!((sky_gate(0.5, &c) - 0.5).abs() < 1e-6);
    }

    // Helper producing a config with a soft band but default sun, kept separate
    // so the struct-update syntax above reads cleanly.
    fn cfg_soft() -> OcclusionConfig {
        OcclusionConfig { sky_depth: 0.5, sky_softness: 0.1, ..OcclusionConfig::default() }
    }

    #[test]
    fn geometry_emits_zero_even_under_the_sun() {
        // A geometry texel exactly at the sun position must still emit nothing:
        // the sun core is gated by sky visibility.
        let c = cfg();
        let emission = build_emission(0.5, Vec3::splat(1.0), c.sun_uv, &c);
        assert_eq!(emission, 0.0);
    }

    #[test]
    fn sky_emits_scene_luminance_without_sun() {
        // Sky texel far from the sun emits (approximately) its luminance.
        let c = OcclusionConfig { sun_intensity: 0.0, ..cfg() };
        let col = Vec3::new(0.4, 0.4, 0.4);
        let emission = build_emission(1.0, col, Vec2::new(0.95, 0.95), &c);
        assert!((emission - luminance(col)).abs() < 1e-6, "emission={emission}");
    }

    #[test]
    fn sun_core_brighter_than_surrounding_sky() {
        let c = cfg();
        let at_sun = build_emission(1.0, Vec3::splat(0.2), c.sun_uv, &c);
        let away = build_emission(1.0, Vec3::splat(0.2), Vec2::new(0.95, 0.95), &c);
        assert!(at_sun > away, "at_sun={at_sun} away={away}");
    }

    #[test]
    fn sun_disk_decays_with_distance() {
        let c = cfg();
        assert!((sun_disk(c.sun_uv, &c) - 1.0).abs() < 1e-6);
        let near = sun_disk(c.sun_uv + Vec2::new(0.05, 0.0), &c);
        let far = sun_disk(c.sun_uv + Vec2::new(0.2, 0.0), &c);
        assert!(far < near && near < 1.0, "near={near} far={far}");
    }

    #[test]
    fn emission_is_clamped_to_max() {
        let c = OcclusionConfig { sun_intensity: 100.0, max_emission: 2.0, ..cfg() };
        let emission = build_emission(1.0, Vec3::splat(1.0), c.sun_uv, &c);
        assert!(emission <= 2.0 + 1e-6, "emission={emission}");
    }

    #[test]
    fn binary_mask_is_crisp() {
        let c = cfg();
        assert_eq!(occlusion_mask_binary(1.0, &c), 1.0);
        assert_eq!(occlusion_mask_binary(0.5, &c), 0.0);
        assert_eq!(occlusion_mask_binary(f32::NAN, &c), 0.0);
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        let c = cfg();
        assert_eq!(build_emission(f32::NAN, Vec3::splat(1.0), c.sun_uv, &c), 0.0);
        assert_eq!(sun_disk(Vec2::new(f32::NAN, 0.0), &c), 0.0);
        let zero_radius = OcclusionConfig { sun_radius: 0.0, ..c };
        assert_eq!(sun_disk(c.sun_uv, &zero_radius), 0.0);
    }

    #[test]
    fn build_emission_buffer_matches_per_texel() {
        let c = cfg();
        let w = 4;
        let h = 3;
        let depth = alloc::vec![1.0_f32; w * h];
        let color = alloc::vec![Vec3::splat(0.5); w * h];
        let buf = build_emission_buffer(&depth, &color, w, h, &c);
        assert_eq!(buf.len(), w * h);
        let inv_w = 1.0 / w as f32;
        let inv_h = 1.0 / h as f32;
        for y in 0..h {
            for x in 0..w {
                let uv = Vec2::new((x as f32 + 0.5) * inv_w, (y as f32 + 0.5) * inv_h);
                let want = build_emission(1.0, Vec3::splat(0.5), uv, &c);
                assert!((buf[y * w + x] - want).abs() < 1e-6);
            }
        }
        assert!(build_emission_buffer(&depth, &color, 0, 0, &c).is_empty());
    }
}
