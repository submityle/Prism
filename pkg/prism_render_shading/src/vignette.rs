//! Backend-neutral CPU golden for AAA vignette (lens shading / darkened edges).
//!
//! Vignette is a shared post-processing base that runs on the resolved,
//! pre-exposed *linear* HDR radiance (see [`crate::exposure`]) alongside the
//! rest of the post chain. It darkens the frame toward its edges, either to
//! reproduce the physical light fall-off of a real lens or as an artistic
//! framing device that draws the eye to the centre. Every illumination model,
//! physical or stylized, writes into the same HDR buffer, so one vignette pass
//! serves them all.
//!
//! Two industry-standard models are provided, split into small pure functions
//! so the GPU twin in `shaders/vignette.wesl` can mirror them arm-for-arm:
//!
//! * **Natural optical fall-off (`cos^4` law).** Real lenses lose irradiance
//!   toward the edge of the image circle by the fourth power of the cosine of
//!   the field angle: `factor = cos(theta)^4`, with
//!   `cos(theta) = f / sqrt(f*f + r*r)` for an off-axis radius `r` and a focal
//!   ratio `f`. The fourth power is evaluated as `c2 = c*c; c2*c2` to stay off
//!   the disallowed `f32::powi` / `f32::powf` path (`f32::sqrt` is permitted).
//! * **Artistic vignette (`Unity` `PPv2` style).** A `smoothstep` fall-off over
//!   a distance that blends a circular (Euclidean) and square (Chebyshev)
//!   metric by a `roundness` knob, scaled by an artist `intensity`, with an
//!   `aspect_ratio` correction on the horizontal axis. `smoothstep` is written
//!   out as `3t^2 - 2t^3` so it is bit-identical with the shader twin.
//!
//! The whole module is polynomial plus `sqrt` (no transcendental, so no
//! `bevy_math::ops` import is needed) and is mirrored arm-for-arm by
//! `shaders/vignette.wesl` (same function split, same constants, same operation
//! order) so the CPU golden and the GPU twin agree.

/// Vignette model selector. The discriminants match the `u32` mode the shader
/// twin branches on, so the CPU golden and GPU twin stay in lock-step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VignetteMode {
    /// Physical `cos^4` lens fall-off (see [`natural_falloff`]).
    Natural = 0,
    /// Artist-controlled `smoothstep` vignette (see [`artistic_falloff`]).
    Artistic = 1,
}

impl VignetteMode {
    /// The `u32` discriminant the shader twin uses to branch on the mode.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// The `cos^4` optical fall-off given the cosine of the field angle.
///
/// Evaluated as `c2 = c*c; c2*c2` to avoid the disallowed power intrinsics. At
/// `cos(theta) = 1` (on-axis) it returns `1`; at `cos(theta) = 0` (grazing) it
/// returns `0`, and it is monotonically increasing in `cos(theta)` over `[0, 1]`.
#[must_use]
pub fn natural_vignette(cos_theta: f32) -> f32 {
    let c2 = cos_theta * cos_theta;
    c2 * c2
}

/// Natural `cos^4` fall-off at a normalised screen coordinate `uv` in `[0, 1]`.
///
/// The optical axis is the frame centre `(0.5, 0.5)`; `r` is the distance from
/// it and `focal_ratio` is the (normalised) lens focal length. Larger
/// `focal_ratio` (a longer lens) flattens the fall-off toward `1`. The centre
/// returns `1` and the value decreases with radius.
#[must_use]
pub fn natural_falloff(uv: [f32; 2], focal_ratio: f32) -> f32 {
    let dx = uv[0] - 0.5;
    let dy = uv[1] - 0.5;
    let r2 = dx * dx + dy * dy;
    let cos_theta = focal_ratio / (focal_ratio * focal_ratio + r2).sqrt();
    natural_vignette(cos_theta)
}

/// Hermite `smoothstep`, written `3t^2 - 2t^3` for bit-identity with the shader
/// twin (`t` clamped to `[0, 1]`). Degenerate `edge0 == edge1` is guarded so the
/// division never produces a non-finite result.
#[must_use]
pub fn vignette_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    let denom = if denom.abs() > 1.0e-6 { denom } else { 1.0e-6 };
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Artist vignette fall-off at `uv`, in the `Unity` `PPv2` idiom.
///
/// `center` is the darkening origin, `intensity` scales the darkening,
/// `smoothness` sets the transition width, `roundness` blends a square
/// (`0`, Chebyshev) and a circular (`1`, Euclidean) shape, and `aspect_ratio`
/// corrects the horizontal axis. Coordinates are mapped to `[-1, 1]` about the
/// centre so the frame edge sits near distance `1`. Returns a multiplier in
/// `[0, 1]`; `intensity = 0` returns `1` (identity) everywhere.
#[must_use]
pub fn artistic_falloff(
    uv: [f32; 2],
    center: [f32; 2],
    intensity: f32,
    smoothness: f32,
    roundness: f32,
    aspect_ratio: f32,
) -> f32 {
    let dx = (uv[0] - center[0]) * 2.0 * aspect_ratio;
    let dy = (uv[1] - center[1]) * 2.0;

    let circular = (dx * dx + dy * dy).sqrt();
    let square = dx.abs().max(dy.abs());
    let dist = square + (circular - square) * roundness;

    let s = if smoothness > 1.0e-4 { smoothness } else { 1.0e-4 };
    let edge0 = 1.0 - s;
    let edge1 = 1.0 + s;
    let t = vignette_smoothstep(edge0, edge1, dist);

    1.0 - (t * intensity).clamp(0.0, 1.0)
}

/// Multiply linear RGB radiance by a scalar vignette `factor` in `[0, 1]`.
/// `factor = 1` is the identity; `factor = 0` blacks the pixel out.
#[must_use]
pub fn apply_vignette(rgb: [f32; 3], factor: f32) -> [f32; 3] {
    [rgb[0] * factor, rgb[1] * factor, rgb[2] * factor]
}

/// Artist controls for the vignette pass. `Default` is disabled (`intensity = 0`
/// in the artistic model), so the default factor is `1` (identity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteParams {
    /// Which fall-off model to evaluate.
    pub mode: VignetteMode,
    /// Darkening origin for the artistic model. Default frame centre.
    pub center: [f32; 2],
    /// Artistic darkening strength. Neutral `0` (identity).
    pub intensity: f32,
    /// Artistic transition width. Default `0.5`.
    pub smoothness: f32,
    /// Extra artistic transition softness, added to `smoothness`. Default `0`.
    pub feather: f32,
    /// Square (`0`) to circular (`1`) shape blend. Default `1` (circular).
    pub roundness: f32,
    /// Horizontal aspect correction for the artistic model. Default `1`.
    pub aspect_ratio: f32,
    /// Focal ratio for the natural `cos^4` model. Default `1`.
    pub focal_ratio: f32,
}

impl Default for VignetteParams {
    fn default() -> Self {
        Self {
            mode: VignetteMode::Artistic,
            center: [0.5, 0.5],
            intensity: 0.0,
            smoothness: 0.5,
            feather: 0.0,
            roundness: 1.0,
            aspect_ratio: 1.0,
            focal_ratio: 1.0,
        }
    }
}

/// Evaluate the vignette factor for `uv` under `params`, dispatching on the
/// selected [`VignetteMode`]. The artistic transition width is `smoothness`
/// plus `feather`.
#[must_use]
pub fn vignette_factor(uv: [f32; 2], params: &VignetteParams) -> f32 {
    match params.mode {
        VignetteMode::Natural => natural_falloff(uv, params.focal_ratio),
        VignetteMode::Artistic => artistic_falloff(
            uv,
            params.center,
            params.intensity,
            params.smoothness + params.feather,
            params.roundness,
            params.aspect_ratio,
        ),
    }
}

/// Apply the full vignette to `rgb` at `uv` under `params`
/// (`rgb * vignette_factor(uv, params)`).
#[must_use]
pub fn apply_vignette_params(rgb: [f32; 3], uv: [f32; 2], params: &VignetteParams) -> [f32; 3] {
    apply_vignette(rgb, vignette_factor(uv, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    // --- natural cos^4 ---

    #[test]
    fn natural_vignette_on_axis_is_one() {
        approx(natural_vignette(1.0), 1.0);
    }

    #[test]
    fn natural_vignette_grazing_is_zero() {
        approx(natural_vignette(0.0), 0.0);
    }

    #[test]
    fn natural_vignette_matches_fourth_power() {
        approx(natural_vignette(0.5), 0.0625); // 0.5^4
        approx(natural_vignette(0.8), 0.4096); // 0.8^4
    }

    #[test]
    fn natural_vignette_is_monotonic() {
        let mut prev = -1.0;
        for i in 0..=10 {
            let c = i as f32 / 10.0;
            let v = natural_vignette(c);
            assert!(v >= prev, "cos^4 should be non-decreasing in cos");
            prev = v;
        }
    }

    #[test]
    fn natural_falloff_center_is_brightest() {
        approx(natural_falloff([0.5, 0.5], 1.0), 1.0);
    }

    #[test]
    fn natural_falloff_corner_darker_than_center() {
        let center = natural_falloff([0.5, 0.5], 1.0);
        let corner = natural_falloff([0.0, 0.0], 1.0);
        assert!(corner < center, "corner should be darker than centre");
    }

    #[test]
    fn natural_falloff_is_radially_symmetric() {
        let a = natural_falloff([0.2, 0.5], 1.0);
        let b = natural_falloff([0.8, 0.5], 1.0);
        let c = natural_falloff([0.5, 0.2], 1.0);
        approx(a, b);
        approx(a, c);
    }

    #[test]
    fn natural_falloff_longer_lens_flattens() {
        let short = natural_falloff([0.0, 0.0], 0.5);
        let long = natural_falloff([0.0, 0.0], 4.0);
        assert!(long > short, "a longer lens should darken the corner less");
    }

    // --- smoothstep ---

    #[test]
    fn smoothstep_endpoints() {
        approx(vignette_smoothstep(0.0, 1.0, 0.0), 0.0);
        approx(vignette_smoothstep(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn smoothstep_midpoint_is_half() {
        approx(vignette_smoothstep(0.0, 1.0, 0.5), 0.5);
    }

    #[test]
    fn smoothstep_clamps_outside_edges() {
        approx(vignette_smoothstep(0.0, 1.0, -1.0), 0.0);
        approx(vignette_smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn smoothstep_degenerate_edges_is_finite() {
        let v = vignette_smoothstep(0.5, 0.5, 0.5);
        assert!(v.is_finite(), "degenerate edges must not produce NaN/inf");
    }

    // --- artistic ---

    #[test]
    fn artistic_zero_intensity_is_identity_center() {
        approx(artistic_falloff([0.5, 0.5], [0.5, 0.5], 0.0, 0.5, 1.0, 1.0), 1.0);
    }

    #[test]
    fn artistic_zero_intensity_is_identity_corner() {
        approx(artistic_falloff([0.0, 0.0], [0.5, 0.5], 0.0, 0.5, 1.0, 1.0), 1.0);
    }

    #[test]
    fn artistic_corner_darker_than_center() {
        let center = artistic_falloff([0.5, 0.5], [0.5, 0.5], 1.0, 0.5, 1.0, 1.0);
        let corner = artistic_falloff([0.0, 0.0], [0.5, 0.5], 1.0, 0.5, 1.0, 1.0);
        assert!(corner < center, "corner should darken under the artistic model");
    }

    #[test]
    fn artistic_is_monotonic_outward() {
        let mut prev = 2.0;
        for i in 0..=5 {
            let uv = 0.5 - i as f32 * 0.1; // walk from centre toward the left edge
            let f = artistic_falloff([uv, 0.5], [0.5, 0.5], 1.0, 0.5, 1.0, 1.0);
            assert!(f <= prev + 1.0e-6, "factor should not increase outward");
            prev = f;
        }
    }

    #[test]
    fn artistic_roundness_changes_corner() {
        // Square (0) vs circular (1) metric differ at the corner.
        let square = artistic_falloff([0.0, 0.0], [0.5, 0.5], 1.0, 0.5, 0.0, 1.0);
        let round = artistic_falloff([0.0, 0.0], [0.5, 0.5], 1.0, 0.5, 1.0, 1.0);
        assert!((square - round).abs() > 1.0e-4, "roundness should change the corner");
    }

    #[test]
    fn artistic_center_shifts_darkening() {
        // Moving the centre makes the point under it the bright spot.
        let f = artistic_falloff([0.2, 0.2], [0.2, 0.2], 1.0, 0.5, 1.0, 1.0);
        approx(f, 1.0);
    }

    #[test]
    fn artistic_aspect_stretches_x() {
        // A wider aspect pushes more x-distance, darkening a horizontal offset.
        let square_aspect = artistic_falloff([0.1, 0.5], [0.5, 0.5], 1.0, 0.5, 1.0, 1.0);
        let wide_aspect = artistic_falloff([0.1, 0.5], [0.5, 0.5], 1.0, 0.5, 1.0, 2.0);
        assert!(wide_aspect < square_aspect, "wider aspect should darken horizontal offset more");
    }

    #[test]
    fn artistic_factor_in_unit_range() {
        for i in 0..=10 {
            let uv = i as f32 / 10.0;
            let f = artistic_falloff([uv, uv], [0.5, 0.5], 1.5, 0.5, 1.0, 1.0);
            assert!((0.0..=1.0).contains(&f), "factor {f} out of [0, 1]");
        }
    }

    // --- apply ---

    #[test]
    fn apply_vignette_identity() {
        let rgb = [0.2, 0.5, 0.9];
        approx3(apply_vignette(rgb, 1.0), rgb);
    }

    #[test]
    fn apply_vignette_black() {
        approx3(apply_vignette([0.2, 0.5, 0.9], 0.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn apply_vignette_scales() {
        approx3(apply_vignette([0.2, 0.4, 0.6], 0.5), [0.1, 0.2, 0.3]);
    }

    // --- params / dispatch ---

    #[test]
    fn params_default_is_disabled() {
        let p = VignetteParams::default();
        assert_eq!(p.mode, VignetteMode::Artistic);
        approx(p.intensity, 0.0);
        // Disabled => identity factor everywhere.
        approx(vignette_factor([0.0, 0.0], &p), 1.0);
        approx(vignette_factor([0.5, 0.5], &p), 1.0);
    }

    #[test]
    fn params_default_apply_is_identity() {
        let p = VignetteParams::default();
        let rgb = [0.15, 0.4, 0.85];
        approx3(apply_vignette_params(rgb, [0.1, 0.9], &p), rgb);
    }

    #[test]
    fn vignette_factor_natural_matches_falloff() {
        let p = VignetteParams {
            mode: VignetteMode::Natural,
            focal_ratio: 1.5,
            ..VignetteParams::default()
        };
        approx(vignette_factor([0.2, 0.3], &p), natural_falloff([0.2, 0.3], 1.5));
    }

    #[test]
    fn vignette_factor_artistic_matches_falloff() {
        let p = VignetteParams {
            mode: VignetteMode::Artistic,
            intensity: 0.8,
            smoothness: 0.3,
            feather: 0.1,
            roundness: 0.5,
            aspect_ratio: 1.2,
            ..VignetteParams::default()
        };
        let expected = artistic_falloff([0.2, 0.3], p.center, 0.8, 0.3 + 0.1, 0.5, 1.2);
        approx(vignette_factor([0.2, 0.3], &p), expected);
    }

    #[test]
    fn mode_discriminants() {
        assert_eq!(VignetteMode::Natural.as_u32(), 0);
        assert_eq!(VignetteMode::Artistic.as_u32(), 1);
    }
}
