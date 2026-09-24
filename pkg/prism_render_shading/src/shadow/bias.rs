//! Shadow depth-bias helpers: normal-offset and slope-scaled depth bias.
//!
//! Two complementary biases fight the two classic shadow artefacts:
//!
//! * **Shadow acne** (self-shadowing stripes on lit surfaces) comes from the
//!   finite shadow-map texel projecting to a slab of world depth.  A
//!   *slope-scaled depth bias* pushes the reference depth away from the light
//!   proportionally to `tan(theta)`, where `theta` is the angle between the
//!   surface normal and the light, so grazing surfaces (which straddle many
//!   texels of depth) get more bias than head-on ones.
//! * **Peter-panning** (shadows detaching from their caster) is what you get if
//!   you fix acne with depth bias alone and crank it too high.  A
//!   *normal offset* instead nudges the sample position along the surface
//!   normal by a fraction of the shadow texel's world size *before* projection,
//!   moving the comparison off the acne-prone surface without lifting the whole
//!   shadow.
//!
//! Both are the CPU golden twins of the bias math in the future `shadow.wesl`.
//! `tan(theta)` is derived from `n·l` via `sqrt(1 - c^2) / c` to avoid the
//! disallowed `f32::tan` and to stay determinstic across backends.

use crate::vecmath::{add, mul_scalar};

/// Offsets `position` along the surface `normal` by
/// `scale * texel_world_size`, scaled up toward grazing angles by `1 / n·l`
/// (clamped), so nearly tangent surfaces - whose texels cover the most world
/// depth - are lifted the most.  Applied **before** the light-space projection.
///
/// `normal` is assumed unit length.  `n_dot_l` is the clamped, non-negative
/// cosine between the normal and the light direction.
pub fn apply_normal_offset(
    position: [f32; 3],
    normal: [f32; 3],
    texel_world_size: f32,
    scale: f32,
    n_dot_l: f32,
) -> [f32; 3] {
    // sin(theta) / cos(theta)-style grazing boost, but bounded: at n·l -> 0 the
    // offset would blow up, so clamp the cosine to a small floor.
    let cos_theta = n_dot_l.clamp(1.0e-2, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    // Grazing surfaces (large sin) get the full offset; head-on surfaces need
    // almost none.  Keep a small constant floor so flat-lit texels still clear
    // their own depth slab.
    let grazing = (sin_theta / cos_theta).clamp(0.0, 8.0);
    let magnitude = texel_world_size * scale * (1.0 + grazing);
    add(position, mul_scalar(normal, magnitude))
}

/// Slope-scaled constant depth bias, in the shadow map's normalized depth
/// units.
///
/// Returns `const_bias + slope_bias * tan(theta)`, clamped to `max_bias`, where
/// `theta` is the angle between the surface normal and the light.  `tan(theta)`
/// is computed as `sqrt(1 - c^2) / c` from the clamped cosine `n_dot_l` so the
/// bias grows without bound only up to the `max_bias` ceiling.
pub fn slope_scaled_depth_bias(
    n_dot_l: f32,
    const_bias: f32,
    slope_bias: f32,
    max_bias: f32,
) -> f32 {
    let cos_theta = n_dot_l.clamp(1.0e-3, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let tan_theta = sin_theta / cos_theta;
    let bias = const_bias + slope_bias * tan_theta;
    bias.clamp(const_bias.min(max_bias), max_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::math::length3;

    /// A head-on surface (`n·l = 1`) still gets a small constant lift but no
    /// grazing boost; the offset is exactly along the normal.
    #[test]
    fn normal_offset_head_on_is_minimal() {
        let p = apply_normal_offset([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, 2.0, 1.0);
        // grazing == 0 -> magnitude = texel * scale * 1 = 0.2, purely along +Y.
        assert!((p[0]).abs() < 1.0e-6);
        assert!((p[2]).abs() < 1.0e-6);
        assert!((p[1] - 0.2).abs() < 1.0e-5);
    }

    /// A grazing surface is lifted much further than a head-on one for the same
    /// texel size, and the offset stays parallel to the normal.
    #[test]
    fn normal_offset_grazing_is_larger() {
        let head = apply_normal_offset([0.0; 3], [0.0, 1.0, 0.0], 0.1, 1.0, 1.0);
        let graze = apply_normal_offset([0.0; 3], [0.0, 1.0, 0.0], 0.1, 1.0, 0.1);
        assert!(length3(graze) > length3(head) * 2.0);
    }

    /// The depth bias increases monotonically as the light grazes the surface,
    /// and never exceeds the ceiling.
    #[test]
    fn depth_bias_grows_with_grazing_and_clamps() {
        let head = slope_scaled_depth_bias(1.0, 0.001, 0.01, 0.05);
        let mid = slope_scaled_depth_bias(0.5, 0.001, 0.01, 0.05);
        let graze = slope_scaled_depth_bias(0.05, 0.001, 0.01, 0.05);
        assert!(head < mid, "{head} !< {mid}");
        assert!(mid < graze, "{mid} !< {graze}");
        assert!(graze <= 0.05 + 1.0e-6);
        // Head-on bias reduces to (approximately) the constant term.
        assert!((head - 0.001).abs() < 1.0e-4);
    }

    /// The ceiling is respected even when the slope term explodes at extreme
    /// grazing angles.
    #[test]
    fn depth_bias_respects_ceiling() {
        let b = slope_scaled_depth_bias(1.0e-3, 0.0, 10.0, 0.02);
        assert!((b - 0.02).abs() < 1.0e-6);
    }
}
