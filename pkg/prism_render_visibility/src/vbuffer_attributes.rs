//! Attribute interpolation and texture `LOD` from barycentric derivatives.
//!
//! Given the perspective-correct barycentric weights and their screen
//! derivatives produced by
//! [`barycentric_derivatives`](crate::barycentric_derivatives), these helpers
//! reconstruct any per-vertex attribute at a visibility-buffer pixel together
//! with its screen-space gradients, and from a `UV` gradient derive the
//! texture mip level a forward rasterizer would have selected.
//!
//! This completes the Deferred Attribute Interpolation Shading (DAIS) path:
//! the geometry pass writes only triangle ids, and this module recovers
//! position/normal/`UV`/color plus `ddx`/`ddy` purely analytically, so textures
//! can be sampled with correct mip selection and anisotropy.

use crate::BarycentricDerivatives;
use bevy_math::ops;

/// An interpolated scalar attribute and its screen-space derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolatedScalar {
    /// Interpolated value at the sample point.
    pub value: f32,
    /// `∂value/∂x`, in units of attribute per pixel.
    pub ddx: f32,
    /// `∂value/∂y`, in units of attribute per pixel.
    pub ddy: f32,
}

/// An interpolated 2-vector attribute (e.g. a `UV`) and its derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolatedVec2 {
    /// Interpolated value at the sample point.
    pub value: [f32; 2],
    /// `∂value/∂x` per component.
    pub ddx: [f32; 2],
    /// `∂value/∂y` per component.
    pub ddy: [f32; 2],
}

/// An interpolated 3-vector attribute (e.g. a normal) and its derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolatedVec3 {
    /// Interpolated value at the sample point.
    pub value: [f32; 3],
    /// `∂value/∂x` per component.
    pub ddx: [f32; 3],
    /// `∂value/∂y` per component.
    pub ddy: [f32; 3],
}

/// Interpolates a scalar vertex attribute `(a0, a1, a2)` at the sample point.
///
/// The value is `sum_i b_i * a_i` and the derivatives follow directly from the
/// barycentric derivatives since the attribute values are per-vertex constants.
#[must_use]
pub fn interpolate_scalar(bary: &BarycentricDerivatives, a: [f32; 3]) -> InterpolatedScalar {
    let value = bary.lambda[0] * a[0] + bary.lambda[1] * a[1] + bary.lambda[2] * a[2];
    let ddx = bary.ddx[0] * a[0] + bary.ddx[1] * a[1] + bary.ddx[2] * a[2];
    let ddy = bary.ddy[0] * a[0] + bary.ddy[1] * a[1] + bary.ddy[2] * a[2];
    InterpolatedScalar { value, ddx, ddy }
}

/// Interpolates a 2-vector vertex attribute (e.g. texture coordinates).
#[must_use]
pub fn interpolate_vec2(bary: &BarycentricDerivatives, a: [[f32; 2]; 3]) -> InterpolatedVec2 {
    let x = interpolate_scalar(bary, [a[0][0], a[1][0], a[2][0]]);
    let y = interpolate_scalar(bary, [a[0][1], a[1][1], a[2][1]]);
    InterpolatedVec2 {
        value: [x.value, y.value],
        ddx: [x.ddx, y.ddx],
        ddy: [x.ddy, y.ddy],
    }
}

/// Interpolates a 3-vector vertex attribute (e.g. object-space position or
/// normal). The result is *not* renormalized; shading code normalizes as
/// needed after interpolation.
#[must_use]
pub fn interpolate_vec3(bary: &BarycentricDerivatives, a: [[f32; 3]; 3]) -> InterpolatedVec3 {
    let x = interpolate_scalar(bary, [a[0][0], a[1][0], a[2][0]]);
    let y = interpolate_scalar(bary, [a[0][1], a[1][1], a[2][1]]);
    let z = interpolate_scalar(bary, [a[0][2], a[1][2], a[2][2]]);
    InterpolatedVec3 {
        value: [x.value, y.value, z.value],
        ddx: [x.ddx, y.ddx, z.ddx],
        ddy: [x.ddy, y.ddy, z.ddy],
    }
}

/// Computes the trilinear texture `LOD` (mip level) for a `UV` gradient.
///
/// `uv_ddx`/`uv_ddy` are the `UV` derivatives in units of texture coordinate
/// per pixel (as produced by [`interpolate_vec2`]). `tex_width`/`tex_height`
/// are the base mip dimensions in texels. This is the standard OpenGL/Direct3D
/// formula: scale the `UV` gradients to texel space, take the longer of the two
/// axis vectors, and `log2` it.
///
/// Returns `0.0` for a zero gradient (a texel-aligned, unmagnified sample).
#[must_use]
pub fn texture_lod(uv_ddx: [f32; 2], uv_ddy: [f32; 2], tex_width: u32, tex_height: u32) -> f32 {
    let w = texels_to_f32(tex_width);
    let h = texels_to_f32(tex_height);
    let dx = [uv_ddx[0] * w, uv_ddx[1] * h];
    let dy = [uv_ddy[0] * w, uv_ddy[1] * h];
    let len_sq_x = dx[0] * dx[0] + dx[1] * dx[1];
    let len_sq_y = dy[0] * dy[0] + dy[1] * dy[1];
    let max_len_sq = len_sq_x.max(len_sq_y);
    if max_len_sq <= 0.0 {
        return 0.0;
    }
    // 0.5 * log2(len_sq) == log2(sqrt(len_sq)) == log2(max pixel footprint).
    0.5 * ops::log2(max_len_sq)
}

/// Anisotropy ratio of a `UV` footprint: the ratio of the longer to the shorter
/// texel-space axis gradient.
///
/// Returns `1.0` when either axis has zero length (isotropic / degenerate).
/// Anisotropic sampling clamps this to the hardware's maximum anisotropy.
#[must_use]
pub fn texture_anisotropy(uv_ddx: [f32; 2], uv_ddy: [f32; 2], tex_width: u32, tex_height: u32) -> f32 {
    let w = texels_to_f32(tex_width);
    let h = texels_to_f32(tex_height);
    let ax = uv_ddx[0] * w;
    let ay = uv_ddx[1] * h;
    let bx = uv_ddy[0] * w;
    let by = uv_ddy[1] * h;
    let len_x = (ax * ax + ay * ay).sqrt();
    let len_y = (bx * bx + by * by).sqrt();
    let (hi, lo) = if len_x >= len_y {
        (len_x, len_y)
    } else {
        (len_y, len_x)
    };
    if lo <= 0.0 {
        return 1.0;
    }
    hi / lo
}

/// Converts a texel dimension to `f32`. Texture dimensions fit comfortably in
/// `f32`'s exact-integer range.
#[must_use]
fn texels_to_f32(texels: u32) -> f32 {
    u16::try_from(texels).map_or_else(
        |_| {
            let hi = u16::try_from(texels >> 16).unwrap_or(u16::MAX);
            let lo = u16::try_from(texels & 0xFFFF).unwrap_or(u16::MAX);
            f32::from(hi) * 65_536.0 + f32::from(lo)
        },
        f32::from,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bary(lambda: [f32; 3], ddx: [f32; 3], ddy: [f32; 3]) -> BarycentricDerivatives {
        BarycentricDerivatives { lambda, ddx, ddy }
    }

    #[test]
    fn scalar_interpolation_matches_manual_weighting() {
        let b = bary([0.2, 0.3, 0.5], [0.1, -0.05, -0.05], [0.0, 0.2, -0.2]);
        let a = [10.0, 20.0, 30.0];
        let s = interpolate_scalar(&b, a);
        assert!((s.value - (0.2 * 10.0 + 0.3 * 20.0 + 0.5 * 30.0)).abs() < 1e-5);
        assert!((s.ddx - (0.1 * 10.0 - 0.05 * 20.0 - 0.05 * 30.0)).abs() < 1e-5);
        assert!((s.ddy - (0.0 * 10.0 + 0.2 * 20.0 - 0.2 * 30.0)).abs() < 1e-5);
    }

    #[test]
    fn vec2_interpolation_is_componentwise() {
        let b = bary([0.25, 0.25, 0.5], [1.0, -0.5, -0.5], [0.0, 1.0, -1.0]);
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let r = interpolate_vec2(&b, uv);
        assert!((r.value[0] - 0.25).abs() < 1e-5);
        assert!((r.value[1] - 0.5).abs() < 1e-5);
        // u gradient = sum ddx_i * u_i = 1*0 + (-0.5)*1 + (-0.5)*0 = -0.5.
        assert!((r.ddx[0] + 0.5).abs() < 1e-5);
        // v gradient along y = 0*0 + 1*0 + (-1)*1 = -1.
        assert!((r.ddy[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn vec3_interpolation_recovers_vertex_at_corner() {
        let b = bary([1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]);
        let n = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let r = interpolate_vec3(&b, n);
        assert_eq!(r.value, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn texture_lod_zero_gradient_is_zero() {
        let lod = texture_lod([0.0, 0.0], [0.0, 0.0], 1024, 1024);
        assert_eq!(lod, 0.0);
    }

    #[test]
    fn texture_lod_one_texel_per_pixel_is_level_zero() {
        // A UV step of exactly one texel per pixel on a 256-wide texture.
        let lod = texture_lod([1.0 / 256.0, 0.0], [0.0, 1.0 / 256.0], 256, 256);
        assert!(lod.abs() < 1e-5, "lod={lod}");
    }

    #[test]
    fn texture_lod_doubling_footprint_adds_one_level() {
        // Two texels per pixel -> footprint sqrt covers 2 texels -> lod = 1.
        let lod = texture_lod([2.0 / 256.0, 0.0], [0.0, 2.0 / 256.0], 256, 256);
        assert!((lod - 1.0).abs() < 1e-5, "lod={lod}");
    }

    #[test]
    fn texture_lod_uses_longer_axis() {
        // x axis covers 4 texels, y axis covers 1 texel; lod driven by x -> 2.
        let lod = texture_lod([4.0 / 256.0, 0.0], [0.0, 1.0 / 256.0], 256, 256);
        assert!((lod - 2.0).abs() < 1e-5, "lod={lod}");
    }

    #[test]
    fn anisotropy_ratio_matches_axis_lengths() {
        let a = texture_anisotropy([4.0 / 256.0, 0.0], [0.0, 1.0 / 256.0], 256, 256);
        assert!((a - 4.0).abs() < 1e-5, "aniso={a}");
    }

    #[test]
    fn anisotropy_isotropic_footprint_is_one() {
        let a = texture_anisotropy([2.0 / 256.0, 0.0], [0.0, 2.0 / 256.0], 256, 256);
        assert!((a - 1.0).abs() < 1e-5, "aniso={a}");
    }
}
