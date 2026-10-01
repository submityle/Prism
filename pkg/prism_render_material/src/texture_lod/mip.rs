//! Final mip-level selection: combine a [`super::TriangleLodConstant`], a
//! propagated cone footprint, and the incidence angle into a continuous LOD,
//! and provide the isotropic footprint helper shared with ray differentials.
//!
//! The ray-cone LOD follows RTGems ch. 20:
//! `lambda = Delta + log2(|cone_width|) - log2(|n . d|)`.
//! The `-log2(|n . d|)` term stretches the footprint at grazing incidence,
//! matching the elongation a surface sees from an oblique ray.
//!
//! # Conventions
//! * `normal` and `direction` are expected normalized; the cosine is taken as
//!   `|dot(normal, direction)|` and clamped away from zero by
//!   [`MIN_COS_INCIDENCE`] so grazing rays stay finite.
//! * The returned LOD is clamped to `[0, max_mip]`, where `max_mip` is the
//!   coarsest valid level of the sampled pyramid (`log2(max(w, h))`).
//!
//! # References
//! Ray Tracing Gems 2019, ch. 20 (ray-cone LOD equation); Ewins et al. 1998
//! (log2 footprint-to-LOD mapping).

use bevy_math::ops;

use super::triangle::TriangleLodConstant;

/// Lower bound on `|n . d|`; a cosine below this is treated as this value so a
/// grazing hit yields a large-but-finite LOD boost instead of `+inf`.
pub const MIN_COS_INCIDENCE: f32 = 1.0e-3;

/// Map an isotropic footprint radius (in texels) to a continuous mip level,
/// clamped to `[0, max_mip]`. A footprint of one texel is mip 0.
#[inline]
#[must_use]
pub fn mip_from_isotropic_footprint(footprint_texels: f32, max_mip: f32) -> f32 {
    let f = footprint_texels.max(f32::MIN_POSITIVE);
    ops::log2(f).clamp(0.0, max_mip.max(0.0))
}

/// Compute the ray-cone mip level for a hit.
///
/// * `tri` — the triangle's texel-density constant.
/// * `cone_width` — the propagated cone diameter at the hit (world units).
/// * `n_dot_d` — dot of the (normalized) surface normal and ray direction.
/// * `max_mip` — coarsest valid mip of the sampled pyramid.
#[must_use]
pub fn cone_mip_level(
    tri: TriangleLodConstant,
    cone_width: f32,
    n_dot_d: f32,
    max_mip: f32,
) -> f32 {
    let width = cone_width.abs().max(f32::MIN_POSITIVE);
    let cos_i = n_dot_d.abs().max(MIN_COS_INCIDENCE);
    let lambda = tri.delta() + ops::log2(width) - ops::log2(cos_i);
    lambda.clamp(0.0, max_mip.max(0.0))
}

/// Anisotropic mip result: the base (minor-axis) LOD plus the anisotropy ratio
/// along which the sampler should take extra taps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisotropicMip {
    /// Continuous mip level selected from the minor footprint axis.
    pub lod: f32,
    /// Ratio of major to minor footprint axis, clamped to `[1, max_anisotropy]`.
    pub anisotropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_triangle() -> TriangleLodConstant {
        // 1:1 texel:world mapping -> Delta == 0.
        TriangleLodConstant::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            1,
            1,
        )
    }

    #[test]
    fn one_texel_footprint_is_mip_zero() {
        assert_eq!(mip_from_isotropic_footprint(1.0, 12.0), 0.0);
    }

    #[test]
    fn footprint_doubling_adds_one_mip() {
        let a = mip_from_isotropic_footprint(4.0, 12.0);
        let b = mip_from_isotropic_footprint(8.0, 12.0);
        assert!((b - a - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn cone_mip_tracks_width_in_log2() {
        let tri = unit_triangle();
        // Head-on incidence, Delta=0: lambda == log2(width).
        let m2 = cone_mip_level(tri, 2.0, 1.0, 20.0);
        let m4 = cone_mip_level(tri, 4.0, 1.0, 20.0);
        assert!((m2 - 1.0).abs() < 1.0e-5, "m2={m2}");
        assert!((m4 - 2.0).abs() < 1.0e-5, "m4={m4}");
    }

    #[test]
    fn grazing_incidence_raises_mip() {
        let tri = unit_triangle();
        let head_on = cone_mip_level(tri, 1.0, 1.0, 20.0);
        let grazing = cone_mip_level(tri, 1.0, 0.1, 20.0);
        assert!(grazing > head_on, "grazing={grazing} head_on={head_on}");
        // -log2(0.1) ~= 3.3219 above the head-on value.
        assert!((grazing - head_on - 3.321928).abs() < 1.0e-3);
    }

    #[test]
    fn zero_cosine_is_finite_and_clamped() {
        let tri = unit_triangle();
        let m = cone_mip_level(tri, 1.0, 0.0, 8.0);
        assert!(m.is_finite());
        assert!(m <= 8.0 + 1.0e-6);
    }

    #[test]
    fn result_never_below_zero() {
        let tri = unit_triangle();
        // Tiny width would give a hugely negative log2; must clamp to 0.
        let m = cone_mip_level(tri, 1.0e-9, 1.0, 10.0);
        assert_eq!(m, 0.0);
    }
}
