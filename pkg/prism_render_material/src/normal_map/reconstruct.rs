//! Reconstructing full tangent-space normals from compressed two-channel maps.
//!
//! AAA pipelines store tangent-space normals in two channels and rebuild the
//! third on the GPU, because the normal is unit length so `z` is redundant.
//! The two staple packings are **BC5 "RG"** (x in red, y in green -- the modern
//! default) and the legacy **DXT5nm "AG"** (x in alpha, y in green, chosen so
//! the two surviving channels sit on BC3's independently-compressed alpha and
//! green blocks). Both decode here with pure analytic math -- no AI/ML -- so a
//! CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # Conventions
//! * Inputs are `unorm` samples in `[0, 1]` (e.g. straight from
//!   [`BcTexelSource`](crate::BcTexelSource)); they are remapped to the signed
//!   `[-1, 1]` tangent axes by `c -> 2c - 1`.
//! * Reconstructed `z = sqrt(max(0, 1 - x^2 - y^2))`, so the result lies in the
//!   upper hemisphere (`z >= 0`), matching the tangent-space convention.
//! * Outputs are normalised; a degenerate near-zero vector falls back to the
//!   geometric normal `(0, 0, 1)` rather than producing NaNs.
//!
//! # References
//! * Mittring, "Finding Next Gen -- CryEngine 2" (two-channel normal storage).
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.7.2.

/// Geometric fallback used when a reconstructed vector is too short to
/// normalise safely.
const GEOMETRIC_NORMAL: [f32; 3] = [0.0, 0.0, 1.0];

#[inline]
fn normalize_or_up(v: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 > 1.0e-12 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        GEOMETRIC_NORMAL
    }
}

/// Map a single `unorm` channel in `[0, 1]` to a signed axis in `[-1, 1]`.
#[inline]
#[must_use]
pub fn unorm_to_snorm(c: f32) -> f32 {
    2.0 * c.clamp(0.0, 1.0) - 1.0
}

/// Reconstruct a unit tangent-space normal from signed `(x, y)` in `[-1, 1]`.
///
/// `z` is recovered as `sqrt(max(0, 1 - x^2 - y^2))`; the inputs are clamped so
/// an over-unit `(x, y)` (possible after filtering) degrades to a grazing
/// normal instead of yielding a NaN.
#[must_use]
pub fn reconstruct_z(xy: [f32; 2]) -> [f32; 3] {
    let x = xy[0].clamp(-1.0, 1.0);
    let y = xy[1].clamp(-1.0, 1.0);
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    normalize_or_up([x, y, z])
}

/// Decode a BC5-style **RG** normal sample: `x` from red, `y` from green.
#[must_use]
pub fn decode_rg(rg: [f32; 2]) -> [f32; 3] {
    reconstruct_z([unorm_to_snorm(rg[0]), unorm_to_snorm(rg[1])])
}

/// Decode a legacy DXT5nm **AG** normal sample: `x` from alpha, `y` from green.
#[must_use]
pub fn decode_ag(rgba: [f32; 4]) -> [f32; 3] {
    reconstruct_z([unorm_to_snorm(rgba[3]), unorm_to_snorm(rgba[1])])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_unit(n: [f32; 3]) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5, "len={len} n={n:?}");
    }

    #[test]
    fn flat_sample_is_geometric_normal() {
        // unorm 0.5 -> snorm 0 on both axes -> (0, 0, 1).
        let n = decode_rg([0.5, 0.5]);
        assert!(n[0].abs() < 1.0e-6 && n[1].abs() < 1.0e-6);
        assert!((n[2] - 1.0).abs() < 1.0e-6);
        is_unit(n);
    }

    #[test]
    fn full_tilt_x_is_unit_and_points_right() {
        // unorm 1.0 -> snorm +1 on x -> (1, 0, 0).
        let n = decode_rg([1.0, 0.5]);
        assert!(n[0] > 0.999);
        assert!(n[2] >= 0.0);
        is_unit(n);
    }

    #[test]
    fn ag_reads_x_from_alpha_y_from_green() {
        // alpha=1 -> x=+1, green=0.5 -> y=0.
        let n = decode_ag([0.0, 0.5, 0.0, 1.0]);
        assert!(n[0] > 0.999 && n[1].abs() < 1.0e-6);
        is_unit(n);
    }

    #[test]
    fn over_unit_xy_degrades_without_nan() {
        let n = reconstruct_z([1.5, 1.5]);
        assert!(n.iter().all(|c| c.is_finite()));
        is_unit(n);
    }

    #[test]
    fn hemisphere_z_is_non_negative() {
        for &c in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(decode_rg([c, 1.0 - c])[2] >= 0.0);
        }
    }
}
