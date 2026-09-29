//! Ray-cone footprint and texture-`LOD` (mip) selection.
//!
//! A primary ray is modeled as a cone whose radius grows linearly with travel
//! distance. When the cone strikes a surface its cross-section defines the
//! screen-space footprint of the shading sample; the ratio of that footprint to
//! the surface's texel size selects a texture mip level. This mirrors the
//! ray-cones texture-`LOD` technique used by production hardware ray tracers.
//!
//! The `GPU` traversal kernels that would evaluate this per hit are pending the
//! GPU backend; this module provides the `CPU`-verifiable arithmetic contract.
//! Only basic arithmetic plus `sqrt` is used: no transcendental functions
//! (`log2`, `tan`, ...) appear here, so the result is deterministic and cheap.

/// Cone description carried alongside a ray.
///
/// `cone_spread_angle` is stored as a slope (surface rise over run, i.e. the
/// tangent of the half-angle) rather than an angle in radians. Keeping it as a
/// slope lets the width grow linearly with distance without evaluating `tan`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RayFootprint {
    /// Cone radius at the ray origin, in world units.
    pub cone_width: f32,
    /// Growth of the cone radius per world unit travelled (a slope).
    pub cone_spread_angle: f32,
    /// Distance from the origin to the surface hit, in world units.
    pub hit_distance: f32,
}

impl RayFootprint {
    /// Builds a footprint from an explicit origin width, spread slope, and hit
    /// distance. Non-finite or negative inputs are clamped to zero so the
    /// downstream mip math stays well defined.
    #[must_use]
    pub fn new(cone_width: f32, cone_spread_angle: f32, hit_distance: f32) -> Self {
        Self {
            cone_width: sanitize_nonneg(cone_width),
            cone_spread_angle: sanitize_nonneg(cone_spread_angle),
            hit_distance: sanitize_nonneg(hit_distance),
        }
    }

    /// World-space cone radius at the surface hit.
    ///
    /// `width = cone_width + hit_distance * cone_spread_angle`.
    #[must_use]
    pub fn projected_width(self) -> f32 {
        let clamped = self.sanitized();
        clamped.cone_width + clamped.hit_distance * clamped.cone_spread_angle
    }

    /// Number of texels the footprint spans given a surface texel's world size.
    ///
    /// A value of `1.0` means the footprint matches one texel (mip 0); larger
    /// values indicate minification and select coarser mips.
    #[must_use]
    pub fn texel_span(self, texel_world_size: f32) -> f32 {
        let texel = sanitize_nonneg(texel_world_size);
        if texel <= 0.0 {
            return 0.0;
        }
        self.projected_width() / texel
    }

    /// Continuous mip level for this footprint, clamped to `[0, max_mip]`.
    ///
    /// The fractional mip is `log2(span)`, approximated without transcendental
    /// functions by [`log2_linear`]. `texel_world_size` is the world extent of
    /// one texel at mip 0.
    #[must_use]
    pub fn mip_level(self, texel_world_size: f32, max_mip: u32) -> f32 {
        let span = self.texel_span(texel_world_size);
        let raw = log2_linear(span);
        let ceiling = max_mip as f32;
        // `raw` is finite for finite inputs, so `clamp` cannot see a NaN here.
        raw.clamp(0.0, ceiling)
    }

    /// Discrete mip level obtained by flooring [`Self::mip_level`].
    #[must_use]
    pub fn mip_floor(self, texel_world_size: f32, max_mip: u32) -> u32 {
        let level = self.mip_level(texel_world_size, max_mip).floor();
        level as u32
    }

    fn sanitized(self) -> Self {
        Self {
            cone_width: sanitize_nonneg(self.cone_width),
            cone_spread_angle: sanitize_nonneg(self.cone_spread_angle),
            hit_distance: sanitize_nonneg(self.hit_distance),
        }
    }
}

/// Replaces non-finite or negative values with zero.
fn sanitize_nonneg(value: f32) -> f32 {
    if !value.is_finite() || value < 0.0 {
        return 0.0;
    }
    value
}

/// Transcendental-free approximation of `log2(ratio)`.
///
/// The integer part is found by repeated halving/doubling into `[1, 2)`, and the
/// fractional part uses the linear approximation `frac ~= mantissa - 1`. The
/// approximation is exact at powers of two and monotonically increasing, which
/// is all the mip selection needs. Inputs `<= 0` or non-finite map to `0.0`.
#[must_use]
pub fn log2_linear(ratio: f32) -> f32 {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 0.0;
    }
    let mut mantissa = ratio;
    let mut exponent = 0.0f32;
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1.0;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1.0;
    }
    exponent + (mantissa - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    #[test]
    fn projected_width_grows_with_distance() {
        let near = RayFootprint::new(0.1, 0.05, 1.0);
        let far = RayFootprint::new(0.1, 0.05, 10.0);
        assert!(near.projected_width() < far.projected_width());
        assert!(close(near.projected_width(), 0.15));
        assert!(close(far.projected_width(), 0.6));
    }

    #[test]
    fn log2_linear_exact_on_powers_of_two() {
        assert!(close(log2_linear(1.0), 0.0));
        assert!(close(log2_linear(2.0), 1.0));
        assert!(close(log2_linear(4.0), 2.0));
        assert!(close(log2_linear(8.0), 3.0));
        assert!(close(log2_linear(0.5), -1.0));
    }

    #[test]
    fn log2_linear_is_monotonic() {
        let mut previous = log2_linear(0.1);
        let mut ratio = 0.2;
        while ratio < 64.0 {
            let current = log2_linear(ratio);
            assert!(current >= previous);
            previous = current;
            ratio += 0.2;
        }
    }

    #[test]
    fn log2_linear_guards_bad_inputs() {
        assert!(close(log2_linear(0.0), 0.0));
        assert!(close(log2_linear(-3.0), 0.0));
        assert!(close(log2_linear(f32::NAN), 0.0));
        assert!(close(log2_linear(f32::INFINITY), 0.0));
    }

    #[test]
    fn mip_level_clamps_to_range() {
        // Span 1 texel -> mip 0.
        let unit = RayFootprint::new(1.0, 0.0, 0.0);
        assert!(close(unit.mip_level(1.0, 8), 0.0));
        // Span 16 texels -> mip 4.
        let coarse = RayFootprint::new(16.0, 0.0, 0.0);
        assert!(close(coarse.mip_level(1.0, 8), 4.0));
        // Beyond the ceiling clamps to max.
        let huge = RayFootprint::new(4096.0, 0.0, 0.0);
        assert!(close(huge.mip_level(1.0, 6), 6.0));
        // Sub-texel footprints clamp to mip 0.
        let tiny = RayFootprint::new(0.25, 0.0, 0.0);
        assert!(close(tiny.mip_level(1.0, 8), 0.0));
    }

    #[test]
    fn mip_floor_matches_expected_bucket() {
        let f = RayFootprint::new(6.0, 0.0, 0.0);
        // log2(6) ~= 2.58 -> floor 2.
        assert_eq!(f.mip_floor(1.0, 8), 2);
    }

    #[test]
    fn texel_span_guards_zero_texel() {
        let f = RayFootprint::new(2.0, 0.0, 0.0);
        assert!(close(f.texel_span(0.0), 0.0));
        assert!(close(f.texel_span(-1.0), 0.0));
    }

    #[test]
    fn new_sanitizes_negative_and_nonfinite() {
        let f = RayFootprint::new(-1.0, f32::NAN, f32::INFINITY);
        assert!(close(f.cone_width, 0.0));
        assert!(close(f.cone_spread_angle, 0.0));
        assert!(close(f.hit_distance, 0.0));
    }
}
