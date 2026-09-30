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

    /// World-space footprint width measured *along the surface*, accounting for
    /// the incidence angle between the ray and the surface normal.
    ///
    /// [`Self::projected_width`] returns the cone's cross-section perpendicular
    /// to the ray. At grazing incidence that cross-section is smeared across a
    /// much larger surface patch: the elongation factor is `1 / |cos θ|`, where
    /// `θ` is the angle between the ray direction and the surface normal
    /// (`cos θ = dot(-ray_dir, normal)`). This anisotropic stretch is what
    /// production ray-cones texture-`LOD` uses to avoid under-blurring textures
    /// on surfaces seen edge-on.
    ///
    /// `cos_incidence` is clamped to `[MIN_COS_INCIDENCE, 1.0]` (via its
    /// magnitude) so a perfectly grazing hit (`cos θ -> 0`) selects a bounded,
    /// very coarse mip instead of dividing by zero.
    #[must_use]
    pub fn projected_width_on_surface(self, cos_incidence: f32) -> f32 {
        self.projected_width() / clamp_cos_incidence(cos_incidence)
    }

    /// Surface-projected analogue of [`Self::texel_span`].
    ///
    /// Uses [`Self::projected_width_on_surface`] so grazing hits report the
    /// larger texel count they actually cover on the surface.
    #[must_use]
    pub fn texel_span_on_surface(self, texel_world_size: f32, cos_incidence: f32) -> f32 {
        let texel = sanitize_nonneg(texel_world_size);
        if texel <= 0.0 {
            return 0.0;
        }
        self.projected_width_on_surface(cos_incidence) / texel
    }

    /// Surface-projected analogue of [`Self::mip_level`].
    #[must_use]
    pub fn mip_level_on_surface(
        self,
        texel_world_size: f32,
        cos_incidence: f32,
        max_mip: u32,
    ) -> f32 {
        let span = self.texel_span_on_surface(texel_world_size, cos_incidence);
        let raw = log2_linear(span);
        let ceiling = max_mip as f32;
        raw.clamp(0.0, ceiling)
    }

    /// Surface-projected analogue of [`Self::mip_floor`].
    #[must_use]
    pub fn mip_floor_on_surface(
        self,
        texel_world_size: f32,
        cos_incidence: f32,
        max_mip: u32,
    ) -> u32 {
        let level = self
            .mip_level_on_surface(texel_world_size, cos_incidence, max_mip)
            .floor();
        level as u32
    }

    /// Returns a copy of this footprint with a new hit distance.
    ///
    /// The distance is sanitized like the constructor input. This is the
    /// ergonomic way to advance a propagated cone (see [`Self::propagate`]) to
    /// the next surface once its travel distance is known.
    #[must_use]
    pub fn with_hit_distance(self, hit_distance: f32) -> Self {
        Self {
            cone_width: self.cone_width,
            cone_spread_angle: self.cone_spread_angle,
            hit_distance: sanitize_nonneg(hit_distance),
        }
    }

    /// Propagates the cone through a surface interaction, yielding the footprint
    /// that enters the *next* ray segment (reflection, refraction, or transmit).
    ///
    /// This is the recurrence at the heart of the Akenine-Möller ray-cones
    /// technique. Single-bounce mip selection (the `projected_width` /
    /// `mip_level` family) only describes the cone at the current hit; a
    /// multi-bounce path (mirror reflections, glossy `GI`) must carry the cone
    /// forward so texture `LOD` stays correct after each interaction. Two things
    /// change at a hit:
    ///
    /// 1. The cone width at the hit, [`Self::projected_width`], becomes the base
    ///    width of the next segment (the interaction point is the new origin, so
    ///    `hit_distance` resets to `0`).
    /// 2. The spread angle grows by `surface_spread`, the curvature-derived
    ///    contribution of the interaction (`0` for a perfectly flat mirror;
    ///    larger for convex/curved or rougher surfaces). Curvature is evaluated
    ///    by the caller — mirroring how `cone_spread_angle` and the incidence
    ///    cosine are supplied — so this layer stays pure arithmetic. The term is
    ///    sanitized and accumulated, so spread is monotonically non-decreasing
    ///    along a path and the footprint never shrinks across bounces.
    #[must_use]
    pub fn propagate(self, surface_spread: f32) -> Self {
        let clamped = self.sanitized();
        Self {
            cone_width: clamped.projected_width(),
            cone_spread_angle: clamped.cone_spread_angle + sanitize_nonneg(surface_spread),
            hit_distance: 0.0,
        }
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

/// Smallest incidence cosine used when projecting a footprint onto a surface.
///
/// A hit at exactly `90` degrees would elongate the footprint infinitely
/// (`1 / cos 90 deg -> inf`). Flooring the cosine at this value caps the
/// anisotropic stretch at `1 / 0.05 = 20x`, which keeps mip selection finite
/// while still choosing a very coarse mip for near-grazing hits.
pub const MIN_COS_INCIDENCE: f32 = 0.05;

/// Sanitizes an incidence cosine and clamps its magnitude to
/// `[MIN_COS_INCIDENCE, 1.0]`.
///
/// The sign is discarded (only the angle between ray and normal matters), and
/// non-finite inputs collapse to the grazing floor so the divisor is always a
/// well-defined positive value in the valid range.
fn clamp_cos_incidence(cos_incidence: f32) -> f32 {
    if !cos_incidence.is_finite() {
        return MIN_COS_INCIDENCE;
    }
    cos_incidence.abs().clamp(MIN_COS_INCIDENCE, 1.0)
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

    #[test]
    fn surface_projection_matches_perpendicular_width_at_normal_incidence() {
        // cos = 1 (head-on): the surface footprint equals the perpendicular one.
        let f = RayFootprint::new(0.1, 0.05, 10.0);
        assert!(close(
            f.projected_width_on_surface(1.0),
            f.projected_width()
        ));
        // Sign of the cosine is irrelevant; only the angle matters.
        assert!(close(
            f.projected_width_on_surface(-1.0),
            f.projected_width()
        ));
    }

    #[test]
    fn grazing_incidence_stretches_footprint_and_coarsens_mip() {
        let f = RayFootprint::new(1.0, 0.0, 0.0);
        // cos = 0.5 (60 deg) doubles the surface footprint.
        assert!(close(f.projected_width_on_surface(0.5), 2.0));
        // A grazing hit must never select a finer mip than a head-on hit.
        let head_on = f.mip_level_on_surface(1.0, 1.0, 8);
        let grazing = f.mip_level_on_surface(1.0, 0.2, 8);
        assert!(grazing >= head_on);
        // The stretched span (1 / 0.2 = 5 texels) lands in mip bucket 2
        // (`log2(5) ~= 2.32`).
        assert_eq!(f.mip_floor_on_surface(1.0, 0.2, 8), 2);
    }

    #[test]
    fn extreme_grazing_is_clamped_by_min_cos_incidence() {
        let f = RayFootprint::new(1.0, 0.0, 0.0);
        // cos -> 0 would blow up; the floor caps the stretch at 1 / 0.05 = 20x.
        assert!(close(f.projected_width_on_surface(0.0), 20.0));
        assert!(close(f.projected_width_on_surface(1.0e-9), 20.0));
        // Non-finite cosines collapse to the same grazing floor.
        assert!(close(f.projected_width_on_surface(f32::NAN), 20.0));
        assert!(close(f.projected_width_on_surface(f32::INFINITY), 20.0));
        // Below the floor the stretch never grows further.
        assert!(close(
            f.projected_width_on_surface(0.01),
            f.projected_width_on_surface(0.05)
        ));
    }

    #[test]
    fn texel_span_on_surface_guards_zero_texel() {
        let f = RayFootprint::new(2.0, 0.0, 0.0);
        assert!(close(f.texel_span_on_surface(0.0, 0.5), 0.0));
        assert!(close(f.texel_span_on_surface(-1.0, 0.5), 0.0));
    }

    #[test]
    fn with_hit_distance_replaces_and_sanitizes() {
        let f = RayFootprint::new(0.2, 0.1, 3.0);
        let moved = f.with_hit_distance(7.0);
        assert!(close(moved.cone_width, 0.2));
        assert!(close(moved.cone_spread_angle, 0.1));
        assert!(close(moved.hit_distance, 7.0));
        // Bad distances collapse to zero, other fields untouched.
        let guarded = f.with_hit_distance(-4.0);
        assert!(close(guarded.hit_distance, 0.0));
        let nan = f.with_hit_distance(f32::NAN);
        assert!(close(nan.hit_distance, 0.0));
    }

    #[test]
    fn propagate_carries_hit_width_into_next_base_width() {
        // Width at the hit is 0.1 + 5*0.02 = 0.2; that becomes the next base.
        let f = RayFootprint::new(0.1, 0.02, 5.0);
        let next = f.propagate(0.0);
        assert!(close(next.cone_width, 0.2));
        // Flat interaction keeps the spread and resets travel to the new origin.
        assert!(close(next.cone_spread_angle, 0.02));
        assert!(close(next.hit_distance, 0.0));
        // The propagated cone starts exactly at the width it ended the last
        // segment with (continuity across the bounce).
        assert!(close(next.projected_width(), f.projected_width()));
    }

    #[test]
    fn propagate_accumulates_surface_spread_and_sanitizes() {
        let f = RayFootprint::new(0.1, 0.02, 5.0);
        // A curved/rough interaction widens the cone's spread.
        let next = f.propagate(0.03);
        assert!(close(next.cone_spread_angle, 0.05));
        // Negative or non-finite curvature contributions are ignored (spread
        // never shrinks across a bounce).
        assert!(close(f.propagate(-1.0).cone_spread_angle, 0.02));
        assert!(close(f.propagate(f32::NAN).cone_spread_angle, 0.02));
    }

    #[test]
    fn multi_bounce_footprint_never_shrinks() {
        // Walk a three-bounce path with equal segment lengths and a little
        // curvature at each hit; the footprint width must be monotonically
        // non-decreasing along the path.
        let mut cone = RayFootprint::new(0.05, 0.01, 0.0);
        let mut previous = 0.0;
        for _ in 0..3 {
            cone = cone.with_hit_distance(4.0);
            let width = cone.projected_width();
            assert!(width >= previous);
            previous = width;
            cone = cone.propagate(0.005);
        }
    }
}
