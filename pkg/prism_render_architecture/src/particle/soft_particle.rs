//! Soft-particle depth fade: the `CPU`-verifiable contract that keeps
//! translucent particles from cutting a hard seam where they intersect opaque
//! scene geometry (design §14).
//!
//! When a translucent sprite is rasterized against solid geometry, the naive
//! result is a razor edge along the intersection line: the sprite is either
//! fully drawn or fully clipped by the depth test, with nothing in between.
//! "Soft particles" replace that binary cut with a smooth fade driven by the
//! *difference* between the already-rendered scene depth and the particle
//! fragment's own depth. The closer the particle sits to the surface behind it,
//! the more it fades, so smoke settles softly onto the floor instead of slicing
//! through it.
//!
//! Everything here works in linear `view`-space depth (larger means farther
//! from the camera), the space a `GPU` kernel obtains by linearizing the
//! hardware `NDC` depth. This module owns three orthogonal pieces:
//!
//! 1. [`DepthFade`] / [`contact_fade`] — the core seam fade from the
//!    scene-minus-particle depth gap.
//! 2. [`NearFade`] — a camera-proximity fade that dissolves particles that
//!    crowd the near plane (a spark drifting into the lens).
//! 3. [`SoftParticleParams::combined_fade`] — the product of the two, the
//!    single opacity multiplier a shader applies.
//!
//! [`linearize_depth`] reconstructs `view`-space depth from `NDC` depth using
//! only multiply / divide / subtract, so this reference stays bit-reproducible
//! against a future `GPU` kernel with no transcendental functions in sight.
//!
//! Apart from an optional `use crate::particle::gpu_layout` this module imports
//! no sibling particle module: it is a self-contained numeric contract.

/// Absolute tolerance for degenerate-interval guards at run time (a fade band
/// whose endpoints are closer than this collapses to a hard step).
///
/// Bare `==` / `!=` on `f32` is avoided throughout; compare magnitudes against
/// this epsilon instead.
pub const EPS: f32 = 1.0e-6;

/// Clamps a raw fade ratio into the unit interval `[0, 1]`.
///
/// Kept private so the intent — "an opacity multiplier can never leave
/// `[0, 1]`" — reads directly at every call site.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The soft-particle seam fade over a fixed `view`-space fade band.
///
/// `fade_distance` is the depth gap (in linear `view`-space units) over which a
/// particle ramps from fully transparent (touching the surface behind it) to
/// fully opaque (a full band or more in front of it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthFade {
    /// Depth gap over which the fade ramps from `0` to `1`; a non-positive
    /// value disables the fade (see [`DepthFade::fade_factor`]).
    pub fade_distance: f32,
}

impl DepthFade {
    /// Builds a fade with the given `view`-space fade band.
    #[must_use]
    pub fn new(fade_distance: f32) -> Self {
        Self { fade_distance }
    }

    /// Returns the opacity multiplier in `[0, 1]` for a particle fragment at
    /// `particle_depth` drawn against opaque geometry at `scene_depth`.
    ///
    /// Both depths are linear `view`-space depths where larger means farther.
    /// The factor is `clamp((scene_depth - particle_depth) / fade_distance,
    /// 0, 1)`:
    ///
    /// * A particle *behind* the geometry (`particle_depth` greater than
    ///   `scene_depth`) yields a negative gap and clamps to `0` — it fades out
    ///   entirely, matching what the depth test would have culled.
    /// * A particle flush against the surface yields `0`.
    /// * A particle a full `fade_distance` (or more) in front stays fully
    ///   opaque at `1`.
    ///
    /// A `fade_distance` of zero or less is a guard: the fade is disabled and
    /// the fragment keeps full opacity (`1`).
    #[must_use]
    pub fn fade_factor(self, scene_depth: f32, particle_depth: f32) -> f32 {
        contact_fade(scene_depth - particle_depth, self.fade_distance)
    }
}

/// The same seam fade as [`DepthFade::fade_factor`], expressed directly on the
/// pre-computed depth gap `depth_diff = scene_depth - particle_depth`.
///
/// Returns `clamp(depth_diff / fade_distance, 0, 1)`. A non-positive
/// `fade_distance` disables the fade and returns `1`. A non-positive
/// `depth_diff` (the particle is at or behind the surface) returns `0`.
#[must_use]
pub fn contact_fade(depth_diff: f32, fade_distance: f32) -> f32 {
    if fade_distance <= 0.0 {
        return 1.0;
    }
    clamp01(depth_diff / fade_distance)
}

/// A camera-proximity fade that dissolves particles crowding the near plane.
///
/// The fade ramps linearly with `view`-space depth: at `near_start` (closest)
/// the particle is fully faded, at `near_end` (farther) it is fully visible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearFade {
    /// `view`-space depth at which particles are fully faded out (nearest).
    pub near_start: f32,
    /// `view`-space depth at which particles regain full opacity (farthest).
    pub near_end: f32,
}

impl NearFade {
    /// Builds a near-plane proximity fade over `[near_start, near_end]`.
    #[must_use]
    pub fn new(near_start: f32, near_end: f32) -> Self {
        Self {
            near_start,
            near_end,
        }
    }

    /// Returns the opacity multiplier in `[0, 1]` for a fragment at
    /// `view_depth` (linear `view`-space, larger means farther).
    ///
    /// Below `near_start` the fragment is fully faded (`0`); above `near_end`
    /// it is fully visible (`1`); between them the factor rises linearly.
    ///
    /// A degenerate band whose endpoints are within [`EPS`] collapses to a hard
    /// step: `0` before `near_end` and `1` at or beyond it, never a divide by a
    /// vanishing width.
    #[must_use]
    pub fn camera_proximity_fade(self, view_depth: f32) -> f32 {
        let width = self.near_end - self.near_start;
        if width.abs() < EPS {
            return if view_depth < self.near_end { 0.0 } else { 1.0 };
        }
        clamp01((view_depth - self.near_start) / width)
    }
}

/// The full soft-particle opacity contract: the seam fade times the near-plane
/// proximity fade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftParticleParams {
    /// Seam-fade band width, forwarded to [`DepthFade`].
    pub fade_distance: f32,
    /// Near-plane fade start, forwarded to [`NearFade`].
    pub near_start: f32,
    /// Near-plane fade end, forwarded to [`NearFade`].
    pub near_end: f32,
}

impl SoftParticleParams {
    /// Builds the combined parameter set.
    #[must_use]
    pub fn new(fade_distance: f32, near_start: f32, near_end: f32) -> Self {
        Self {
            fade_distance,
            near_start,
            near_end,
        }
    }

    /// The seam-fade component alone.
    #[must_use]
    pub fn depth_fade(self) -> DepthFade {
        DepthFade::new(self.fade_distance)
    }

    /// The near-plane proximity-fade component alone.
    #[must_use]
    pub fn near_fade(self) -> NearFade {
        NearFade::new(self.near_start, self.near_end)
    }

    /// Returns the final opacity multiplier in `[0, 1]`: the product of the
    /// seam fade (from `scene_depth` versus `particle_depth`) and the
    /// near-plane fade (evaluated at the particle's own `view`-space depth).
    ///
    /// Each factor lives in `[0, 1]`, so their product does too.
    #[must_use]
    pub fn combined_fade(self, scene_depth: f32, particle_depth: f32) -> f32 {
        let seam = self.depth_fade().fade_factor(scene_depth, particle_depth);
        let near = self.near_fade().camera_proximity_fade(particle_depth);
        seam * near
    }
}

/// Reconstructs linear `view`-space depth from a hardware `NDC` depth.
///
/// `ndc_depth` is the post-projection depth in `[0, 1]` (the `WebGPU` / `D3D`
/// convention). With a standard perspective projection the `view`-space depth
/// is `(near * far) / (far - ndc_depth * (far - near))`, using only multiply,
/// divide, and subtract — no transcendental functions — so the result matches a
/// `GPU` kernel bit for bit.
///
/// At the endpoints the map is exact: `ndc_depth == 0` returns `near` and
/// `ndc_depth == 1` returns `far`.
///
/// A degenerate frustum whose `near` and `far` are within [`EPS`] is a guard:
/// it returns `near` rather than dividing by a vanishing range.
#[must_use]
pub fn linearize_depth(ndc_depth: f32, near: f32, far: f32) -> f32 {
    let range = far - near;
    if range.abs() < EPS {
        return near;
    }
    (near * far) / (far - ndc_depth * range)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` equality assertions in this module's tests.
    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn seam_fade_is_monotonic_in_the_gap() {
        let fade = DepthFade::new(2.0);
        // Larger scene-minus-particle gap -> larger (or equal) factor, rising
        // to a clamped 1.
        let mut prev = -1.0;
        let mut diff = 0.0;
        while diff <= 4.0 {
            let f = fade.fade_factor(diff, 0.0);
            assert!(f >= prev - CMP_EPS, "not monotonic at diff={diff}");
            assert!((0.0..=1.0).contains(&f));
            prev = f;
            diff += 0.25;
        }
    }

    #[test]
    fn particle_behind_geometry_fades_to_zero() {
        let fade = DepthFade::new(1.5);
        // particle_depth > scene_depth => particle is farther/behind => 0.
        assert!(approx(fade.fade_factor(2.0, 3.0), 0.0));
        assert!(approx(fade.fade_factor(2.0, 5.0), 0.0));
        // Flush against the surface is also 0.
        assert!(approx(fade.fade_factor(2.0, 2.0), 0.0));
    }

    #[test]
    fn non_positive_fade_distance_disables_fade() {
        assert!(approx(DepthFade::new(0.0).fade_factor(5.0, 1.0), 1.0));
        assert!(approx(DepthFade::new(-3.0).fade_factor(5.0, 1.0), 1.0));
        assert!(approx(contact_fade(0.5, 0.0), 1.0));
        assert!(approx(contact_fade(-9.0, -1.0), 1.0));
    }

    #[test]
    fn contact_fade_clamps_both_boundaries() {
        // Below the band -> 0.
        assert!(approx(contact_fade(-1.0, 2.0), 0.0));
        // Exactly one band -> 1.
        assert!(approx(contact_fade(2.0, 2.0), 1.0));
        // Beyond the band -> clamped 1.
        assert!(approx(contact_fade(10.0, 2.0), 1.0));
        // Mid band -> proportional.
        assert!(approx(contact_fade(1.0, 2.0), 0.5));
    }

    #[test]
    fn near_fade_is_monotonic_and_clamped() {
        let fade = NearFade::new(1.0, 3.0);
        // Too close -> 0.
        assert!(approx(fade.camera_proximity_fade(0.5), 0.0));
        assert!(approx(fade.camera_proximity_fade(1.0), 0.0));
        // Midpoint -> 0.5.
        assert!(approx(fade.camera_proximity_fade(2.0), 0.5));
        // Far enough -> 1.
        assert!(approx(fade.camera_proximity_fade(3.0), 1.0));
        assert!(approx(fade.camera_proximity_fade(9.0), 1.0));

        // Monotone non-decreasing across the sweep.
        let mut prev = -1.0;
        let mut d = 0.0;
        while d <= 4.0 {
            let f = fade.camera_proximity_fade(d);
            assert!(f >= prev - CMP_EPS);
            prev = f;
            d += 0.2;
        }
    }

    #[test]
    fn near_fade_degenerate_band_is_a_hard_step() {
        let fade = NearFade::new(2.0, 2.0);
        assert!(approx(fade.camera_proximity_fade(1.999), 0.0));
        assert!(approx(fade.camera_proximity_fade(2.0), 1.0));
        assert!(approx(fade.camera_proximity_fade(5.0), 1.0));
    }

    #[test]
    fn linearize_depth_hits_the_endpoints() {
        let near = 0.5;
        let far = 100.0;
        assert!(approx(linearize_depth(0.0, near, far), near));
        assert!(approx(linearize_depth(1.0, near, far), far));
        // Interior stays within [near, far] and is farther than near.
        let mid = linearize_depth(0.5, near, far);
        assert!(mid > near && mid < far);
    }

    #[test]
    fn linearize_depth_degenerate_range_guard() {
        assert!(approx(linearize_depth(0.7, 4.0, 4.0), 4.0));
        assert!(approx(linearize_depth(0.0, 4.0, 4.0 + 1.0e-9), 4.0));
    }

    #[test]
    fn combined_fade_is_the_product_of_both_factors() {
        let params = SoftParticleParams::new(2.0, 1.0, 3.0);
        let scene = 5.0;
        let particle = 2.0;
        let seam = params.depth_fade().fade_factor(scene, particle);
        let near = params.near_fade().camera_proximity_fade(particle);
        assert!(approx(params.combined_fade(scene, particle), seam * near));
        // With particle behind geometry the whole product collapses to 0.
        assert!(approx(params.combined_fade(1.0, 4.0), 0.0));
        // Deep in front and past the near band -> full opacity.
        assert!(approx(params.combined_fade(100.0, 50.0), 1.0));
    }
}
