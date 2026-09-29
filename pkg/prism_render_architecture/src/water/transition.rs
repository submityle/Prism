//! Solver transition bands: distance-weighted blending across solver regions.
//!
//! A single scene can run several water solvers at once — particle-based
//! `FLIP`/`PBF` up close, shallow-water (`SWE`) at mid range, and spectral
//! ocean waves in the distance — each best in its own band. Where two regions
//! meet, their surfaces must be cross-faded so no seam appears in the composited
//! height or normal. This module owns that blend as pure, deterministic
//! functions whose weights always form a partition of unity (they sum to one),
//! which is what keeps the fused surface continuous.
//!
//! Two crossfade bands are laid out along the view distance: particle to
//! shallow-water around one midpoint, and shallow-water to spectral around a
//! farther one, each with a half-width. Inside a band the two neighbours trade
//! weight linearly; outside every band exactly one solver owns the sample. Only
//! `sqrt` is used (via the shared vector type), there are no `f32` equality
//! tests, and there is no AI/ML.

use super::{Vec3, EPS};

/// The three solver weights for one blended sample, summing to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverBlendWeights {
    /// Weight of the near particle solver (`FLIP`/`PBF`).
    pub particle: f32,
    /// Weight of the mid-range shallow-water (`SWE`) solver.
    pub shallow_water: f32,
    /// Weight of the far spectral ocean solver.
    pub spectral: f32,
}

impl SolverBlendWeights {
    /// Sum of the three weights, which the blend guarantees is one.
    #[must_use]
    pub fn sum(self) -> f32 {
        self.particle + self.shallow_water + self.spectral
    }
}

/// Distances laying out the two crossfade bands along the view direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionBands {
    /// Midpoint distance of the particle-to-shallow-water crossfade.
    pub particle_to_swe: f32,
    /// Midpoint distance of the shallow-water-to-spectral crossfade.
    pub swe_to_spectral: f32,
    /// Half-width of each crossfade band; outside it a single solver owns the
    /// sample.
    pub half_width: f32,
}

/// Linear ramp from `0` to `1` as `x` crosses `[lo, hi]`, clamped at the ends.
fn smooth_ramp(x: f32, lo: f32, hi: f32) -> f32 {
    if hi - lo <= EPS {
        return if x < lo { 0.0 } else { 1.0 };
    }
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Computes the three solver weights at a given view `distance`.
///
/// The near band cross-fades particle into shallow-water; the far band
/// cross-fades shallow-water into spectral. Below the near band the sample is
/// pure particle, between the bands pure shallow-water, and beyond the far band
/// pure spectral. The three weights are each in `0..=1` and always sum to one,
/// so the fused height and normal stay continuous across the seams.
#[must_use]
pub fn solver_blend_weights(distance: f32, bands: TransitionBands) -> SolverBlendWeights {
    let d = distance.max(0.0);
    let hw = bands.half_width.max(0.0);
    // Fraction moved from particle into shallow-water across the near band.
    let to_swe = smooth_ramp(d, bands.particle_to_swe - hw, bands.particle_to_swe + hw);
    // Fraction moved from shallow-water into spectral across the far band.
    let to_spectral = smooth_ramp(d, bands.swe_to_spectral - hw, bands.swe_to_spectral + hw);
    let particle = 1.0 - to_swe;
    let shallow_water = to_swe * (1.0 - to_spectral);
    let spectral = to_swe * to_spectral;
    SolverBlendWeights {
        particle,
        shallow_water,
        spectral,
    }
}

/// Blends three scalar solver outputs (such as surface height) by the weights.
///
/// Returns `w.particle*particle + w.shallow_water*shallow_water +
/// w.spectral*spectral`. Because the weights sum to one this is an affine
/// combination: if all three inputs agree the output equals them exactly, so a
/// crossfade never overshoots.
#[must_use]
pub fn blend_scalar(
    weights: SolverBlendWeights,
    particle: f32,
    shallow_water: f32,
    spectral: f32,
) -> f32 {
    weights.particle * particle
        + weights.shallow_water * shallow_water
        + weights.spectral * spectral
}

/// Blends three solver normals by the weights and renormalizes.
///
/// Combines the normals as a weighted sum and returns the unit result; a
/// degenerate zero-length sum falls back to the zero vector via
/// [`Vec3::normalize_or_zero`]. Blending before normalizing is what removes the
/// crease that swapping normals abruptly would leave at a seam.
#[must_use]
pub fn blend_normal(
    weights: SolverBlendWeights,
    particle: Vec3,
    shallow_water: Vec3,
    spectral: Vec3,
) -> Vec3 {
    particle
        .scale(weights.particle)
        .add(shallow_water.scale(weights.shallow_water))
        .add(spectral.scale(weights.spectral))
        .normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BANDS: TransitionBands = TransitionBands {
        particle_to_swe: 20.0,
        swe_to_spectral: 80.0,
        half_width: 5.0,
    };

    #[test]
    fn weights_sum_to_one_everywhere() {
        let mut d = 0.0;
        while d <= 150.0 {
            let w = solver_blend_weights(d, BANDS);
            assert!((w.sum() - 1.0).abs() < EPS, "partition of unity at {d}");
            assert!(w.particle >= 0.0 && w.shallow_water >= 0.0 && w.spectral >= 0.0);
            d += 0.5;
        }
    }

    #[test]
    fn regions_are_pure_outside_the_bands() {
        // Well inside: pure particle.
        let near = solver_blend_weights(2.0, BANDS);
        assert!((near.particle - 1.0).abs() < EPS);
        // Between the two bands: pure shallow-water.
        let mid = solver_blend_weights(50.0, BANDS);
        assert!((mid.shallow_water - 1.0).abs() < EPS);
        // Well beyond: pure spectral.
        let far = solver_blend_weights(140.0, BANDS);
        assert!((far.spectral - 1.0).abs() < EPS);
    }

    #[test]
    fn near_band_hands_particle_to_shallow_water_monotonically() {
        let mut prev = solver_blend_weights(15.0, BANDS).shallow_water;
        let mut d = 15.0;
        while d <= 25.0 {
            let w = solver_blend_weights(d, BANDS);
            assert!(
                w.shallow_water + EPS >= prev,
                "swe weight rises across band"
            );
            assert!(w.spectral.abs() < EPS, "spectral not active in near band");
            prev = w.shallow_water;
            d += 0.5;
        }
    }

    #[test]
    fn scalar_blend_is_affine() {
        let w = solver_blend_weights(22.0, BANDS);
        // Equal inputs reproduce the value exactly (weights sum to one).
        assert!((blend_scalar(w, 3.0, 3.0, 3.0) - 3.0).abs() < EPS);
        // A midpoint blend lies between the two active inputs.
        let mixed = blend_scalar(w, 0.0, 10.0, 0.0);
        assert!(mixed > 0.0 && mixed < 10.0);
    }

    #[test]
    fn normal_blend_returns_unit_vector() {
        let w = solver_blend_weights(22.0, BANDS);
        let n = blend_normal(
            w,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!((n.length() - 1.0).abs() < 1e-4);
        // Agreeing normals pass through as the same unit direction.
        let up = Vec3::new(0.0, 1.0, 0.0);
        let same = blend_normal(w, up, up, up);
        assert!((same.length() - 1.0).abs() < 1e-4);
    }
}
