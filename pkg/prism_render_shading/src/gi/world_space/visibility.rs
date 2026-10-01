//! DDGI-style Chebyshev depth-moment visibility — CPU golden.
//!
//! Interpolating irradiance from a sparse grid of probes leaks light through
//! thin walls: a shading point on the dark side of a wall still gathers energy
//! from a probe sitting on the lit side.  DDGI/RTXGI suppress this with a
//! *two-moment depth map* per probe.  Instead of a single occluder distance,
//! each probe texel stores the running mean depth `r = E[d]` and mean-squared
//! depth `r2 = E[d^2]` of the nearest geometry along that direction.  At
//! shading time the Chebyshev (one-tailed Cantelli) inequality turns those two
//! moments into a soft visibility weight that smoothly falls off as a point
//! moves behind the mean occluder, giving leak-free yet filtered probe blends.
//!
//! This module is the backend-neutral reference for that test:
//!
//! * [`DepthMomentOct`] packs the per-probe moments into an octahedral map of
//!   configurable resolution, reusing the sibling [`super::octahedral`] mapping
//!   so a direction round-trips to the same texel the GPU twin will address.
//! * [`DepthMomentOct::update`] folds a freshly sampled occluder distance into
//!   a texel with an exponential moving average (the RTXGI hysteresis blend).
//! * [`DepthMomentOct::sample`] reconstructs the two moments for an arbitrary
//!   direction with bilinear filtering across the four surrounding texels.
//! * [`chebyshev_weight`] and [`DepthMomentOct::visibility`] evaluate the
//!   anti-leak weight itself.
//!
//! # Conventions
//! * Depths are world-space distances from the probe centre to the nearest
//!   occluder, in the same units as the shading-point distance queried later.
//! * Moments are stored as `f32` pairs `[mean, mean_sq]` to match the GPU
//!   texture twin (an `RG16F`/`RG32F` octahedral atlas).
//! * The weight is always clamped to `[0, 1]`; variance is clamped to be
//!   non-negative before use so `f32` round-off in `r2 - r*r` can never inject
//!   a negative or `NaN` weight.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and the only allocation is the moment buffer owned by [`DepthMomentOct`].

use alloc::vec::Vec;
use bevy_math::{Vec2, Vec3};

use super::octahedral::dir_to_oct;

/// Evaluates the DDGI Chebyshev visibility weight from two depth moments.
///
/// Given the mean occluder depth `mean` (`E[d]`), the mean-squared occluder
/// depth `mean_sq` (`E[d^2]`), and the distance `distance` from the probe to
/// the shading point, this returns how likely the point is to be *visible* to
/// the probe (`1.0` fully visible, `0.0` fully occluded):
///
/// * If `distance <= mean` the point is in front of (or at) the average
///   occluder and is treated as fully visible, weight `1.0`.  This is the
///   standard DDGI short-circuit and also makes a zero-variance surface behave
///   like a hard depth test.
/// * Otherwise the Cantelli one-tailed inequality bounds the visible fraction
///   by `variance / (variance + (distance - mean)^2)`, where
///   `variance = max(mean_sq - mean^2, 0)`.
///
/// The variance clamp guarantees a finite, non-negative numerator; the final
/// value is clamped to `[0, 1]` for defensive robustness even though the
/// formula already lies in that range for non-negative inputs.
#[inline]
pub fn chebyshev_weight(mean: f32, mean_sq: f32, distance: f32) -> f32 {
    if distance <= mean {
        return 1.0;
    }
    // Clamp away negative round-off: Var = E[d^2] - E[d]^2 >= 0 analytically.
    let variance = (mean_sq - mean * mean).max(0.0);
    let delta = distance - mean;
    let weight = variance / (variance + delta * delta);
    weight.clamp(0.0, 1.0)
}

/// A per-probe octahedral map of two-moment occluder depths.
///
/// The map is a square `resolution x resolution` grid of texels laid out in
/// row-major order; each texel stores `[mean, mean_sq]` as described in the
/// module docs.  Directions are addressed through the shared octahedral
/// parameterisation so a direction, its stored texel, and the GPU texture twin
/// all agree.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMomentOct {
    /// Side length of the square octahedral grid, always `>= 1`.
    resolution: usize,
    /// Row-major `resolution * resolution` texels of `[mean, mean_sq]` depth
    /// moments.
    texels: Vec<[f32; 2]>,
}

impl DepthMomentOct {
    /// Creates a zero-initialised map of the given side resolution.
    ///
    /// `resolution` is clamped to at least `1` so the grid is never empty.  A
    /// zeroed texel reads `mean = mean_sq = 0`, which makes any point at a
    /// positive distance read as fully occluded until the texel is populated by
    /// [`update`](Self::update) — the conservative choice that avoids leaking
    /// through not-yet-measured directions.
    #[inline]
    pub fn new(resolution: usize) -> Self {
        Self::filled(resolution, 0.0, 0.0)
    }

    /// Creates a map whose every texel is initialised to the same moments.
    ///
    /// Useful for seeding a probe with a large uniform depth (so it starts
    /// fully visible) or for tests that need a known uniform field.
    /// `resolution` is clamped to at least `1`.
    #[inline]
    pub fn filled(resolution: usize, mean: f32, mean_sq: f32) -> Self {
        let resolution = resolution.max(1);
        let mut texels = Vec::new();
        texels.resize(resolution * resolution, [mean, mean_sq]);
        Self { resolution, texels }
    }

    /// Returns the side length of the octahedral grid.
    #[inline]
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Returns the number of texels (`resolution * resolution`).
    #[inline]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Always `false`: the grid holds at least one texel by construction.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// Returns the raw `[mean, mean_sq]` moments stored at a texel index.
    ///
    /// Out-of-range indices read as `[0.0, 0.0]` so callers never panic on a
    /// stale index.
    #[inline]
    pub fn moments_at(&self, index: usize) -> [f32; 2] {
        self.texels.get(index).copied().unwrap_or([0.0, 0.0])
    }

    /// Maps a direction to the nearest texel's row-major index.
    ///
    /// The direction need not be normalised; the octahedral mapping handles the
    /// projection.  The returned index is always in `0..self.len()`.
    #[inline]
    pub fn index_for_dir(&self, dir: Vec3) -> usize {
        let (ix, iy) = self.nearest_texel(dir_to_oct(dir));
        self.texel_index(ix, iy)
    }

    /// Row-major index for integer texel coordinates (already clamped).
    #[inline]
    fn texel_index(&self, ix: usize, iy: usize) -> usize {
        iy * self.resolution + ix
    }

    /// Clamps a UV in `[0, 1]^2` to the nearest integer texel coordinate.
    #[inline]
    fn nearest_texel(&self, uv: Vec2) -> (usize, usize) {
        let last = self.resolution - 1;
        let fx = (uv.x * self.resolution as f32).floor();
        let fy = (uv.y * self.resolution as f32).floor();
        let ix = clamp_index(fx, last);
        let iy = clamp_index(fy, last);
        (ix, iy)
    }

    /// Folds a sampled occluder distance into the nearest texel using an
    /// exponential moving average (RTXGI hysteresis blend).
    ///
    /// With blend factor `alpha` the texel updates as
    /// `mean <- lerp(mean, distance, alpha)` and
    /// `mean_sq <- lerp(mean_sq, distance^2, alpha)`.  `alpha` is clamped to
    /// `[0, 1]`: `0` keeps the texel unchanged, `1` overwrites it with the new
    /// sample.  Negative distances are clamped to `0`.  The update is
    /// deterministic and allocation-free.
    #[inline]
    pub fn update(&mut self, dir: Vec3, distance: f32, alpha: f32) {
        let alpha = alpha.clamp(0.0, 1.0);
        let distance = distance.max(0.0);
        let (ix, iy) = self.nearest_texel(dir_to_oct(dir));
        let index = self.texel_index(ix, iy);
        let texel = &mut self.texels[index];
        texel[0] += alpha * (distance - texel[0]);
        texel[1] += alpha * (distance * distance - texel[1]);
    }

    /// Reconstructs the `[mean, mean_sq]` moments for an arbitrary direction
    /// with bilinear filtering across the four surrounding texels.
    ///
    /// Texel centres sit at `(i + 0.5) / resolution` in UV space.  Fractional
    /// coordinates are bilinearly blended; coordinates outside the grid are
    /// clamped to the edge texel (clamp-to-edge).  This edge clamp is a
    /// deliberate simplification of the true octahedral seam wrapping — it is
    /// exact in the interior and only approximate within half a texel of the
    /// border, which is adequate for the CPU reference and documented here for
    /// the GPU twin to match.  A uniformly filled map therefore samples back to
    /// its fill value for every direction.
    #[inline]
    pub fn sample(&self, dir: Vec3) -> [f32; 2] {
        let uv = dir_to_oct(dir);
        let last = self.resolution - 1;
        let res = self.resolution as f32;
        // Shift to texel-centre space so (i + 0.5)/res maps to integer i.
        let fx = uv.x * res - 0.5;
        let fy = uv.y * res - 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = (fx - x0f).clamp(0.0, 1.0);
        let ty = (fy - y0f).clamp(0.0, 1.0);
        let x0 = clamp_index(x0f, last);
        let y0 = clamp_index(y0f, last);
        let x1 = clamp_index(x0f + 1.0, last);
        let y1 = clamp_index(y0f + 1.0, last);

        let m00 = self.texels[self.texel_index(x0, y0)];
        let m10 = self.texels[self.texel_index(x1, y0)];
        let m01 = self.texels[self.texel_index(x0, y1)];
        let m11 = self.texels[self.texel_index(x1, y1)];

        let mut out = [0.0f32; 2];
        for c in 0..2 {
            let top = m00[c] + (m10[c] - m00[c]) * tx;
            let bottom = m01[c] + (m11[c] - m01[c]) * tx;
            out[c] = top + (bottom - top) * ty;
        }
        out
    }

    /// Evaluates the Chebyshev visibility weight toward a shading point.
    ///
    /// `dir` is the (not necessarily normalised) direction from the probe to
    /// the point and `distance` is their separation.  The two moments are
    /// bilinearly sampled via [`sample`](Self::sample) and fed to
    /// [`chebyshev_weight`]; the result lies in `[0, 1]`.
    #[inline]
    pub fn visibility(&self, dir: Vec3, distance: f32) -> f32 {
        let [mean, mean_sq] = self.sample(dir);
        chebyshev_weight(mean, mean_sq, distance)
    }
}

/// Clamps a floored UV-scaled coordinate into `0..=last` as a texel index.
///
/// Handles the `fx < 0` and `fx > last` cases without `usize` underflow and
/// keeps the result inside the grid.
#[inline]
fn clamp_index(value: f32, last: usize) -> usize {
    if value <= 0.0 {
        0
    } else {
        let v = value as usize;
        if v > last {
            last
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn fully_visible_point_has_unit_weight() {
        // Uniform large occluder depth: anything closer than the mean is lit.
        let map = DepthMomentOct::filled(8, 10.0, 100.0);
        let dir = Vec3::new(0.3, -0.6, 0.7);
        assert_eq!(map.visibility(dir, 5.0), 1.0);
        // Distance exactly at the mean is still treated as visible.
        assert_eq!(map.visibility(dir, 10.0), 1.0);
    }

    #[test]
    fn occluded_point_weight_decreases_with_distance() {
        // mean = 5, variance = mean_sq - mean^2 = 26 - 25 = 1.
        let map = DepthMomentOct::filled(8, 5.0, 26.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let w_near = map.visibility(dir, 5.5); // delta 0.5 -> 1/1.25 = 0.8
        let w_mid = map.visibility(dir, 6.0); //  delta 1.0 -> 1/2.0  = 0.5
        let w_far = map.visibility(dir, 7.0); //  delta 2.0 -> 1/5.0  = 0.2

        for w in [w_near, w_mid, w_far] {
            assert!((0.0..1.0).contains(&w), "weight {w} must be in [0, 1)");
        }
        assert!(w_near > w_mid, "{w_near} !> {w_mid}");
        assert!(w_mid > w_far, "{w_mid} !> {w_far}");
        assert!((w_near - 0.8).abs() < 1e-5, "{w_near}");
        assert!((w_mid - 0.5).abs() < 1e-5, "{w_mid}");
        assert!((w_far - 0.2).abs() < 1e-5, "{w_far}");
    }

    #[test]
    fn variance_never_negative_after_clamp() {
        // mean_sq < mean^2 would give a negative "variance"; it must clamp to 0
        // and, past the mean, produce a hard-zero weight (not NaN / negative).
        let w = chebyshev_weight(5.0, 1.0, 9.0);
        assert_eq!(w, 0.0);
        assert!(w.is_finite());

        // A zero-variance surface is a hard depth test past the mean.
        assert_eq!(chebyshev_weight(5.0, 25.0, 6.0), 0.0);
        assert_eq!(chebyshev_weight(5.0, 25.0, 4.0), 1.0);

        // Sweep a grid of inputs: the weight stays within [0, 1] and finite.
        for mi in 0..10 {
            for sq in 0..10 {
                for di in 0..12 {
                    let mean = mi as f32;
                    let mean_sq = sq as f32;
                    let distance = di as f32 * 0.75;
                    let w = chebyshev_weight(mean, mean_sq, distance);
                    assert!(w.is_finite() && (0.0..=1.0).contains(&w), "w={w}");
                }
            }
        }
    }

    #[test]
    fn update_applies_exponential_moving_average() {
        let dir = Vec3::new(-0.2, 0.5, 0.8);
        let mut map = DepthMomentOct::new(4);

        // alpha = 1 overwrites: mean = 4, mean_sq = 16.
        map.update(dir, 4.0, 1.0);
        let idx = map.index_for_dir(dir);
        let m = map.moments_at(idx);
        assert!((m[0] - 4.0).abs() < 1e-6, "mean {}", m[0]);
        assert!((m[1] - 16.0).abs() < 1e-6, "mean_sq {}", m[1]);

        // alpha = 0.5 blends toward distance 6: mean = 5, mean_sq = 26.
        map.update(dir, 6.0, 0.5);
        let m = map.moments_at(idx);
        assert!((m[0] - 5.0).abs() < 1e-6, "mean {}", m[0]);
        assert!((m[1] - 26.0).abs() < 1e-6, "mean_sq {}", m[1]);

        // alpha = 0 is a no-op.
        map.update(dir, 100.0, 0.0);
        let m = map.moments_at(idx);
        assert!((m[0] - 5.0).abs() < 1e-6);
        assert!((m[1] - 26.0).abs() < 1e-6);
    }

    #[test]
    fn negative_distance_is_clamped_in_update() {
        let dir = Vec3::Z;
        let mut map = DepthMomentOct::new(2);
        map.update(dir, -3.0, 1.0);
        let m = map.moments_at(map.index_for_dir(dir));
        assert_eq!(m, [0.0, 0.0]);
    }

    #[test]
    fn uniform_fill_samples_back_to_fill_value() {
        let map = DepthMomentOct::filled(8, 3.0, 10.0);
        for dir in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(0.4, 0.5, -0.7),
            Vec3::new(-0.9, 0.1, 0.3),
        ] {
            let [mean, mean_sq] = map.sample(dir);
            assert!((mean - 3.0).abs() < 1e-5, "mean {mean} for {dir:?}");
            assert!((mean_sq - 10.0).abs() < 1e-5, "mean_sq {mean_sq} for {dir:?}");
        }
    }

    #[test]
    fn construction_and_indexing_stay_in_bounds() {
        // Zero resolution clamps up to a 1x1 grid.
        let tiny = DepthMomentOct::new(0);
        assert_eq!(tiny.resolution(), 1);
        assert_eq!(tiny.len(), 1);
        assert!(!tiny.is_empty());
        assert_eq!(tiny.index_for_dir(Vec3::new(1.0, 2.0, -3.0)), 0);

        // A larger grid: every direction maps to a valid in-range texel and
        // sampling never panics.
        let res = 16;
        let map = DepthMomentOct::new(res);
        assert_eq!(map.len(), res * res);
        let n = 20;
        for i in 0..=n {
            for j in 0..=n {
                let u = (i as f32 / n as f32) * 2.0 - 1.0;
                let v = (j as f32 / n as f32) * 2.0 - 1.0;
                let dir = Vec3::new(u, v, 0.5);
                let idx = map.index_for_dir(dir);
                assert!(idx < map.len(), "idx {idx} >= {}", map.len());
                let _ = map.sample(dir);
                let _ = map.visibility(dir, 1.0);
            }
        }

        // Stale indices read as zero rather than panicking.
        assert_eq!(map.moments_at(map.len()), [0.0, 0.0]);
        assert_eq!(map.moments_at(usize::MAX), [0.0, 0.0]);
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut m = DepthMomentOct::new(8);
            m.update(Vec3::new(0.1, 0.2, 0.9), 4.0, 0.75);
            m.update(Vec3::new(0.1, 0.2, 0.9), 6.0, 0.5);
            m.update(Vec3::new(-0.7, 0.3, 0.6), 2.0, 0.9);
            m
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);

        let dir = Vec3::new(0.1, 0.2, 0.9);
        assert_eq!(a.sample(dir), b.sample(dir));
        assert_eq!(a.visibility(dir, 5.0), b.visibility(dir, 5.0));
    }

    #[test]
    fn edge_clamp_keeps_border_directions_finite() {
        // Directions near the octahedral border (z ~ 0 seam) must sample and
        // weight without NaNs even though the fold wrapping is approximated.
        let map = DepthMomentOct::filled(4, 2.0, 5.0);
        for dir in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.7, 0.7, 0.0),
            Vec3::new(-0.7, -0.7, 0.0),
        ] {
            let [mean, mean_sq] = map.sample(dir);
            assert!(mean.is_finite() && mean_sq.is_finite());
            let w = map.visibility(dir, 3.0);
            assert!(w.is_finite() && (0.0..=1.0).contains(&w));
        }
    }
}
