//! DDGI probe depth / visibility field — octahedral two-moment depth with a
//! sharpened cosine kernel, hysteresis blending, gutter border, and
//! Chebyshev (variance) leak suppression plus a self-shadow bias.
//!
//! This is the CPU golden reference for the *visibility* half of Majercik et
//! al. 2019.  Each probe stores an octahedral map of `[mean, mean_sq]` occluder
//! depth moments (`E[d]`, `E[d^2]`).  During interpolation the Chebyshev /
//! Cantelli inequality turns those moments into a soft visibility weight that
//! vanishes smoothly as a shading point slides behind the mean occluder, so a
//! probe on the lit side of a thin wall stops leaking onto the dark side.
//!
//! * [`DdgiDepthOct`] owns the padded octahedral depth atlas with the same
//!   one-texel gutter border scheme as the sibling [`super::ddgi_probe`]
//!   irradiance field.
//! * [`DdgiDepthOct::update`] re-integrates every interior texel from a ray
//!   batch using a *sharpened* cosine kernel (`max(0, dot)^sharpness`, the
//!   RTXGI depth exponent that concentrates each texel on near-axis occluders),
//!   clamps distances to `max_distance`, and blends with temporal hysteresis.
//! * [`DdgiDepthOct::sample`] bilinearly reconstructs the two moments and
//!   [`DdgiDepthOct::visibility`] evaluates the Chebyshev weight with an
//!   optional self-shadow bias.
//!
//! The scalar leak test [`chebyshev_weight`] is reused from
//! [`crate::gi::world_space::visibility`] so this module and the Lumen-style
//! world-space path share one definition of the anti-leak weight.
//!
//! # Conventions
//! * Depths are world-space distances from the probe centre to the nearest
//!   occluder, in the same units as the shading-point distance queried later.
//! * Moments are stored as `[mean, mean_sq]` `f32` pairs to match the GPU
//!   `RG16F`/`RG32F` octahedral twin; variance is clamped non-negative before
//!   use so round-off can never inject a negative or `NaN` weight.
//! * The atlas is row-major over the padded side length `interior + 2`;
//!   interior texels occupy indices `1..=interior` and the gutter is kept in
//!   sync by [`copy_border`](DdgiDepthOct::copy_border).
//! * Every function is deterministic: no RNG, no I/O, no GPU, no `unsafe`.  The
//!   only allocation is the moment buffer owned by [`DdgiDepthOct`].

use alloc::vec::Vec;
use bevy_math::{ops, Vec2, Vec3};

use crate::gi::world_space::octahedral::{dir_to_oct, oct_to_dir};
use crate::gi::world_space::visibility::chebyshev_weight;

/// Smallest weight-sum denominator accepted before a texel keeps its prior
/// moments instead of dividing by an almost-zero cosine sum.
const WEIGHT_EPSILON: f32 = 1.0e-9;

/// Default depth cosine exponent (RTXGI's probe-distance sharpness).
pub const DEFAULT_DEPTH_SHARPNESS: f32 = 50.0;

/// Integrates a ray batch into one texel's `[mean, mean_sq]` depth moments.
///
/// Each ray contributes its hit `distance` weighted by the *sharpened* cosine
/// `max(0, dot(texel_dir, ray_dir))^sharpness`.  The large exponent focuses the
/// texel on occluders near its own axis, matching the DDGI depth kernel.
/// Distances are clamped to `[0, max_distance]`; `sharpness` is clamped to be
/// `>= 1`.  Returns `None` when no ray lands in the texel's hemisphere (so the
/// caller can preserve history), otherwise `Some([mean, mean_sq])`.
#[inline]
pub fn sharpened_depth_moments(
    texel_dir: Vec3,
    rays: &[(Vec3, f32)],
    sharpness: f32,
    max_distance: f32,
) -> Option<[f32; 2]> {
    let axis_len_sq = texel_dir.length_squared();
    if axis_len_sq <= f32::MIN_POSITIVE {
        return None;
    }
    let axis = texel_dir * axis_len_sq.sqrt().recip();
    let sharpness = sharpness.max(1.0);
    let max_distance = max_distance.max(0.0);
    let mut sum_d = 0.0f32;
    let mut sum_d2 = 0.0f32;
    let mut weight_sum = 0.0f32;
    for &(dir, distance) in rays {
        let len_sq = dir.length_squared();
        if len_sq <= f32::MIN_POSITIVE || !distance.is_finite() {
            continue;
        }
        let d = dir * len_sq.sqrt().recip();
        let c = axis.dot(d).max(0.0);
        if c <= 0.0 {
            continue;
        }
        let w = ops::powf(c, sharpness);
        if w <= 0.0 {
            continue;
        }
        let dist = distance.clamp(0.0, max_distance);
        sum_d += dist * w;
        sum_d2 += dist * dist * w;
        weight_sum += w;
    }
    if weight_sum <= WEIGHT_EPSILON {
        return None;
    }
    let inv = weight_sum.recip();
    Some([sum_d * inv, sum_d2 * inv])
}

/// A single probe's octahedral depth-moment field with a gutter border.
#[derive(Clone, Debug, PartialEq)]
pub struct DdgiDepthOct {
    /// Interior side length (texels per octahedral axis), always `>= 1`.
    interior: usize,
    /// Padded side length `interior + 2`.
    padded: usize,
    /// Row-major `padded * padded` `[mean, mean_sq]` texels.
    texels: Vec<[f32; 2]>,
}

impl DdgiDepthOct {
    /// Creates a zero-initialised field with the given interior resolution.
    ///
    /// `interior` is clamped to at least `1`.  Zeroed moments read as fully
    /// occluded for any positive distance, the conservative default until a
    /// texel is populated by [`update`](Self::update).
    #[inline]
    pub fn new(interior: usize) -> Self {
        Self::filled(interior, 0.0, 0.0)
    }

    /// Creates a field whose every texel is initialised to the same moments.
    ///
    /// `interior` is clamped to at least `1`; `mean`/`mean_sq` are clamped
    /// non-negative.
    #[inline]
    pub fn filled(interior: usize, mean: f32, mean_sq: f32) -> Self {
        let interior = interior.max(1);
        let padded = interior + 2;
        let m = [mean.max(0.0), mean_sq.max(0.0)];
        Self {
            interior,
            padded,
            texels: alloc::vec![m; padded * padded],
        }
    }

    /// Interior side length.
    #[inline]
    pub fn interior(&self) -> usize {
        self.interior
    }

    /// Padded side length `interior + 2`.
    #[inline]
    pub fn padded(&self) -> usize {
        self.padded
    }

    /// Total number of texels (`padded * padded`).
    #[inline]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Always `false`: the atlas holds at least `3 * 3` texels.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.padded + x
    }

    /// Reads moments at padded coordinates, returning `[0, 0]` out of range.
    #[inline]
    pub fn moments_at(&self, x: usize, y: usize) -> [f32; 2] {
        if x >= self.padded || y >= self.padded {
            return [0.0, 0.0];
        }
        self.texels[self.index(x, y)]
    }

    /// Decodes the hemisphere axis an interior texel represents.
    #[inline]
    pub fn interior_texel_dir(&self, ix: usize, iy: usize) -> Vec3 {
        let n = self.interior as f32;
        let u = (ix.min(self.interior - 1) as f32 + 0.5) / n;
        let v = (iy.min(self.interior - 1) as f32 + 0.5) / n;
        oct_to_dir(Vec2::new(u, v))
    }

    /// Re-integrates every interior texel from `rays` and blends with temporal
    /// hysteresis, then refreshes the gutter border.
    ///
    /// Each texel's fresh moments come from [`sharpened_depth_moments`]; a texel
    /// whose hemisphere received no ray keeps its prior value.  Blending is
    /// `stored <- lerp(fresh, stored, hysteresis)` with `hysteresis` clamped to
    /// `[0, 1]`.  `sharpness` is clamped `>= 1` and `max_distance` `>= 0`.
    #[inline]
    pub fn update(
        &mut self,
        rays: &[(Vec3, f32)],
        hysteresis: f32,
        sharpness: f32,
        max_distance: f32,
    ) {
        let hysteresis = hysteresis.clamp(0.0, 1.0);
        for iy in 0..self.interior {
            for ix in 0..self.interior {
                let dir = self.interior_texel_dir(ix, iy);
                if let Some(fresh) = sharpened_depth_moments(dir, rays, sharpness, max_distance) {
                    let idx = self.index(ix + 1, iy + 1);
                    let prev = self.texels[idx];
                    let mean = fresh[0] + (prev[0] - fresh[0]) * hysteresis;
                    let mean_sq = fresh[1] + (prev[1] - fresh[1]) * hysteresis;
                    self.texels[idx] = [mean.max(0.0), mean_sq.max(0.0)];
                }
            }
        }
        self.copy_border();
    }

    /// Replicates interior texels into the gutter border using the standard
    /// octahedral mirror/diagonal rule (see [`super::ddgi_probe`]).  Idempotent.
    pub fn copy_border(&mut self) {
        let n = self.interior;
        let last = self.padded - 1;
        for i in 1..=n {
            let m = n + 1 - i;
            let (dst, src) = (self.index(i, 0), self.index(m, 1));
            self.texels[dst] = self.texels[src];
            let (dst, src) = (self.index(i, last), self.index(m, n));
            self.texels[dst] = self.texels[src];
            let (dst, src) = (self.index(0, i), self.index(1, m));
            self.texels[dst] = self.texels[src];
            let (dst, src) = (self.index(last, i), self.index(n, m));
            self.texels[dst] = self.texels[src];
        }
        let (dst, src) = (self.index(0, 0), self.index(n, n));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(last, 0), self.index(1, n));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(0, last), self.index(n, 1));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(last, last), self.index(1, 1));
        self.texels[dst] = self.texels[src];
    }

    /// Bilinearly reconstructs `[mean, mean_sq]` for an arbitrary direction.
    ///
    /// Uses the same padded-atlas addressing as [`super::ddgi_probe`]: interior
    /// texel centre `i` maps to padded coordinate `i`, and the four surrounding
    /// texels (possibly gutter) are blended.  A uniformly filled map samples
    /// back to its fill value.
    #[inline]
    pub fn sample(&self, dir: Vec3) -> [f32; 2] {
        let oct = dir_to_oct(dir);
        let n = self.interior as f32;
        let fx = oct.x * n + 0.5;
        let fy = oct.y * n + 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = (fx - x0f).clamp(0.0, 1.0);
        let ty = (fy - y0f).clamp(0.0, 1.0);
        let last = self.padded - 1;
        let x0 = clamp_padded(x0f, last);
        let y0 = clamp_padded(y0f, last);
        let x1 = clamp_padded(x0f + 1.0, last);
        let y1 = clamp_padded(y0f + 1.0, last);

        let m00 = self.texels[self.index(x0, y0)];
        let m10 = self.texels[self.index(x1, y0)];
        let m01 = self.texels[self.index(x0, y1)];
        let m11 = self.texels[self.index(x1, y1)];
        let mut out = [0.0f32; 2];
        for c in 0..2 {
            let top = m00[c] + (m10[c] - m00[c]) * tx;
            let bottom = m01[c] + (m11[c] - m01[c]) * tx;
            out[c] = (top + (bottom - top) * ty).max(0.0);
        }
        out
    }

    /// Evaluates the Chebyshev visibility weight toward a shading point.
    ///
    /// `dir` is the (not necessarily normalised) probe→point direction and
    /// `distance` their separation.  `bias` is a non-negative self-shadow bias
    /// subtracted from the queried distance before the Chebyshev test, pulling
    /// the shading point slightly toward the probe so a surface does not shadow
    /// itself through depth-map quantisation.  The result lies in `[0, 1]`.
    #[inline]
    pub fn visibility(&self, dir: Vec3, distance: f32, bias: f32) -> f32 {
        let [mean, mean_sq] = self.sample(dir);
        let biased = (distance - bias.max(0.0)).max(0.0);
        chebyshev_weight(mean, mean_sq, biased)
    }
}

/// Clamps a floored, UV-scaled coordinate into `0..=last` as a padded index.
#[inline]
fn clamp_padded(value: f32, last: usize) -> usize {
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
    fn sharpened_kernel_focuses_on_axis() {
        // One ray on-axis at distance 4, one off-axis at distance 100.  With a
        // high sharpness the on-axis distance dominates the mean.
        let rays = [(Vec3::Z, 4.0), (Vec3::new(0.6, 0.0, 0.8), 100.0)];
        let m = sharpened_depth_moments(Vec3::Z, &rays, 50.0, 1000.0).unwrap();
        assert!((m[0] - 4.0).abs() < 0.5, "mean {} should be near 4", m[0]);
    }

    #[test]
    fn no_ray_in_hemisphere_returns_none() {
        let rays = [(Vec3::NEG_Z, 5.0)];
        assert!(sharpened_depth_moments(Vec3::Z, &rays, 50.0, 100.0).is_none());
        assert!(sharpened_depth_moments(Vec3::ZERO, &rays, 50.0, 100.0).is_none());
        assert!(sharpened_depth_moments(Vec3::Z, &[], 50.0, 100.0).is_none());
    }

    #[test]
    fn chebyshev_suppresses_leak_behind_wall() {
        // Probe sees a wall at mean depth 5 (variance ~ 1); a point just in
        // front is visible, a point well behind is suppressed.
        let map = DdgiDepthOct::filled(8, 5.0, 26.0);
        let dir = Vec3::Z;
        let lit = map.visibility(dir, 4.0, 0.0);
        let near = map.visibility(dir, 6.0, 0.0);
        let behind = map.visibility(dir, 9.0, 0.0);
        assert_eq!(lit, 1.0, "point in front of wall must be fully visible");
        assert!(behind < near, "{behind} !< {near}");
        assert!(behind < 0.25, "deep-behind leak not suppressed: {behind}");
    }

    #[test]
    fn self_shadow_bias_increases_visibility() {
        let map = DdgiDepthOct::filled(8, 5.0, 26.0);
        let dir = Vec3::Z;
        let without = map.visibility(dir, 5.4, 0.0);
        let with = map.visibility(dir, 5.4, 0.6); // biased distance 4.8 < mean -> fully visible
        assert!(with >= without, "{with} !>= {without}");
        assert_eq!(with, 1.0);
    }

    #[test]
    fn update_blends_and_border_mirrors() {
        let rays: Vec<(Vec3, f32)> = {
            let n = 8;
            let mut v = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    let u = (i as f32 + 0.5) / n as f32;
                    let w = (j as f32 + 0.5) / n as f32;
                    v.push((oct_to_dir(Vec2::new(u, w)), 7.0));
                }
            }
            v
        };
        let mut map = DdgiDepthOct::new(6);
        map.update(&rays, 0.0, 50.0, 100.0);
        // Uniform distance 7 -> mean ~ 7, mean_sq ~ 49, so a point at distance 6
        // is visible.
        let [mean, mean_sq] = map.sample(Vec3::Z);
        assert!((mean - 7.0).abs() < 0.5, "mean {mean}");
        assert!((mean_sq - 49.0).abs() < 5.0, "mean_sq {mean_sq}");

        // Border copy idempotent.
        let before = map.clone();
        map.copy_border();
        assert_eq!(map, before);
    }

    #[test]
    fn uniform_fill_samples_back_to_fill_value() {
        let map = DdgiDepthOct::filled(8, 3.0, 10.0);
        for dir in [Vec3::X, Vec3::NEG_Y, Vec3::new(0.4, 0.5, -0.7)] {
            let [mean, mean_sq] = map.sample(dir);
            assert!((mean - 3.0).abs() < 1e-5, "mean {mean}");
            assert!((mean_sq - 10.0).abs() < 1e-5, "mean_sq {mean_sq}");
        }
    }

    #[test]
    fn construction_clamps_and_is_safe() {
        let tiny = DdgiDepthOct::new(0);
        assert_eq!(tiny.interior(), 1);
        assert_eq!(tiny.padded(), 3);
        assert_eq!(tiny.len(), 9);
        assert!(!tiny.is_empty());
        assert_eq!(tiny.moments_at(99, 99), [0.0, 0.0]);

        // Negative fill clamps to zero.
        let f = DdgiDepthOct::filled(2, -1.0, -5.0);
        assert_eq!(f.sample(Vec3::Z), [0.0, 0.0]);
    }

    #[test]
    fn visibility_is_finite_everywhere() {
        let map = DdgiDepthOct::filled(4, 2.0, 5.0);
        let n = 16;
        for i in 0..=n {
            for j in 0..=n {
                let u = (i as f32 / n as f32) * 2.0 - 1.0;
                let v = (j as f32 / n as f32) * 2.0 - 1.0;
                let dir = Vec3::new(u, v, 0.3);
                let w = map.visibility(dir, 3.0, 0.1);
                assert!(w.is_finite() && (0.0..=1.0).contains(&w), "w {w}");
            }
        }
    }

    #[test]
    fn results_are_deterministic() {
        let rays = [(Vec3::new(0.1, 0.2, 0.9), 4.0), (Vec3::new(-0.7, 0.3, 0.6), 2.0)];
        let build = || {
            let mut m = DdgiDepthOct::new(6);
            m.update(&rays, 0.3, 50.0, 100.0);
            m
        };
        assert_eq!(build(), build());
    }
}
