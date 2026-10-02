//! Neighborhood statistics and history clipping for temporal reconstruction.
//!
//! The core anti-ghosting rule of a temporal upsampler is: *the reprojected
//! history is only trustworthy if it resembles the current frame's local
//! neighborhood.* Each output pixel gathers a small window of current-frame
//! samples (classically the `3 x 3` block around it), summarizes it as a
//! bounding region of plausible colors, and clips the history into that region
//! before blending. Stale history from behind a moving edge is pulled back
//! toward the current color instead of smearing across the screen.
//!
//! This module owns that math in a space-agnostic way: callers feed it samples
//! already in their chosen clipping space (typically tone-mapped `YCoCg`), and
//! it returns the min/max/mean/standard-deviation summary plus two clipping
//! primitives — a hard axis-aligned bounding box (`AABB`) and Marco Salvi's
//! variance box (`mean +/- gamma * stddev`). Clipping uses the line-vs-box
//! "clip toward center" form (Karis) rather than a naive clamp, which shifts
//! the chroma of an out-of-gamut history point less. Everything is `+`, `-`,
//! `*`, `/`, `min`/`max`, and `sqrt`, so a `GPU` kernel reproduces it exactly.

/// Summary statistics of a current-frame sample neighborhood.
///
/// All fields are per-channel in the caller's clipping space. The struct is
/// built once per output pixel from the gathered window and then queried for
/// the clipping bounds; it carries the second moment so a variance box can be
/// derived without a second pass over the samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborhoodStats {
    /// Component-wise minimum over the sampled window.
    min: [f32; 3],
    /// Component-wise maximum over the sampled window.
    max: [f32; 3],
    /// Component-wise arithmetic mean over the sampled window.
    mean: [f32; 3],
    /// Component-wise mean of squares (second raw moment), used to recover the
    /// variance as `m2 - mean^2`.
    m2: [f32; 3],
}

impl NeighborhoodStats {
    /// Builds statistics from a non-empty window of samples.
    ///
    /// Accumulates the component-wise min, max, running sum, and sum of squares
    /// in a single pass, then normalizes the sums by the sample count. An empty
    /// slice has no neighborhood to summarize, so it collapses to a degenerate
    /// zero region at the origin; callers should treat that as "no clipping"
    /// upstream.
    #[must_use]
    pub fn from_samples(samples: &[[f32; 3]]) -> Self {
        if samples.is_empty() {
            return Self {
                min: [0.0; 3],
                max: [0.0; 3],
                mean: [0.0; 3],
                m2: [0.0; 3],
            };
        }
        let mut min = samples[0];
        let mut max = samples[0];
        let mut sum = [0.0f32; 3];
        let mut sum_sq = [0.0f32; 3];
        for s in samples {
            for c in 0..3 {
                min[c] = min[c].min(s[c]);
                max[c] = max[c].max(s[c]);
                sum[c] += s[c];
                sum_sq[c] += s[c] * s[c];
            }
        }
        let inv_n = 1.0 / samples.len() as f32;
        let mut mean = [0.0f32; 3];
        let mut m2 = [0.0f32; 3];
        for c in 0..3 {
            mean[c] = sum[c] * inv_n;
            m2[c] = sum_sq[c] * inv_n;
        }
        Self { min, max, mean, m2 }
    }

    /// Component-wise minimum of the window.
    #[must_use]
    pub const fn min(self) -> [f32; 3] {
        self.min
    }

    /// Component-wise maximum of the window.
    #[must_use]
    pub const fn max(self) -> [f32; 3] {
        self.max
    }

    /// Component-wise mean of the window.
    #[must_use]
    pub const fn mean(self) -> [f32; 3] {
        self.mean
    }

    /// Component-wise standard deviation of the window.
    ///
    /// Derived from the stored moments as `sqrt(max(m2 - mean^2, 0))`. The inner
    /// `max(.., 0)` guards the tiny negative values that floating-point
    /// cancellation can produce for a near-constant neighborhood before the
    /// square root.
    #[must_use]
    pub fn stddev(self) -> [f32; 3] {
        core::array::from_fn(|c| {
            let variance = (self.m2[c] - self.mean[c] * self.mean[c]).max(0.0);
            variance.sqrt()
        })
    }

    /// The hard bounding box of the window: `(min, max)`.
    ///
    /// This is the tightest axis-aligned region that still contains every
    /// sampled color. It never rejects a real neighbor but can be loosened by a
    /// single outlier, which is why the variance box is usually preferred.
    #[must_use]
    pub const fn hard_aabb(self) -> ([f32; 3], [f32; 3]) {
        (self.min, self.max)
    }

    /// Marco Salvi's variance clipping box: `mean +/- gamma * stddev`.
    ///
    /// Instead of the hard min/max, the plausible region is centered on the
    /// mean and sized by the neighborhood's standard deviation scaled by
    /// `gamma` (typically around `1.0`). This rejects a lone bright outlier
    /// (which barely moves the mean) and adapts to how noisy the neighborhood
    /// actually is. The box is intersected with the hard `AABB` so it can only
    /// ever tighten, never invent colors no sample exhibited. A negative or
    /// non-finite `gamma` is treated as `0` (clip straight to the mean).
    #[must_use]
    pub fn variance_aabb(self, gamma: f32) -> ([f32; 3], [f32; 3]) {
        let gamma = if gamma > 0.0 { gamma } else { 0.0 };
        let stddev = self.stddev();
        let mut lo = [0.0f32; 3];
        let mut hi = [0.0f32; 3];
        for c in 0..3 {
            let spread = gamma * stddev[c];
            // Center on the mean, then intersect with the hard box so the
            // region never grows beyond colors the window actually contained.
            lo[c] = (self.mean[c] - spread).max(self.min[c]);
            hi[c] = (self.mean[c] + spread).min(self.max[c]);
        }
        (lo, hi)
    }
}

/// Clips `point` into the axis-aligned box `[aabb_min, aabb_max]` along the ray
/// toward the box center (Karis "clip to `AABB`").
///
/// A naive per-channel clamp snaps an out-of-box color to the nearest face,
/// which can swing its chroma hard. Clipping instead walks the point back along
/// the segment to the box center until it first touches the box, which changes
/// the color's direction far less. If the point already lies inside the box it
/// is returned unchanged.
///
/// The implementation scales the offset-from-center by the largest per-axis
/// overshoot: `t = max_c(|v_c| / extent_c)`; when `t > 1` the point is outside
/// and `center + v / t` lands exactly on the box surface. A zero-extent axis
/// contributes no overshoot (its offset is forced to the center), so a
/// degenerate box clips every point to its center.
#[must_use]
pub fn clip_to_aabb(aabb_min: [f32; 3], aabb_max: [f32; 3], point: [f32; 3]) -> [f32; 3] {
    let mut center = [0.0f32; 3];
    let mut extent = [0.0f32; 3];
    let mut v = [0.0f32; 3];
    for c in 0..3 {
        center[c] = 0.5 * (aabb_min[c] + aabb_max[c]);
        extent[c] = 0.5 * (aabb_max[c] - aabb_min[c]);
        v[c] = point[c] - center[c];
    }
    let mut t = 0.0f32;
    for c in 0..3 {
        if extent[c] > 0.0 {
            t = t.max(v[c].abs() / extent[c]);
        } else if v[c].abs() > 0.0 {
            // Zero-width axis with a non-zero offset: the point can never lie in
            // the box, so force a clip straight to the center on this axis.
            t = f32::INFINITY;
        }
    }
    if t <= 1.0 {
        return point;
    }
    let inv_t = 1.0 / t;
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = center[c] + v[c] * inv_t;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the statistics and clipping checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    /// True when `point` is inside `[lo, hi]` on every axis (within tolerance).
    fn inside(lo: [f32; 3], hi: [f32; 3], point: [f32; 3]) -> bool {
        (0..3).all(|c| point[c] >= lo[c] - 1e-5 && point[c] <= hi[c] + 1e-5)
    }

    #[test]
    fn empty_window_is_degenerate() {
        let s = NeighborhoodStats::from_samples(&[]);
        assert_eq!(s.min(), [0.0; 3]);
        assert_eq!(s.max(), [0.0; 3]);
        assert_eq!(s.mean(), [0.0; 3]);
    }

    #[test]
    fn min_max_mean_are_exact() {
        let samples = [[0.0, 1.0, 2.0], [4.0, 1.0, 0.0], [2.0, 1.0, 4.0]];
        let s = NeighborhoodStats::from_samples(&samples);
        assert_eq!(s.min(), [0.0, 1.0, 0.0]);
        assert_eq!(s.max(), [4.0, 1.0, 4.0]);
        assert!(approx(s.mean()[0], 2.0));
        assert!(approx(s.mean()[1], 1.0));
        assert!(approx(s.mean()[2], 2.0));
    }

    #[test]
    fn stddev_matches_population_formula() {
        // Samples {0, 2, 4} on channel 0: mean 2, variance (4+0+4)/3 = 8/3.
        let samples = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let s = NeighborhoodStats::from_samples(&samples);
        let expected = (8.0f32 / 3.0).sqrt();
        assert!(approx(s.stddev()[0], expected), "{}", s.stddev()[0]);
        // A constant channel has zero spread.
        assert!(approx(s.stddev()[1], 0.0));
    }

    #[test]
    fn variance_box_rejects_a_lone_outlier() {
        // Eight samples near 1.0 plus one bright outlier at 10.0.
        let mut samples = vec![[1.0, 0.0, 0.0]; 8];
        samples.push([10.0, 0.0, 0.0]);
        let s = NeighborhoodStats::from_samples(&samples);
        let (_lo, hi) = s.variance_aabb(1.0);
        // The variance box upper bound sits far below the raw max of 10.
        assert!(
            hi[0] < 10.0,
            "variance box did not reject outlier: {}",
            hi[0]
        );
        // ...but never exceeds the hard max.
        assert!(hi[0] <= s.max()[0] + 1e-5);
    }

    #[test]
    fn variance_box_is_contained_in_hard_box() {
        let samples = [[0.2, 0.5, 0.9], [0.3, 0.4, 0.8], [0.25, 0.45, 0.85]];
        let s = NeighborhoodStats::from_samples(&samples);
        let (lo, hi) = s.variance_aabb(1.0);
        let (hlo, hhi) = s.hard_aabb();
        for c in 0..3 {
            assert!(lo[c] >= hlo[c] - 1e-5 && hi[c] <= hhi[c] + 1e-5);
        }
    }

    #[test]
    fn clip_leaves_interior_points_untouched() {
        let p = [0.5, 0.5, 0.5];
        let out = clip_to_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], p);
        assert_eq!(out, p);
    }

    #[test]
    fn clip_pulls_exterior_point_onto_the_box() {
        // Point far outside on one axis: clips back onto the surface, and the
        // result must lie inside the box.
        let lo = [0.0, 0.0, 0.0];
        let hi = [1.0, 1.0, 1.0];
        let out = clip_to_aabb(lo, hi, [5.0, 0.5, 0.5]);
        assert!(inside(lo, hi, out), "clipped point outside box: {out:?}");
        // The dominant axis lands on the max face.
        assert!(approx(out[0], 1.0), "{}", out[0]);
    }

    #[test]
    fn clip_preserves_direction_toward_center() {
        // A diagonal exterior point keeps its direction from the center: the
        // ratio of the two offending offsets is preserved after clipping.
        let lo = [0.0, 0.0, 0.0];
        let hi = [2.0, 2.0, 2.0];
        let out = clip_to_aabb(lo, hi, [4.0, 2.0, 1.0]);
        // center is (1,1,1); offsets were (3,1,0); max overshoot on axis 0.
        // After clip the offset vector is scaled uniformly, so axis1 offset /
        // axis0 offset stays 1/3.
        let o0 = out[0] - 1.0;
        let o1 = out[1] - 1.0;
        assert!(approx(o1 / o0, 1.0 / 3.0), "ratio {}", o1 / o0);
    }

    #[test]
    fn clip_handles_degenerate_box() {
        // Zero-volume box clips every exterior point to the single center.
        let out = clip_to_aabb([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [5.0, -2.0, 3.0]);
        assert!(approx(out[0], 1.0) && approx(out[1], 1.0) && approx(out[2], 1.0));
    }
}
