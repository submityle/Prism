//! Parallel-split shadow-map (PSSM / "cascaded shadow maps") split scheme and
//! cascade selection.
//!
//! A single directional light covers the whole view frustum, so the shadow map
//! is split into a handful of *cascades*, each covering a progressively larger
//! slice of view-space depth.  The near cascades get a tight projection (high
//! resolution close to the camera) while the far cascades trade resolution for
//! coverage.
//!
//! The split distances follow the practical logarithmic/uniform blend from
//! Zhang et al. (used by both Unreal's `ComputeShadowCullingVolume` cascade
//! distances and the classic GPU Gems PSSM article):
//!
//! ```text
//! split_i = lambda * near * (far / near)^(i / N)
//!         + (1 - lambda) * (near + (far - near) * (i / N))
//! ```
//!
//! with `lambda` blending between the purely logarithmic (`1.0`) and purely
//! uniform (`0.0`) distributions.  The transcendental `powf` routes through
//! `bevy_math::ops` so the CPU reference stays bit-identical to the GPU twin.

use bevy_math::ops;

/// Maximum number of directional cascades the fixed-size split table can hold.
/// Matches the four-cascade layout used by the GPU shadow atlas uniform.
pub const MAX_CASCADE_COUNT: usize = 4;

/// Ordered far-plane view-space distances for each cascade, plus the shared
/// near plane, describing a `count`-cascade split of `[near, far]`.
///
/// `distances[i]` is the *far* boundary of cascade `i`; the near boundary of
/// cascade `i` is `distances[i - 1]` (or `near` for cascade `0`).  Entries past
/// `count` are set to the final far plane so out-of-range reads clamp cleanly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeSplits {
    /// Shared near plane of cascade `0`.
    pub near: f32,
    /// Far boundary of each cascade, in ascending view-space distance.
    pub distances: [f32; MAX_CASCADE_COUNT],
    /// Number of active cascades in `distances` (`1..=MAX_CASCADE_COUNT`).
    pub count: usize,
}

impl CascadeSplits {
    /// Near boundary of cascade `index` (its predecessor's far plane, or the
    /// shared near plane for cascade `0`).  Out-of-range indices clamp to the
    /// last active cascade.
    pub fn cascade_near(&self, index: usize) -> f32 {
        let index = index.min(self.count.saturating_sub(1));
        if index == 0 {
            self.near
        } else {
            self.distances[index - 1]
        }
    }

    /// Far boundary of cascade `index`.  Out-of-range indices clamp to the last
    /// active cascade.
    pub fn cascade_far(&self, index: usize) -> f32 {
        let index = index.min(self.count.saturating_sub(1));
        self.distances[index]
    }
}

/// Computes the practical logarithmic/uniform PSSM split of `[near, far]` into
/// `count` cascades.
///
/// `count` is clamped into `1..=MAX_CASCADE_COUNT` and `lambda` into `[0, 1]`.
/// `near` is floored to a small positive epsilon because the logarithmic term
/// divides by it.  The returned table's `distances[count - 1]` is always
/// exactly `far`.
pub fn compute_cascade_splits(near: f32, far: f32, count: usize, lambda: f32) -> CascadeSplits {
    let count = count.clamp(1, MAX_CASCADE_COUNT);
    let lambda = lambda.clamp(0.0, 1.0);
    let near = near.max(1.0e-4);
    let far = far.max(near + 1.0e-4);

    let ratio = far / near;
    let range = far - near;
    let inv_count = (count as f32).recip();

    let mut distances = [far; MAX_CASCADE_COUNT];
    for (i, slot) in distances.iter_mut().enumerate().take(count) {
        let fraction = ((i + 1) as f32) * inv_count;
        let logarithmic = near * ops::powf(ratio, fraction);
        let uniform = near + range * fraction;
        *slot = lambda * logarithmic + (1.0 - lambda) * uniform;
    }
    // Guard against `powf` rounding nudging the last split off the far plane so
    // depth clamping and cascade selection agree exactly at the boundary.
    distances[count - 1] = far;

    CascadeSplits {
        near,
        distances,
        count,
    }
}

/// Selects the cascade covering `view_depth` (a positive view-space distance
/// from the camera): the first cascade whose far boundary is `>= view_depth`,
/// falling back to the last cascade for anything beyond the final split.
pub fn select_cascade(view_depth: f32, splits: &CascadeSplits) -> usize {
    for i in 0..splits.count {
        if view_depth <= splits.distances[i] {
            return i;
        }
    }
    splits.count - 1
}

/// Blend weight, in `[0, 1]`, for cross-fading the *selected* cascade into the
/// next one near its far boundary to hide the hard resolution seam.
///
/// `band_fraction` is the width of the transition band expressed as a fraction
/// of the selected cascade's depth range.  `0.0` is returned when there is no
/// next cascade (the selected one is last) or when `view_depth` is outside the
/// transition band, so callers can skip the second, more expensive lookup.
pub fn cascade_blend_weight(
    view_depth: f32,
    splits: &CascadeSplits,
    index: usize,
    band_fraction: f32,
) -> f32 {
    if index + 1 >= splits.count {
        return 0.0;
    }
    let band_fraction = band_fraction.clamp(0.0, 1.0);
    if band_fraction <= 0.0 {
        return 0.0;
    }

    let near = splits.cascade_near(index);
    let far = splits.cascade_far(index);
    let cascade_range = (far - near).max(1.0e-4);
    let band = cascade_range * band_fraction;
    let band_start = far - band;

    if view_depth <= band_start {
        0.0
    } else {
        ((view_depth - band_start) / band).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The uniform distribution (`lambda = 0`) spaces splits evenly and always
    /// terminates exactly on the far plane.
    #[test]
    fn uniform_split_is_evenly_spaced() {
        let s = compute_cascade_splits(1.0, 5.0, 4, 0.0);
        assert_eq!(s.count, 4);
        assert!((s.distances[0] - 2.0).abs() < 1.0e-5);
        assert!((s.distances[1] - 3.0).abs() < 1.0e-5);
        assert!((s.distances[2] - 4.0).abs() < 1.0e-5);
        assert!((s.distances[3] - 5.0).abs() < 1.0e-6);
    }

    /// The logarithmic distribution (`lambda = 1`) packs cascades toward the
    /// near plane: each split is a fixed geometric ratio of the previous.
    #[test]
    fn logarithmic_split_is_geometric() {
        let s = compute_cascade_splits(1.0, 16.0, 4, 1.0);
        // (16/1)^(1/4) = 2, so splits are 2, 4, 8, 16.
        assert!((s.distances[0] - 2.0).abs() < 1.0e-4);
        assert!((s.distances[1] - 4.0).abs() < 1.0e-4);
        assert!((s.distances[2] - 8.0).abs() < 1.0e-4);
        assert!((s.distances[3] - 16.0).abs() < 1.0e-5);
    }

    /// Selection returns the first cascade whose far boundary contains the
    /// depth, and clamps to the last cascade past the final split.
    #[test]
    fn selection_picks_first_covering_cascade() {
        let s = compute_cascade_splits(1.0, 5.0, 4, 0.0); // 2,3,4,5
        assert_eq!(select_cascade(0.5, &s), 0);
        assert_eq!(select_cascade(2.0, &s), 0);
        assert_eq!(select_cascade(2.5, &s), 1);
        assert_eq!(select_cascade(4.5, &s), 3);
        assert_eq!(select_cascade(100.0, &s), 3);
    }

    /// The blend weight ramps from 0 to 1 across the transition band and is
    /// zero for the final cascade (nothing to blend into).
    #[test]
    fn blend_weight_ramps_in_band() {
        let s = compute_cascade_splits(0.0, 4.0, 4, 0.0); // near clamped, ~1,2,3,4
                                                          // Cascade 0 spans roughly [near, 1]; use cascade 1 spanning [1, 2].
        let far = s.cascade_far(1);
        let near = s.cascade_near(1);
        let range = far - near;
        // Just inside the band (band_fraction 0.5 -> band = range/2).
        let mid = far - range * 0.25;
        let w = cascade_blend_weight(mid, &s, 1, 0.5);
        assert!(w > 0.0 && w < 1.0, "expected partial blend, got {w}");
        // Right at the far boundary -> fully blended into the next cascade.
        assert!((cascade_blend_weight(far, &s, 1, 0.5) - 1.0).abs() < 1.0e-5);
        // Before the band -> no blend.
        assert_eq!(cascade_blend_weight(near, &s, 1, 0.5), 0.0);
        // Last cascade -> never blends.
        assert_eq!(cascade_blend_weight(far, &s, 3, 0.5), 0.0);
    }

    #[test]
    fn count_and_lambda_are_clamped() {
        let s = compute_cascade_splits(1.0, 10.0, 99, 5.0);
        assert_eq!(s.count, MAX_CASCADE_COUNT);
        assert!((s.distances[MAX_CASCADE_COUNT - 1] - 10.0).abs() < 1.0e-5);
    }
}
