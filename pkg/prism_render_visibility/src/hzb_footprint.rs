//! Hierarchical-Z (HZB) occlusion-footprint math for two-phase culling.
//!
//! Nanite-style occlusion culling tests a bounding volume against a reverse-Z
//! *min-reduction* depth pyramid. Picking the right pyramid level is the half
//! of the test [`HzbTest`](crate::HzbTest) does **not** do: the candidate's
//! projected bounds must be reduced to a single conservative occluder depth
//! before the comparison is meaningful. This module supplies that reduction in
//! two exact, allocation-free, `no_std`-safe steps that mirror UE's HZB bound
//! test:
//!
//! 1. [`HzbFootprint::sample_mip`] selects the coarsest-safe mip whose `2x2`
//!    texel gather fully covers the screen-space footprint, matching the UE
//!    rule `ceil(log2(max(width, height)))` evaluated on integer texel spans so
//!    the choice is bit-identical on `CPU` and `GPU` (no `log2`, no rounding
//!    intrinsic, hence no libm/FMA divergence).
//! 2. [`conservative_occluder_reverse_z`] reduces the sampled `2x2` gather to
//!    the single farthest occluder — the *minimum* reverse-Z value — so a
//!    candidate is only ever rejected when it is behind every texel in the
//!    footprint. Non-finite taps are ignored and an empty gather yields
//!    [`None`], keeping the candidate visible rather than wrongly culled.
//!
//! The output pair `(mip, occluder_depth)` feeds straight into
//! [`HzbTest`](crate::HzbTest): `sample_mip` fills `HzbTest::mip` and the
//! reduced depth fills `HzbTest::occluder_depth`.

/// Screen-space, pixel-space axis-aligned bounds of a projected volume,
/// measured in mip-0 HZB texels (one texel per pixel). `min`/`max` are
/// `[x, y]`; callers obtain them by projecting the world-space bounds and
/// mapping clip space to the view's pixel viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HzbFootprint {
    /// Lower-left corner in mip-0 texels (`[x, y]`).
    pub min: [f32; 2],
    /// Upper-right corner in mip-0 texels (`[x, y]`).
    pub max: [f32; 2],
}

impl HzbFootprint {
    /// Builds a footprint from its pixel-space corners.
    pub const fn new(min: [f32; 2], max: [f32; 2]) -> Self {
        Self { min, max }
    }

    /// Width and height of the footprint in mip-0 texels. A degenerate,
    /// inverted, or non-finite span collapses to `0.0` on that axis so mip
    /// selection stays conservative (never samples a coarser mip than the
    /// bounds justify).
    pub fn extent(self) -> [f32; 2] {
        [
            sanitize_span(self.max[0] - self.min[0]),
            sanitize_span(self.max[1] - self.min[1]),
        ]
    }

    /// Selects the HZB mip whose `2x2` gather conservatively covers this
    /// footprint.
    ///
    /// The level is `ceil(log2(ceil(max(width, height))))`, clamped to
    /// `[0, sampled_mip_count - 1]`. A sub-texel, degenerate, or non-finite
    /// footprint selects mip `0`. When `sampled_mip_count` is `0` (no pyramid)
    /// the result is `0`; the caller is expected to treat an empty pyramid as
    /// "cannot reject" via [`HzbTest`](crate::HzbTest)'s own guards.
    pub fn sample_mip(self, sampled_mip_count: u32) -> u32 {
        if sampled_mip_count == 0 {
            return 0;
        }
        let [width, height] = self.extent();
        let span = if width >= height { width } else { height };
        if !span.is_finite() || span <= 1.0 {
            return 0;
        }
        let span_texels = ceil_to_u32(span);
        let mip = ceil_log2_u32(span_texels);
        let max_mip = sampled_mip_count - 1;
        if mip > max_mip {
            max_mip
        } else {
            mip
        }
    }
}

/// Reduces a reverse-Z HZB gather (typically the `2x2` taps at the mip chosen
/// by [`HzbFootprint::sample_mip`]) to the single conservative occluder depth:
/// the farthest surface, i.e. the *minimum* reverse-Z value. Non-finite taps
/// are skipped; a gather with no finite taps returns [`None`] so the caller
/// keeps the candidate visible.
pub fn conservative_occluder_reverse_z(samples: &[f32]) -> Option<f32> {
    let mut occluder: Option<f32> = None;
    for &sample in samples {
        if !sample.is_finite() {
            continue;
        }
        occluder = Some(match occluder {
            Some(current) if current <= sample => current,
            _ => sample,
        });
    }
    occluder
}

/// Non-finite, zero, or negative spans collapse to `0.0`.
fn sanitize_span(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Integer ceil of a non-negative, finite `f32`. Truncating casts round toward
/// zero, so a positive fractional part bumps the result by one. Saturates to
/// [`u32::MAX`] for out-of-range inputs, which [`ceil_log2_u32`] maps to the
/// maximum level before clamping.
fn ceil_to_u32(value: f32) -> u32 {
    let truncated = value as u32;
    if (truncated as f32) < value {
        truncated.saturating_add(1)
    } else {
        truncated
    }
}

/// `ceil(log2(n))` over integers, with `ceil_log2(0) == ceil_log2(1) == 0`.
/// Exact and branch-cheap: for `n > 1` the value is the bit width minus the
/// leading zeros of `n - 1`.
const fn ceil_log2_u32(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        u32::BITS - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HzbPhase, HzbTest, ViewFlags};

    #[test]
    fn extent_is_nonnegative_and_rejects_degenerate_or_nonfinite_spans() {
        assert_eq!(
            HzbFootprint::new([10.0, 20.0], [14.0, 23.0]).extent(),
            [4.0, 3.0]
        );
        // Inverted bounds collapse to zero rather than reporting a negative span.
        assert_eq!(
            HzbFootprint::new([14.0, 23.0], [10.0, 20.0]).extent(),
            [0.0, 0.0]
        );
        // Non-finite corners collapse to zero.
        assert_eq!(
            HzbFootprint::new([f32::NAN, 0.0], [1.0, f32::INFINITY]).extent(),
            [0.0, 0.0]
        );
    }

    #[test]
    fn sample_mip_matches_ceil_log2_of_the_texel_span() {
        let plenty = 16;
        // 1x1 (and sub-texel) footprints need no coarser mip.
        assert_eq!(HzbFootprint::new([0.0, 0.0], [1.0, 1.0]).sample_mip(plenty), 0);
        assert_eq!(HzbFootprint::new([0.0, 0.0], [0.5, 0.5]).sample_mip(plenty), 0);
        // 4 texels wide -> ceil(log2(4)) = 2.
        assert_eq!(HzbFootprint::new([0.0, 0.0], [4.0, 2.0]).sample_mip(plenty), 2);
        // 5 texels tall -> ceil(log2(5)) = 3, driven by the larger axis.
        assert_eq!(HzbFootprint::new([0.0, 0.0], [2.0, 5.0]).sample_mip(plenty), 3);
        // A fractional span ceils before the log: 4.1 -> 5 -> ceil(log2(5)) = 3.
        assert_eq!(HzbFootprint::new([0.0, 0.0], [4.1, 1.0]).sample_mip(plenty), 3);
    }

    #[test]
    fn sample_mip_clamps_to_available_pyramid_levels() {
        // A 1024-texel span wants mip 10 but only mips 0..=2 exist.
        let huge = HzbFootprint::new([0.0, 0.0], [1024.0, 1024.0]);
        assert_eq!(huge.sample_mip(3), 2);
        // An empty pyramid pins the mip to zero; HzbTest's own guards reject it.
        assert_eq!(huge.sample_mip(0), 0);
        // Non-finite footprints never select a coarse mip.
        assert_eq!(
            HzbFootprint::new([0.0, 0.0], [f32::INFINITY, 1.0]).sample_mip(8),
            0
        );
    }

    #[test]
    fn conservative_occluder_takes_the_farthest_reverse_z_tap() {
        // Reverse-Z: smaller value is farther, so the min is the conservative
        // (least-likely-to-reject) occluder.
        assert_eq!(
            conservative_occluder_reverse_z(&[0.5, 0.2, 0.4, 0.3]),
            Some(0.2)
        );
        // Non-finite taps are ignored, not propagated.
        assert_eq!(
            conservative_occluder_reverse_z(&[f32::NAN, 0.6, f32::INFINITY, 0.55]),
            Some(0.55)
        );
        // No finite taps -> None, so the caller keeps the candidate visible.
        assert_eq!(conservative_occluder_reverse_z(&[]), None);
        assert_eq!(
            conservative_occluder_reverse_z(&[f32::NAN, f32::NEG_INFINITY]),
            None
        );
    }

    #[test]
    fn footprint_reduction_feeds_the_hzb_test_end_to_end() {
        // A 4x3-texel footprint selects mip 2; its 2x2 gather's farthest tap is
        // 0.75. A candidate whose nearest point sits at 0.25 is behind that
        // occluder and must be rejected.
        let footprint = HzbFootprint::new([100.0, 50.0], [104.0, 53.0]);
        let mip = footprint.sample_mip(8);
        assert_eq!(mip, 2);
        let occluder = conservative_occluder_reverse_z(&[0.80, 0.75, 0.78, 0.76])
            .expect("finite gather");
        let occluded = HzbTest {
            nearest_depth: 0.25,
            occluder_depth: occluder,
            depth_bias: 0.0,
            projected_velocity: 0.0,
            mip,
            sampled_mip_count: 8,
            history_epoch: 1,
            expected_history_epoch: 1,
            view_flags: ViewFlags::REVERSE_Z,
        };
        assert!(occluded.is_occluded(HzbPhase::Current));
        // A candidate whose nearest point is in front of the farthest occluder
        // survives the same test.
        assert!(!HzbTest { nearest_depth: 0.90, ..occluded }.is_occluded(HzbPhase::Current));
    }
}
