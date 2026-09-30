//! Variance Shadow Maps (`VSM`): the `CPU`-verifiable reference for the
//! moment-based soft-shadow test that lets a particle self-shadow and receive
//! filtered shadows without the ray-march of its siblings (design §16-§21).
//!
//! A classic depth shadow map stores one occluder depth per texel and answers
//! the shadow test with a hard `receiver > occluder` comparison, so it cannot
//! be pre-filtered (blurred, mip-mapped, or `MSAA`-resolved) the way a color
//! texture can — averaging raw depths is meaningless. `VSM` fixes this by
//! storing the first two statistical *moments* of the occluder-depth
//! distribution, `E[d]` and `E[d^2]`, per texel. Those moments *are* linear, so
//! a separable blur or a mip chain can average them directly, and the shadow
//! test becomes a probabilistic *upper bound* on the fraction of occluders in
//! front of the receiver via the one-sided `Chebyshev` inequality. The result
//! is a naturally soft, filterable shadow whose maths a future `GPU` kernel can
//! reproduce bit for bit.
//!
//! The pieces are: (1) [`Moments`] plus [`compute_moments`] turn an occluder
//! depth into its `(m1, m2)` pair; (2) [`filter_moments`] takes the weighted
//! average that models a blur tap set or a mip fetch; (3) [`variance`] recovers
//! `m2 - m1^2`; (4) [`chebyshev_upper_bound`] evaluates the one-sided
//! `Chebyshev` visibility bound `p = variance / (variance + (t - m1)^2)`,
//! returning full visibility ahead of the mean; (5) [`light_bleed_reduction`]
//! remaps the bound with a `linstep` to suppress the light-bleed artefact `VSM`
//! is known for; and (6) [`resolve_visibility`] runs the whole test over a
//! batch of texels, matching the layout a compute pass would iterate.
//!
//! # Strict scope
//!
//! This file is *only* the moment/`Chebyshev` maths of `VSM`. It shares nothing
//! with the screen-space ray-march in [`super::contact_shadow`] (which steps
//! `view`-space depth samples) or the signed-distance-field trace in
//! [`super::distance_field_shadow`] (which cone-traces an `SDF`); it neither
//! imports nor reuses their types, and it owns its own [`Moments`] record. It
//! deliberately does **not** implement the exponential warp of `EVSM` or the
//! four-moment `MSM` variant: both would require transcendental (`EVSM`) or
//! matrix-solve (`MSM`) machinery that the determinism-locked contract layer
//! forbids, so this module stays on the standard two-moment `VSM` path.
//!
//! # Determinism
//!
//! Only `f32::clamp`, `f32::max`, integer arithmetic, and multiplication are
//! used: squaring is `x * x`, never `powi`/`powf`, and no `exp`/`ln` warp is
//! applied. Every branch has a defined, `NaN`-free fallback, so a future `GPU`
//! `VSM` kernel reproduces this `CPU` reference closely.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};
use alloc::vec::Vec;

/// Denominators and interval widths with magnitude below this are treated as
/// (near) zero so the `linstep` remap and the moment average fall back to a
/// defined result instead of dividing by zero or propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided in this contract.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Byte size of the `std430` packing of a [`Moments`] record.
///
/// The two moments occupy a single `vec4` slot (`vec4(m1, m2, pad, pad)`), so
/// the block is exactly one [`VEC4_STRIDE`], the natural alignment a `GPU`
/// storage buffer of moments binds against.
pub const VARIANCE_SHADOW_STD430_SIZE: usize = VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` visibility range without branching on
/// equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The first two statistical moments of an occluder-depth distribution.
///
/// `m1` is the mean depth `E[d]` and `m2` is the mean squared depth `E[d^2]`.
/// Both are linear in the distribution, which is exactly what makes a `VSM`
/// filterable: a blur or mip fetch may average `Moments` component-wise (see
/// [`filter_moments`]) and still yield a valid moment pair.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    /// First moment `E[d]`: the mean occluder depth.
    pub m1: f32,
    /// Second moment `E[d^2]`: the mean squared occluder depth.
    pub m2: f32,
}

impl Moments {
    /// Builds a moment pair from its raw components.
    ///
    /// This is a plain constructor and does not enforce `m2 >= m1^2`; a
    /// physically consistent pair comes from [`Moments::from_depth`] or a
    /// convex [`filter_moments`] average of such pairs.
    #[must_use]
    pub const fn new(m1: f32, m2: f32) -> Self {
        Self { m1, m2 }
    }

    /// The moment pair of a single occluder at `depth`: `m1 = d`, `m2 = d^2`.
    ///
    /// A point distribution has zero variance, so this pair reproduces the hard
    /// depth-map comparison until it is averaged with neighbours.
    #[must_use]
    pub fn from_depth(depth: f32) -> Self {
        Self::new(depth, depth * depth)
    }
}

/// The moment pair of a single occluder `depth` (free-function form of
/// [`Moments::from_depth`], mirroring the per-texel write a shadow pass emits).
#[must_use]
pub fn compute_moments(depth: f32) -> Moments {
    Moments::from_depth(depth)
}

/// The weighted average of a set of `samples`, modelling a blur tap set or a
/// mip-level fetch over the moment texture.
///
/// Each sample is paired with the matching entry in `weights` (extra entries in
/// either slice are ignored). The moments are averaged component-wise and
/// normalized by the total weight, which is exactly the operation a separable
/// `VSM` blur performs. A vanishing (or absent) total weight yields the zero
/// moment pair rather than dividing by zero.
#[must_use]
pub fn filter_moments(samples: &[Moments], weights: &[f32]) -> Moments {
    let mut total_weight = 0.0_f32;
    let mut acc_m1 = 0.0_f32;
    let mut acc_m2 = 0.0_f32;
    for (sample, &weight) in samples.iter().zip(weights.iter()) {
        total_weight += weight;
        acc_m1 += weight * sample.m1;
        acc_m2 += weight * sample.m2;
    }
    if total_weight.abs() < MIN_DENOM {
        return Moments::new(0.0, 0.0);
    }
    let inv = 1.0 / total_weight;
    Moments::new(acc_m1 * inv, acc_m2 * inv)
}

/// The variance `m2 - m1^2` of a moment pair, clamped to be non-negative.
///
/// Exact arithmetic keeps `m2 >= m1^2`, but a filtered or quantized moment pair
/// can dip slightly below, producing a tiny negative variance that would make
/// the `Chebyshev` bound ill-defined; clamping to `0` keeps the shadow test
/// stable. The `min_variance` floor that guards the division is applied later,
/// in [`chebyshev_upper_bound`].
#[must_use]
pub fn variance(moments: &Moments) -> f32 {
    (moments.m2 - moments.m1 * moments.m1).max(0.0)
}

/// The one-sided `Chebyshev` upper bound on the visibility of a receiver at
/// `receiver_depth` given the occluder `moments`.
///
/// When the receiver sits at or in front of the mean occluder depth
/// (`receiver_depth <= m1`) it cannot be occluded by this distribution, so the
/// bound is full visibility `1.0`. Otherwise the bound is `p = variance /
/// (variance + (t - m1)^2)`, where the variance is floored at `min_variance` to
/// bound light bleed and avoid a razor-thin denominator. The `(t - m1)^2` term
/// is strictly positive in this branch, so the denominator never vanishes. The
/// result is clamped into `0..=1`.
#[must_use]
pub fn chebyshev_upper_bound(moments: &Moments, receiver_depth: f32, min_variance: f32) -> f32 {
    if receiver_depth <= moments.m1 {
        return 1.0;
    }
    let var = variance(moments).max(min_variance);
    let diff = receiver_depth - moments.m1;
    let denom = var + diff * diff;
    if denom < MIN_DENOM {
        return 0.0;
    }
    clamp01(var / denom)
}

/// Suppresses `VSM` light bleed by remapping a visibility bound `p` with a
/// `linstep`: the `[amount, 1]` sub-range is stretched onto `[0, 1]`, so any
/// partial visibility below `amount` is driven to fully shadowed.
///
/// Light bleed is the artefact where a bright background leaks through a thin
/// occluder because the `Chebyshev` bound over-estimates visibility; clamping
/// off the low tail hides it at the cost of slightly darker penumbrae. A
/// degenerate `amount` at (or above) `1` collapses the remap to a hard step so
/// no divide-by-zero occurs. The output stays in `0..=1`.
#[must_use]
pub fn light_bleed_reduction(p: f32, amount: f32) -> f32 {
    let lo = clamp01(amount);
    let span = 1.0 - lo;
    if span < MIN_DENOM {
        return if p < lo { 0.0 } else { 1.0 };
    }
    clamp01((p - lo) / span)
}

/// Resolves the light-bleed-corrected `VSM` visibility for a batch of texels.
///
/// Entry `i` pairs `moments_map[i]` with `receiver_depths[i]` (extra entries in
/// either slice are ignored), evaluates [`chebyshev_upper_bound`] with the
/// shared `min_variance` floor, and folds the result through
/// [`light_bleed_reduction`] with the shared `bleed` amount. The output vector
/// has one visibility in `0..=1` per resolved texel, matching the flat layout a
/// `GPU` compute pass would write.
#[must_use]
pub fn resolve_visibility(
    moments_map: &[Moments],
    receiver_depths: &[f32],
    min_variance: f32,
    bleed: f32,
) -> Vec<f32> {
    moments_map
        .iter()
        .zip(receiver_depths.iter())
        .map(|(moments, &receiver_depth)| {
            let p = chebyshev_upper_bound(moments, receiver_depth, min_variance);
            light_bleed_reduction(p, bleed)
        })
        .collect()
}

/// Serializes a [`Moments`] record into its `std430` `vec4` slot.
///
/// The layout is `vec4(m1, m2, pad, pad)`: the two moments occupy the first two
/// lanes and the trailing two lanes are zero padding, matching
/// [`VARIANCE_SHADOW_STD430_SIZE`]. All scalars are little-endian `f32`.
#[must_use]
pub fn to_std430(moments: &Moments) -> [u8; VARIANCE_SHADOW_STD430_SIZE] {
    let mut bytes = [0u8; VARIANCE_SHADOW_STD430_SIZE];
    bytes[0..4].copy_from_slice(&moments.m1.to_le_bytes());
    bytes[4..8].copy_from_slice(&moments.m2.to_le_bytes());
    bytes
}

/// Total `GPU` storage size for `count` serialized [`Moments`], clamped up to a
/// single element so an empty batch still yields a valid storage binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(VARIANCE_SHADOW_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts two `f32` values agree within [`CMP_EPS`] without a direct `==`.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn moments_new_stores_components() {
        let m = Moments::new(2.0, 5.0);
        assert!(approx(m.m1, 2.0));
        assert!(approx(m.m2, 5.0));
    }

    #[test]
    fn from_depth_squares_second_moment() {
        let m = Moments::from_depth(3.0);
        assert!(approx(m.m1, 3.0));
        assert!(approx(m.m2, 9.0));
    }

    #[test]
    fn compute_moments_matches_from_depth() {
        let d = 0.75;
        assert!(approx(compute_moments(d).m1, Moments::from_depth(d).m1));
        assert!(approx(compute_moments(d).m2, Moments::from_depth(d).m2));
    }

    #[test]
    fn default_moments_are_zero() {
        let m = Moments::default();
        assert!(approx(m.m1, 0.0));
        assert!(approx(m.m2, 0.0));
    }

    #[test]
    fn filter_moments_uniform_weights_equals_mean() {
        let samples = [
            Moments::from_depth(1.0),
            Moments::from_depth(2.0),
            Moments::from_depth(3.0),
        ];
        let weights = [1.0, 1.0, 1.0];
        let filtered = filter_moments(&samples, &weights);
        // mean depth = 2, mean square = (1 + 4 + 9) / 3 = 14/3.
        assert!(approx(filtered.m1, 2.0));
        assert!(approx(filtered.m2, 14.0 / 3.0));
    }

    #[test]
    fn filter_moments_weighted_average_is_biased() {
        let samples = [Moments::from_depth(0.0), Moments::from_depth(4.0)];
        let weights = [3.0, 1.0];
        let filtered = filter_moments(&samples, &weights);
        // m1 = (3*0 + 1*4) / 4 = 1, m2 = (3*0 + 1*16) / 4 = 4.
        assert!(approx(filtered.m1, 1.0));
        assert!(approx(filtered.m2, 4.0));
    }

    #[test]
    fn filter_moments_zero_total_weight_is_zero() {
        let samples = [Moments::from_depth(1.0), Moments::from_depth(2.0)];
        let weights = [0.0, 0.0];
        let filtered = filter_moments(&samples, &weights);
        assert!(approx(filtered.m1, 0.0));
        assert!(approx(filtered.m2, 0.0));
    }

    #[test]
    fn filter_moments_empty_is_zero() {
        let filtered = filter_moments(&[], &[]);
        assert!(approx(filtered.m1, 0.0));
        assert!(approx(filtered.m2, 0.0));
    }

    #[test]
    fn variance_is_m2_minus_m1_squared() {
        let samples = [Moments::from_depth(1.0), Moments::from_depth(3.0)];
        let weights = [1.0, 1.0];
        let filtered = filter_moments(&samples, &weights);
        // m1 = 2, m2 = 5, variance = 5 - 4 = 1.
        assert!(approx(variance(&filtered), 1.0));
    }

    #[test]
    fn variance_of_single_depth_is_zero() {
        assert!(approx(variance(&Moments::from_depth(7.0)), 0.0));
    }

    #[test]
    fn variance_never_negative() {
        // A degenerate pair with m2 < m1^2 clamps to zero rather than negative.
        let bad = Moments::new(2.0, 1.0);
        assert!(variance(&bad) >= 0.0);
        assert!(approx(variance(&bad), 0.0));
    }

    #[test]
    fn chebyshev_fully_visible_before_occluder() {
        let m = filter_moments(
            &[Moments::from_depth(4.0), Moments::from_depth(6.0)],
            &[1.0, 1.0],
        );
        // receiver at depth 1 sits in front of the mean (5) -> fully lit.
        assert!(approx(chebyshev_upper_bound(&m, 1.0, 1e-4), 1.0));
    }

    #[test]
    fn chebyshev_at_mean_is_fully_visible() {
        let m = filter_moments(
            &[Moments::from_depth(4.0), Moments::from_depth(6.0)],
            &[1.0, 1.0],
        );
        assert!(approx(chebyshev_upper_bound(&m, m.m1, 1e-4), 1.0));
    }

    #[test]
    fn chebyshev_beyond_mean_is_less_than_one() {
        let m = filter_moments(
            &[Moments::from_depth(4.0), Moments::from_depth(6.0)],
            &[1.0, 1.0],
        );
        let p = chebyshev_upper_bound(&m, 8.0, 1e-4);
        assert!(p < 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn chebyshev_stays_in_unit_range() {
        let m = filter_moments(
            &[Moments::from_depth(2.0), Moments::from_depth(10.0)],
            &[1.0, 1.0],
        );
        for step in 0u8..=40 {
            let t = f32::from(step) * 0.5;
            let p = chebyshev_upper_bound(&m, t, 1e-4);
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn chebyshev_zero_variance_is_hard_step() {
        // A single occluder depth -> zero variance -> hard shadow boundary.
        let m = Moments::from_depth(5.0);
        // In front of the occluder: fully lit.
        assert!(approx(chebyshev_upper_bound(&m, 4.999, 0.0), 1.0));
        // Behind the occluder with no variance floor: fully shadowed.
        assert!(approx(chebyshev_upper_bound(&m, 5.001, 0.0), 0.0));
    }

    #[test]
    fn chebyshev_monotonic_decreasing_with_depth() {
        let m = filter_moments(
            &[Moments::from_depth(3.0), Moments::from_depth(7.0)],
            &[1.0, 1.0],
        );
        let mut prev = 1.0_f32;
        for step in 0u8..=30 {
            let t = 5.0 + f32::from(step) * 0.5;
            let p = chebyshev_upper_bound(&m, t, 1e-4);
            assert!(p <= prev + CMP_EPS);
            prev = p;
        }
    }

    #[test]
    fn chebyshev_min_variance_bounds_bleed() {
        // A larger min_variance raises the bound (more bleed) at a fixed depth.
        let m = Moments::from_depth(5.0);
        let low = chebyshev_upper_bound(&m, 6.0, 1e-4);
        let high = chebyshev_upper_bound(&m, 6.0, 1.0);
        assert!(high >= low);
    }

    #[test]
    fn light_bleed_identity_at_zero_amount() {
        for step in 0u8..=10 {
            let p = f32::from(step) * 0.1;
            assert!(approx(light_bleed_reduction(p, 0.0), clamp01(p)));
        }
    }

    #[test]
    fn light_bleed_clamps_below_amount_to_zero() {
        assert!(approx(light_bleed_reduction(0.1, 0.3), 0.0));
        assert!(approx(light_bleed_reduction(0.3, 0.3), 0.0));
    }

    #[test]
    fn light_bleed_reaches_one_at_full_visibility() {
        assert!(approx(light_bleed_reduction(1.0, 0.3), 1.0));
    }

    #[test]
    fn light_bleed_is_monotonic_non_decreasing() {
        let amount = 0.2;
        let mut prev = -1.0_f32;
        for step in 0u8..=20 {
            let p = f32::from(step) * 0.05;
            let reduced = light_bleed_reduction(p, amount);
            assert!(reduced >= prev - CMP_EPS);
            prev = reduced;
        }
    }

    #[test]
    fn light_bleed_stays_in_unit_range() {
        for step in 0u8..=20 {
            let p = f32::from(step) * 0.05;
            let reduced = light_bleed_reduction(p, 0.4);
            assert!((0.0..=1.0).contains(&reduced));
        }
    }

    #[test]
    fn light_bleed_degenerate_amount_is_hard_step() {
        // amount == 1 collapses to a hard step, no divide-by-zero.
        assert!(approx(light_bleed_reduction(0.99, 1.0), 0.0));
        assert!(approx(light_bleed_reduction(1.0, 1.0), 1.0));
    }

    #[test]
    fn light_bleed_midpoint_remaps_linearly() {
        // linstep(0.2, 1.0, 0.6) = (0.6 - 0.2) / 0.8 = 0.5.
        assert!(approx(light_bleed_reduction(0.6, 0.2), 0.5));
    }

    #[test]
    fn resolve_visibility_matches_manual() {
        let map = [
            Moments::from_depth(2.0),
            filter_moments(
                &[Moments::from_depth(2.0), Moments::from_depth(6.0)],
                &[1.0, 1.0],
            ),
        ];
        let depths = [1.0, 8.0];
        let out = resolve_visibility(&map, &depths, 1e-4, 0.1);
        let expected0 = light_bleed_reduction(chebyshev_upper_bound(&map[0], 1.0, 1e-4), 0.1);
        let expected1 = light_bleed_reduction(chebyshev_upper_bound(&map[1], 8.0, 1e-4), 0.1);
        assert!(approx(out[0], expected0));
        assert!(approx(out[1], expected1));
    }

    #[test]
    fn resolve_visibility_length_matches_input() {
        let map = [
            Moments::from_depth(1.0),
            Moments::from_depth(2.0),
            Moments::from_depth(3.0),
        ];
        let depths = [0.5, 2.5, 4.0];
        let out = resolve_visibility(&map, &depths, 1e-4, 0.0);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn resolve_visibility_empty_is_empty() {
        let out = resolve_visibility(&[], &[], 1e-4, 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(VARIANCE_SHADOW_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(VARIANCE_SHADOW_STD430_SIZE, 16);
    }

    #[test]
    fn std430_layout_roundtrips() {
        let m = Moments::new(1.5, 3.25);
        let bytes = to_std430(&m);
        assert_eq!(bytes.len(), VARIANCE_SHADOW_STD430_SIZE);
        let mut lane = [0u8; 4];
        lane.copy_from_slice(&bytes[0..4]);
        assert!(approx(f32::from_le_bytes(lane), 1.5));
        lane.copy_from_slice(&bytes[4..8]);
        assert!(approx(f32::from_le_bytes(lane), 3.25));
        // Trailing padding lanes are zeroed.
        assert_eq!(&bytes[8..16], &[0u8; 8]);
    }

    #[test]
    fn gpu_storage_bytes_reserves_one_element_when_empty() {
        assert_eq!(gpu_storage_bytes(0), VARIANCE_SHADOW_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(1), VARIANCE_SHADOW_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(4), 4 * VARIANCE_SHADOW_STD430_SIZE);
    }
}
