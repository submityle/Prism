//! Screen-space contact shadows (`SSCS`): the `CPU`-verifiable reference for the
//! short-range ray-march that grounds particles against nearby geometry
//! (design §16-§21).
//!
//! A translucent sprite floating just above a surface leaves a tell-tale gap
//! unless *something* darkens the few pixels where it nearly touches. Cascaded
//! or distance-field shadow maps are too coarse for that contact region, so
//! production engines add a cheap screen-space pass: from the shaded point they
//! step a handful of samples along the light direction, read the already-shaded
//! scene depth at each step, and darken the point when the ray dips *behind* a
//! surface that sits between it and the camera. Because the whole march lives in
//! `view`-space depth (larger means farther from the camera), a future `GPU`
//! kernel can reproduce this reference bit for bit.
//!
//! The pieces are: (1) a self-contained integer hash turns a per-pixel seed into
//! a sub-step jitter offset ([`jitter_offset`]) so the fixed step count does not
//! leave visible banding; (2) [`march_fractions`] lays out the jittered,
//! normalized march positions; (3) [`contact_shadow_occlusion`] walks those
//! positions, tests each depth gap against the `(bias .. bias + thickness)`
//! acceptance window with [`RangeBounds::contains`](core::ops::RangeBounds), and
//! folds the nearest hit into an occlusion factor via a rational distance
//! falloff plus a `smoothstep` soft edge; and (4) [`ContactShadowParams`] gathers
//! the tunables in a `std430`-friendly record whose `GPU` packing follows the
//! shared `vec4` alignment from [`super::gpu_layout`].
//!
//! Deliberately out of scope: cascaded or distance-field shadows (a future
//! `distance_field_shadow`), hierarchical-Z occlusion *culling* (the `HzbPyramid`
//! in [`super::occlusion`]), and soft-particle depth fade
//! ([`super::soft_particle`]). This file marches the contact region and nothing
//! else, and imports no sibling particle module beyond [`super::gpu_layout`].
//!
//! Only `f32::sqrt` / `f32::floor`, integer arithmetic, and integer hashing are
//! used — no transcendental functions — so results stay deterministic and
//! platform independent.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Soft-edge / denominator guard below which a `smoothstep` interval collapses
/// to a hard step, so no divide-by-zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// `2^32` as an `f32`: the normalizing span turning a `u32` hash word into a
/// unit-interval fraction.
const U32_SPAN: f32 = 4_294_967_296.0;

/// Byte stride of one [`ContactShadowParams`] record in a `std430` storage
/// buffer.
///
/// The six scalars pack into two `vec4` slots: `vec4(step_count, max_distance,
/// thickness, bias)` followed by `vec4(intensity, falloff, pad, pad)`.
pub const CONTACT_SHADOW_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-equal)
/// interval collapses to a hard step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Integer avalanche hash mixing a `u32` seed into a well-distributed `u32`.
///
/// A self-contained mixer (no dependency on any noise or determinism module):
/// xor-shifts and odd-constant multiplies giving a deterministic,
/// platform-independent word.
#[must_use]
fn hash_u32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Normalizes a `u32` hash word into a `0..1` unit fraction.
#[must_use]
fn to_unit(x: f32) -> f32 {
    x / U32_SPAN
}

/// Deterministic sub-step jitter offset in `[0, 1)` for a per-pixel `seed`.
///
/// The march uses a fixed step count, so every pixel would otherwise sample the
/// exact same normalized positions and leave stair-stepped banding along shadow
/// edges. Offsetting the whole march by a hashed fraction of one step
/// decorrelates neighbours while staying bit-reproducible for a given `seed`.
#[must_use]
pub fn jitter_offset(seed: u32) -> f32 {
    to_unit(hash_u32(seed) as f32)
}

/// Builds the per-step jittered march fractions in `[0, 1)`.
///
/// Each entry is `(index + jitter) / step_count`, the normalized distance along
/// the light ray at which the march reads the scene depth. The shared jitter
/// (see [`jitter_offset`]) shifts every position by the same sub-step fraction,
/// and a `step_count` of `0` yields an empty schedule. This is exposed so a
/// `GPU` kernel can be cross-checked against the exact positions this reference
/// samples.
#[must_use]
pub fn march_fractions(params: &ContactShadowParams, jitter_seed: u32) -> Vec<f32> {
    let steps = usize::try_from(params.step_count).unwrap_or(usize::MAX);
    if steps == 0 {
        return Vec::new();
    }
    let jitter = jitter_offset(jitter_seed);
    let inv_steps = 1.0 / (params.step_count as f32);
    (0..steps)
        .map(|index| (index as f32 + jitter) * inv_steps)
        .collect()
}

/// Marches the contact region and returns an occlusion factor in `0..=1`.
///
/// `start_depth` is the `view`-space depth of the shaded point and
/// `ray_depth_span` is the signed change in `view`-space depth accumulated over
/// the full march toward the light (larger depth means farther from the camera).
/// `scene_depths` holds the scene depth fetched at each march step, in march
/// order. At step `k` the ray sits at `ray_z = start_depth + ray_depth_span *
/// frac`, and the depth gap `diff = ray_z - scene_z` is positive when the scene
/// surface is *nearer* than the ray — a potential occluder. A step counts as a
/// hit only when `diff` lands inside the half-open window
/// `(bias .. bias + thickness)`: the `bias` lower bound skips grazing
/// self-contact that would otherwise cause acne, and the `thickness` width caps
/// how far behind a surface the ray may pass before it is assumed to have
/// slipped through a thin object into empty space.
///
/// The nearest hit wins. Its shadow strength is `intensity` scaled by a rational
/// distance falloff `1 / (1 + falloff * dist * dist)` (nearer contacts darken
/// more) and a `smoothstep` soft edge that fades shadows toward the end of the
/// search radius. The point's lighting is then `clamp(1 - shadow, 0, 1)`, so the
/// result follows the convention **`1.0` = fully lit / unoccluded** and **`0.0`
/// = fully shadowed**. When no step hits — including an empty schedule or a
/// non-positive `thickness`, whose window is empty — the point stays fully lit
/// (`1.0`).
#[must_use]
pub fn contact_shadow_occlusion(
    params: &ContactShadowParams,
    start_depth: f32,
    ray_depth_span: f32,
    scene_depths: &[f32],
    jitter_seed: u32,
) -> f32 {
    let lo = params.bias;
    let hi = params.bias + params.thickness;
    march_fractions(params, jitter_seed)
        .iter()
        .zip(scene_depths)
        .find_map(|(&frac, &scene_z)| {
            let ray_z = start_depth + ray_depth_span * frac;
            let diff = ray_z - scene_z;
            (lo..hi).contains(&diff).then(|| {
                let dist = frac * params.max_distance;
                let falloff = 1.0 / (1.0 + params.falloff * dist * dist);
                let edge = 1.0 - smoothstep(0.0, 1.0, frac);
                let shadow = clamp01(params.intensity * falloff * edge);
                clamp01(1.0 - shadow)
            })
        })
        .unwrap_or(1.0)
}

/// Number of `GPU` workgroups needed to cover `pixel_count` invocations at
/// `group_size` threads each, via integer `div_ceil` (never zero-sized).
#[must_use]
pub fn dispatch_groups(pixel_count: u32, group_size: u32) -> u32 {
    pixel_count.div_ceil(group_size.max(1))
}

/// Tunables for the screen-space contact-shadow march (design §16-§21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactShadowParams {
    /// Number of samples taken along the light ray (a fixed short march).
    pub step_count: u32,
    /// `view`-space distance the march covers from the shaded point toward the
    /// light; scales the rational distance falloff.
    pub max_distance: f32,
    /// Width of the acceptance window: a depth gap wider than this is assumed to
    /// pass behind a thin object into empty space and does not occlude.
    pub thickness: f32,
    /// Minimum depth gap that counts as an occluder, skipping grazing
    /// self-contact that would otherwise produce shadow acne.
    pub bias: f32,
    /// Scales how strongly a contact hit darkens the shaded point.
    pub intensity: f32,
    /// Rational distance-falloff coefficient; larger fades distant contacts
    /// faster.
    pub falloff: f32,
}

impl ContactShadowParams {
    /// Creates contact-shadow parameters from all fields.
    #[must_use]
    pub const fn new(
        step_count: u32,
        max_distance: f32,
        thickness: f32,
        bias: f32,
        intensity: f32,
        falloff: f32,
    ) -> Self {
        Self {
            step_count,
            max_distance,
            thickness,
            bias,
            intensity,
            falloff,
        }
    }

    /// Evaluates the contact-shadow occlusion factor in `0..=1` for these
    /// parameters; see [`contact_shadow_occlusion`] for the full contract.
    #[must_use]
    pub fn occlusion(
        &self,
        start_depth: f32,
        ray_depth_span: f32,
        scene_depths: &[f32],
        jitter_seed: u32,
    ) -> f32 {
        contact_shadow_occlusion(self, start_depth, ray_depth_span, scene_depths, jitter_seed)
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[step_count, max_distance, thickness, bias, intensity, falloff,
    /// pad, pad]` as raw `u32` words (the five `f32` fields via `f32::to_bits`)
    /// — two `vec4` slots, matching [`CONTACT_SHADOW_PARAMS_STRIDE`]. The
    /// trailing words are padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.step_count,
            self.max_distance.to_bits(),
            self.thickness.to_bits(),
            self.bias.to_bits(),
            self.intensity.to_bits(),
            self.falloff.to_bits(),
            0,
            0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`ContactShadowParams`] records.
///
/// Uses [`CONTACT_SHADOW_PARAMS_STRIDE`] and the shared clamp-to-one-element
/// rule from [`storage_bytes`], so an empty set still yields a valid `GPU`
/// binding.
#[must_use]
pub fn contact_shadow_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(CONTACT_SHADOW_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// Parameters with no `bias` and a `0.5` thickness window, `1.0` intensity.
    fn base_params(step_count: u32) -> ContactShadowParams {
        ContactShadowParams::new(step_count, 1.0, 0.5, 0.0, 1.0, 1.0)
    }

    #[test]
    fn no_occluder_stays_fully_lit() {
        // Ray holds depth 10; every scene sample sits far behind it, so the gap
        // is negative and nothing lands in the window.
        let params = base_params(8);
        let depths = [20.0_f32; 8];
        assert!(approx_eq(params.occlusion(10.0, 0.0, &depths, 1), 1.0));
    }

    #[test]
    fn direct_occluder_darkens_the_point() {
        // A surface 0.2 nearer than the ray sits inside (0.0 .. 0.5): a hit.
        let params = base_params(8);
        let depths = [9.8_f32; 8];
        let factor = params.occlusion(10.0, 0.0, &depths, 7);
        assert!(factor < 1.0);
        assert!((0.0..=1.0).contains(&factor));
    }

    #[test]
    fn gap_wider_than_thickness_does_not_occlude() {
        // Gap of 0.6 exceeds the 0.5 window upper bound -> the ray is assumed to
        // have slipped behind a thin object into empty space.
        let params = base_params(8);
        let depths = [9.4_f32; 8];
        assert!(approx_eq(params.occlusion(10.0, 0.0, &depths, 3), 1.0));
        // Shrinking the gap back inside the window brings the hit back.
        let inside = [9.6_f32; 8];
        assert!(params.occlusion(10.0, 0.0, &inside, 3) < 1.0);
    }

    #[test]
    fn gap_below_bias_is_skipped_as_self_contact() {
        // Window (0.3 .. 0.8); a 0.2 gap is grazing self-contact and skipped.
        let params = ContactShadowParams::new(8, 1.0, 0.5, 0.3, 1.0, 1.0);
        let depths = [9.8_f32; 8];
        assert!(approx_eq(params.occlusion(10.0, 0.0, &depths, 5), 1.0));
    }

    #[test]
    fn bias_shifts_the_acceptance_window() {
        // The same 0.2 gap is a hit with zero bias...
        let depths = [9.8_f32; 8];
        let no_bias = ContactShadowParams::new(8, 1.0, 0.5, 0.0, 1.0, 1.0);
        assert!(no_bias.occlusion(10.0, 0.0, &depths, 9) < 1.0);
        // ...and a miss once the bias lifts the window's lower bound past it.
        let biased = ContactShadowParams::new(8, 1.0, 0.5, 0.3, 1.0, 1.0);
        assert!(approx_eq(biased.occlusion(10.0, 0.0, &depths, 9), 1.0));
    }

    #[test]
    fn distance_falloff_is_monotone_in_hit_distance() {
        // Isolate a single hit at step `k` by making only that step's surface
        // land inside the window; every other step is far behind the ray.
        let params = base_params(8);
        let seed = 42;
        let single_hit = |k: usize| -> f32 {
            let mut depths = [20.0_f32; 8];
            depths[k] = 9.8;
            params.occlusion(10.0, 0.0, &depths, seed)
        };
        // A nearer hit (smaller step index) casts a stronger shadow, so the lit
        // factor rises with hit distance.
        let mut prev = -1.0;
        for k in 0..8 {
            let factor = single_hit(k);
            assert!(factor >= prev - CMP_EPS, "not monotone at step {k}");
            assert!((0.0..=1.0).contains(&factor));
            prev = factor;
        }
        // The near end is strictly darker than the far end.
        assert!(single_hit(0) < single_hit(7));
    }

    #[test]
    fn jitter_offset_is_deterministic_and_seed_sensitive() {
        assert!(approx_eq(jitter_offset(1234), jitter_offset(1234)));
        assert!((jitter_offset(1234) - jitter_offset(1235)).abs() > CMP_EPS);
    }

    #[test]
    fn jitter_offset_stays_in_the_unit_interval() {
        for seed in 0..256_u32 {
            let j = jitter_offset(seed.wrapping_mul(2_654_435_761));
            assert!((0.0..1.0).contains(&j), "jitter out of range for {seed}");
        }
    }

    #[test]
    fn march_fractions_are_jittered_and_deterministic() {
        let params = base_params(4);
        let a = march_fractions(&params, 11);
        let b = march_fractions(&params, 11);
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
        // Each fraction is the previous plus one normalized step.
        for pair in a.windows(2) {
            assert!(approx_eq(pair[1] - pair[0], 0.25));
        }
        // All positions remain within the unit march span.
        for &f in &a {
            assert!((0.0..1.0).contains(&f));
        }
        // A zero step count yields an empty schedule.
        assert!(march_fractions(&base_params(0), 11).is_empty());
    }

    #[test]
    fn zero_step_count_stays_fully_lit() {
        let params = base_params(0);
        let depths = [9.8_f32; 8];
        assert!(approx_eq(params.occlusion(10.0, 0.0, &depths, 2), 1.0));
    }

    #[test]
    fn non_positive_thickness_disables_the_march() {
        // An empty (bias .. bias) window can never contain a gap.
        let params = ContactShadowParams::new(8, 1.0, 0.0, 0.0, 1.0, 1.0);
        let depths = [9.8_f32; 8];
        assert!(approx_eq(params.occlusion(10.0, 0.0, &depths, 4), 1.0));
        let negative = ContactShadowParams::new(8, 1.0, -0.5, 0.0, 1.0, 1.0);
        assert!(approx_eq(negative.occlusion(10.0, 0.0, &depths, 4), 1.0));
    }

    #[test]
    fn intensity_saturates_at_full_shadow() {
        // A huge intensity clamps the shadow to 1, driving the factor to 0.
        let params = ContactShadowParams::new(8, 1.0, 0.5, 0.0, 100.0, 1.0);
        let depths = [9.8_f32; 8];
        let factor = params.occlusion(10.0, 0.0, &depths, 6);
        assert!((0.0..=1.0).contains(&factor));
        assert!(factor < 0.1);
        // Zero intensity leaves the point fully lit despite a valid hit.
        let unlit = ContactShadowParams::new(8, 1.0, 0.5, 0.0, 0.0, 1.0);
        assert!(approx_eq(unlit.occlusion(10.0, 0.0, &depths, 6), 1.0));
    }

    #[test]
    fn smoothstep_endpoints_and_degenerate_span() {
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
        // A collapsed interval becomes a hard step at the upper edge.
        assert!(approx_eq(smoothstep(2.0, 2.0, 1.9), 0.0));
        assert!(approx_eq(smoothstep(2.0, 2.0, 2.0), 1.0));
    }

    #[test]
    fn std430_layout_and_bytes() {
        assert_eq!(CONTACT_SHADOW_PARAMS_STRIDE, 32);
        let params = ContactShadowParams::new(16, 1.5, 0.5, 0.05, 0.75, 2.0);
        let packed = params.to_std430();
        // The integer step count is stored verbatim.
        assert_eq!(packed[0], 16);
        // The five f32 fields survive as raw bits.
        assert_eq!(packed[1], 1.5f32.to_bits());
        assert_eq!(packed[2], 0.5f32.to_bits());
        assert_eq!(packed[3], 0.05f32.to_bits());
        assert_eq!(packed[4], 0.75f32.to_bits());
        assert_eq!(packed[5], 2.0f32.to_bits());
        // Padding words are zero.
        assert_eq!(packed[6], 0);
        assert_eq!(packed[7], 0);

        assert_eq!(contact_shadow_params_buffer_bytes(3), 96);
        // An empty set still reserves one element.
        assert_eq!(
            contact_shadow_params_buffer_bytes(0),
            CONTACT_SHADOW_PARAMS_STRIDE
        );
    }

    #[test]
    fn dispatch_groups_round_up() {
        assert_eq!(dispatch_groups(0, 64), 0);
        assert_eq!(dispatch_groups(1, 64), 1);
        assert_eq!(dispatch_groups(64, 64), 1);
        assert_eq!(dispatch_groups(65, 64), 2);
        // A zero group size is guarded to one thread per group.
        assert_eq!(dispatch_groups(10, 0), 10);
    }

    #[test]
    fn occlusion_factor_always_in_unit_interval() {
        let params = ContactShadowParams::new(12, 2.0, 0.75, 0.05, 1.5, 0.5);
        let mut start = 5.0;
        while start <= 15.0 {
            let depths = [start - 0.3; 12];
            let factor = params.occlusion(start, 1.0, &depths, start.to_bits());
            assert!((0.0..=1.0).contains(&factor));
            start += 0.5;
        }
    }
}
