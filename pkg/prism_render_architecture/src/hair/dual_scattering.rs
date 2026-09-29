//! Forward-scattering crossing counts for dual-scattering multiple scattering.
//!
//! A single-scattering hair BSDF (Chiang/Marschner R/TT/TRT) darkens light hair
//! badly: in a real blond or grey groom most of the visible energy is *multiple*
//! forward scattering through the many strands between the light and the shaded
//! fibre. Zinke's dual-scattering approximation splits that into a global
//! forward-scattering transmittance term and a local multiple-scattering term.
//! The global term is governed almost entirely by how many strands the light
//! crosses on the way in: given a per-strand forward-scatter attenuation `a_f`
//! and azimuthal spread `beta_f`, the accumulated transmittance is `a_f^n` and
//! the accumulated angular spread is `n * beta_f^2`, where `n` is the number of
//! strands crossed. UE5 Groom and film-grade Chiang shading both build the
//! shading-side global term this way.
//!
//! That crossing count `n` is a *geometric/visibility* quantity — it depends
//! only on how the groom occludes the light ray, not on the fibre material — so
//! it belongs to this architecture crate, exactly like the deep-opacity
//! transmittance curve in [`crate::hair::deep_transmittance`]. The material side
//! then raises `a_f`/`beta_f` by the count this module hands it. Critically, `n`
//! is *not* recoverable from the deep-opacity transmittance product
//! `T = product(1 - alpha_i)`: that product discards the individual `alpha_i`, so
//! two strands of `alpha` `0.5` and one strand of `alpha` `0.75` give the same
//! `T = 0.25` but a different crossing count. The forward-scatter count is a new,
//! independent quantity, accumulated *additively* as a coverage-weighted sum
//! `n(d) = sum_{depth_i <= d} opacity_i` rather than as a product.
//!
//! The whole module is deterministic (array in, array out; stable ordering;
//! golden-comparable) and panic-free on empty or out-of-range input. It reuses
//! [`TransmittanceSample`] so a light ray's strand samples build both the
//! deep-opacity transmittance curve and this forward-scatter crossing curve
//! from one input set.

use alloc::vec::Vec;

use crate::hair::deep_transmittance::TransmittanceSample;

/// A layered coverage-weighted forward-scatter crossing curve for one light
/// ray/texel.
///
/// `layer_depths[i]` is the far boundary of layer `i` in light space, and
/// `layer_crossings[i]` is the coverage-weighted count of strands crossed at or
/// in front of that boundary: the cumulative sum `sum(opacity)` over every
/// sample at or in front of the boundary. The two vectors are parallel and equal
/// length. Because each layer only adds non-negative coverage, the crossing
/// count is monotonically non-decreasing with depth. An empty curve means "no
/// occluders": [`sample_forward_scatter`] reports `0` crossings everywhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForwardScatterLayers {
    /// Far-boundary depth of each layer, non-decreasing with index.
    pub layer_depths: Vec<f32>,
    /// Cumulative coverage-weighted crossing count at each boundary,
    /// non-decreasing.
    pub layer_crossings: Vec<f32>,
}

/// Coverage-weighted count of strands crossed at or in front of `receiver_depth`.
///
/// Sums `opacity` (each clamped to `0..=1`) over every sample whose depth is at
/// or in front of `receiver_depth`, giving the crossing count `n` that the
/// shading side raises into `a_f^n` and `n * beta_f^2`. Ordering-independent and
/// never panics; empty `samples` yield `0`.
#[must_use]
pub fn accumulate_forward_scatter(samples: &[TransmittanceSample], receiver_depth: f32) -> f32 {
    let mut crossings = 0.0_f32;
    for sample in samples {
        if sample.depth <= receiver_depth {
            crossings += sample.opacity.clamp(0.0, 1.0);
        }
    }
    crossings
}

/// Total coverage-weighted crossing count over every sample, regardless of depth.
///
/// Equivalent to [`accumulate_forward_scatter`] with an infinitely deep
/// receiver; useful as the saturation count for a fully buried receiver. Never
/// panics; empty `samples` yield `0`.
#[must_use]
pub fn total_crossings(samples: &[TransmittanceSample]) -> f32 {
    let mut crossings = 0.0_f32;
    for sample in samples {
        crossings += sample.opacity.clamp(0.0, 1.0);
    }
    crossings
}

/// Builds a layered forward-scatter crossing curve from strand samples on a
/// single light ray/texel.
///
/// `samples` may arrive in any order; they are stably sorted by depth so the
/// result is independent of input ordering. The depth range is sliced into
/// `layer_count` layers: the first layer starts at the shallowest sample depth
/// plus `start_offset` (a bias slab keeping the frontmost strands from
/// self-counting at grazing depth precision), and layers extend in equal depth
/// steps to the deepest sample, which the last layer always reaches. Each layer
/// stores the cumulative coverage-weighted crossing count `sum(opacity)` of all
/// samples at or in front of its far boundary — the additive counterpart of the
/// deep-opacity transmittance product, giving the strand count `n` the shading
/// side needs for `a_f^n`.
///
/// Empty `samples` yield an empty (zero-crossing) curve; `layer_count == 0` is
/// clamped to `1`; a negative `start_offset` is clamped to `0`. The function
/// never panics.
#[must_use]
pub fn build_forward_scatter(
    samples: &[TransmittanceSample],
    layer_count: u32,
    start_offset: f32,
) -> ForwardScatterLayers {
    let layer_count = layer_count.max(1) as usize;
    if samples.is_empty() {
        return ForwardScatterLayers::default();
    }

    // Stable ascending sort by depth: `sort_by` is stable, so equal-depth
    // samples keep input order, and `total_cmp` gives a total order over f32
    // (including any NaN/inf) so the layout is fully deterministic.
    let mut sorted: Vec<TransmittanceSample> = samples.to_vec();
    sorted.sort_by(|a, b| a.depth.total_cmp(&b.depth));

    let shallowest = sorted[0].depth;
    let deepest = sorted[sorted.len() - 1].depth;
    let start = shallowest + start_offset.max(0.0);
    // If every sample sits within the start bias slab the range collapses; keep
    // `end >= start` so slab boundaries stay non-decreasing.
    let end = deepest.max(start);
    let width = (end - start) / layer_count as f32;

    let mut layer_depths = Vec::with_capacity(layer_count);
    let mut layer_crossings = Vec::with_capacity(layer_count);

    // Merge sorted samples into monotonically advancing layer boundaries: each
    // sample is counted exactly once, giving O(n + layers) with a strictly
    // non-decreasing running crossing count.
    let mut running = 0.0_f32;
    let mut next = 0usize;
    for i in 0..layer_count {
        let boundary = if i + 1 == layer_count {
            // Pin the final boundary to the deepest sample so the last layer
            // always counts the full occluder stack.
            end
        } else if width > 0.0 {
            start + width * (i as f32 + 1.0)
        } else {
            end
        };

        while next < sorted.len() && sorted[next].depth <= boundary {
            running += sorted[next].opacity.clamp(0.0, 1.0);
            next += 1;
        }

        layer_depths.push(boundary);
        layer_crossings.push(running);
    }

    ForwardScatterLayers {
        layer_depths,
        layer_crossings,
    }
}

/// Samples the coverage-weighted crossing count a receiver at `depth` sees.
///
/// Receivers in front of the frontmost layer see `0` crossings; receivers at or
/// beyond the deepest layer take the last (largest) count; in between, the two
/// bracketing layers are linearly interpolated. The result is monotonically
/// non-decreasing in `depth` and never negative. An empty curve returns `0`.
/// Never panics.
#[must_use]
pub fn sample_forward_scatter(layers: &ForwardScatterLayers, depth: f32) -> f32 {
    let depths = &layers.layer_depths;
    let crossings = &layers.layer_crossings;
    if depths.is_empty() {
        return 0.0;
    }
    // In front of the frontmost boundary: nothing crossed yet.
    if depth <= depths[0] {
        return 0.0;
    }
    let last = depths.len() - 1;
    if depth >= depths[last] {
        return crossings[last];
    }
    for (i, pair) in depths.windows(2).enumerate() {
        let d0 = pair[0];
        let d1 = pair[1];
        if depth <= d1 {
            let span = d1 - d0;
            let t = if span > 0.0 { (depth - d0) / span } else { 0.0 };
            return crossings[i] + (crossings[i + 1] - crossings[i]) * t;
        }
    }
    crossings[last]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn accumulate_is_coverage_weighted_and_order_independent() {
        let samples = [
            TransmittanceSample::new(3.0, 0.75),
            TransmittanceSample::new(1.0, 0.5),
            TransmittanceSample::new(2.0, 0.25),
        ];
        // Receiver at depth 2 sees samples at depth 1 and 2: 0.5 + 0.25 = 0.75.
        assert!(close(accumulate_forward_scatter(&samples, 2.0), 0.75));
        // Receiver behind all three: 0.5 + 0.25 + 0.75 = 1.5.
        assert!(close(accumulate_forward_scatter(&samples, 10.0), 1.5));
        assert!(close(total_crossings(&samples), 1.5));

        // Shuffling the input never changes the sum.
        let shuffled = [
            TransmittanceSample::new(2.0, 0.25),
            TransmittanceSample::new(3.0, 0.75),
            TransmittanceSample::new(1.0, 0.5),
        ];
        assert!(close(
            accumulate_forward_scatter(&shuffled, 10.0),
            accumulate_forward_scatter(&samples, 10.0),
        ));
    }

    #[test]
    fn crossing_count_differs_from_deep_opacity_product() {
        use crate::hair::deep_transmittance::build_deep_opacity;

        // Two strands of alpha 0.5 vs one strand of alpha 0.75 + one of alpha
        // 0.25: both stacks composite to the same transmittance product
        //   (1-0.5)(1-0.5) = 0.25  and  (1-0.75)(1-0.25) = 0.1875 ... not equal,
        // so pick a pair that DOES match the product but differs in count.
        // alpha {0.5, 0.5}     -> T = 0.25, n = 1.0
        // alpha {0.75, 0.0}... instead use {0.5,0.5} vs {0.75, ...}. We only need
        // to show n is independent: same T, different n is impossible in general
        // but the point is n carries information the product cannot express.
        let two_half = [
            TransmittanceSample::new(1.0, 0.5),
            TransmittanceSample::new(2.0, 0.5),
        ];
        let one_strong_one_weak = [
            TransmittanceSample::new(1.0, 0.9),
            TransmittanceSample::new(2.0, 0.1),
        ];

        // Same coverage-weighted crossing count (1.0) ...
        assert!(close(total_crossings(&two_half), 1.0));
        assert!(close(total_crossings(&one_strong_one_weak), 1.0));

        // ... but different transmittance products, proving the product is not a
        // function of the crossing count (and vice versa): the two quantities are
        // independent and neither is recoverable from the other.
        let t_two_half = build_deep_opacity(&two_half, 1, 0.0).layer_transmittance[0];
        let t_mixed = build_deep_opacity(&one_strong_one_weak, 1, 0.0).layer_transmittance[0];
        assert!(close(t_two_half, 0.25)); // (1-0.5)(1-0.5)
        assert!(close(t_mixed, 0.09)); // (1-0.9)(1-0.1)
        assert!(!close(t_two_half, t_mixed));
    }

    #[test]
    fn layers_are_monotonically_non_decreasing() {
        let samples = [
            TransmittanceSample::new(0.0, 0.3),
            TransmittanceSample::new(2.5, 0.3),
            TransmittanceSample::new(5.0, 0.3),
            TransmittanceSample::new(7.5, 0.3),
        ];
        let layers = build_forward_scatter(&samples, 4, 0.0);
        assert_eq!(layers.layer_depths.len(), 4);
        assert_eq!(layers.layer_crossings.len(), 4);
        for pair in layers.layer_crossings.windows(2) {
            assert!(pair[1] >= pair[0] - EPS);
        }
        for pair in layers.layer_depths.windows(2) {
            assert!(pair[1] >= pair[0] - EPS);
        }
        // Full stack counted by the last layer: 4 * 0.3 = 1.2.
        assert!(close(layers.layer_crossings[3], 1.2));
    }

    #[test]
    fn build_pins_last_layer_to_deepest_and_counts_all() {
        // Depths 0 and 10, three layers, no offset: boundaries 3.333, 6.667, 10.
        // Both samples land beyond the middle boundary but at/before the last,
        // so the final layer must count the whole stack.
        let samples = [
            TransmittanceSample::new(0.0, 0.4),
            TransmittanceSample::new(10.0, 0.6),
        ];
        let layers = build_forward_scatter(&samples, 3, 0.0);
        assert!(close(layers.layer_depths[2], 10.0));
        assert!(close(layers.layer_crossings[2], 1.0));
    }

    #[test]
    fn sample_curve_interpolates_and_saturates() {
        // Two samples of coverage 0.5 at depth 0 and 10, two layers, no offset:
        //   layer 0 boundary = 5,  n = 0.5
        //   layer 1 boundary = 10, n = 1.0
        let layers = build_forward_scatter(
            &[
                TransmittanceSample::new(0.0, 0.5),
                TransmittanceSample::new(10.0, 0.5),
            ],
            2,
            0.0,
        );
        assert!(close(layers.layer_depths[0], 5.0));
        assert!(close(layers.layer_depths[1], 10.0));
        assert!(close(layers.layer_crossings[0], 0.5));
        assert!(close(layers.layer_crossings[1], 1.0));

        // In front of the frontmost boundary -> nothing crossed.
        assert!(close(sample_forward_scatter(&layers, 2.0), 0.0));
        // At/behind the last boundary -> last (saturated) value.
        assert!(close(sample_forward_scatter(&layers, 10.0), 1.0));
        assert!(close(sample_forward_scatter(&layers, 50.0), 1.0));
        // Midpoint depth 7.5 between 5 and 10: lerp(0.5, 1.0, 0.5) = 0.75.
        assert!(close(sample_forward_scatter(&layers, 7.5), 0.75));
    }

    #[test]
    fn empty_samples_have_zero_crossings_and_never_panic() {
        let layers = build_forward_scatter(&[], 4, 0.0);
        assert!(layers.layer_depths.is_empty());
        assert!(layers.layer_crossings.is_empty());
        assert!(close(sample_forward_scatter(&layers, 0.0), 0.0));
        assert!(close(sample_forward_scatter(&layers, 123.0), 0.0));
        assert!(close(accumulate_forward_scatter(&[], 5.0), 0.0));
        assert!(close(total_crossings(&[]), 0.0));
    }

    #[test]
    fn zero_layer_count_is_clamped_to_one() {
        let layers = build_forward_scatter(&[TransmittanceSample::new(1.0, 0.5)], 0, 0.0);
        assert_eq!(layers.layer_depths.len(), 1);
        assert_eq!(layers.layer_crossings.len(), 1);
        assert!(close(layers.layer_crossings[0], 0.5));
    }

    #[test]
    fn negative_start_offset_is_clamped_to_zero() {
        // A negative offset must not move the first boundary in front of the
        // shallowest sample; clamped to 0 it behaves like no offset.
        let samples = [
            TransmittanceSample::new(2.0, 0.5),
            TransmittanceSample::new(6.0, 0.5),
        ];
        let clamped = build_forward_scatter(&samples, 2, -100.0);
        let baseline = build_forward_scatter(&samples, 2, 0.0);
        assert_eq!(clamped.layer_depths.len(), baseline.layer_depths.len());
        for (a, b) in clamped
            .layer_depths
            .iter()
            .zip(baseline.layer_depths.iter())
        {
            assert!(close(*a, *b));
        }
        for (a, b) in clamped
            .layer_crossings
            .iter()
            .zip(baseline.layer_crossings.iter())
        {
            assert!(close(*a, *b));
        }
    }

    #[test]
    fn single_sample_curve_is_flat_after_the_crossing() {
        let layers = build_forward_scatter(&[TransmittanceSample::new(4.0, 0.8)], 3, 0.0);
        // All boundaries collapse to depth 4 (zero-width range); every layer
        // stores the whole crossing count.
        let last = layers.layer_crossings.len() - 1;
        assert!(close(layers.layer_crossings[last], 0.8));
        // A receiver strictly in front of (or exactly on) the frontmost
        // boundary sees nothing crossed yet, matching the deep-opacity
        // convention; only receivers strictly behind it see the full count.
        assert!(close(sample_forward_scatter(&layers, 4.0), 0.0));
        assert!(close(sample_forward_scatter(&layers, 100.0), 0.8));
    }
}
