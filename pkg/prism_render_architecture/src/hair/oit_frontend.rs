//! Order-independent transparency front-end for hair strand fragments.
//!
//! Hair silhouettes are built from thousands of sub-pixel strands whose
//! semi-transparent edges overlap in depth, so a single opaque depth test
//! cannot composite them correctly: the visible colour depends on resolving the
//! fragments `front-to-back`. GPU hair renderers solve this with a `per-pixel`
//! linked list (`PPLL`) that captures every strand fragment covering a pixel,
//! then resolve the nearest `k-layer` stack analytically with `multi-layer`
//! `alpha` blending (`MLAB`) or a `moment-based` approximation. AMD `TressFX`
//! and UE5 Groom both ship a variant of this `k-layer` resolve.
//!
//! This module owns the **contract + CPU golden** half of that pipeline, not
//! any GPU state. Given the unordered set of hair fragments that a `PPLL`
//! captured for one pixel, it (1) stably sorts them `front-to-back` and clips
//! the stack to `k` resolved layers — flattening the overflow into a single
//! tail term — and (2) evaluates the `MLAB` `k-layer` transmittance and
//! coverage from that layered stack. Every mapping is a deterministic
//! array-in/array-out function (stable ordering, golden-comparable) that
//! sanitizes hostile input and never panics, and the whole thing is
//! transcendental-free: `alpha` compositing is a plain `(1 - alpha)` product,
//! so results are bit-stable across platforms.
//!
//! A per-layer bucketing helper mirrors the virtual-geometry bin pattern: it
//! fans a batch of per-pixel resolves out into one bucket per layer index, so a
//! downstream pass can dispatch layer `i` of every pixel together.

use alloc::vec::Vec;

/// Upper bound on the number of resolved `MLAB` layers kept per pixel.
///
/// A `PPLL` can capture arbitrarily many fragments per pixel, but the analytic
/// resolve only keeps the nearest handful; everything deeper is flattened into
/// the tail term. Eight layers matches the common `TressFX`/`MLAB` tuning and
/// bounds the per-pixel storage. [`sort_and_clip`] and [`bin_fragments_by_layer`]
/// clamp their requested `k` to this ceiling.
pub const MAX_OIT_LAYERS: usize = 8;

/// Depths within this tolerance are treated as coincident, so stable sorting
/// preserves their original capture order rather than reordering ties.
const DEPTH_EPS: f32 = 1.0e-6;

/// One hair fragment captured by the `PPLL` for a single pixel.
///
/// `depth` orders the fragment `front-to-back` (smaller is nearer). `alpha` is
/// the coverage/opacity in `0..=1`. `transmittance_weight` is an optional
/// pre-multiplied colour weight the shading side carries alongside the
/// fragment; it is kept in `0..=1` so a hostile value cannot poison a bucket.
/// [`HairFragment::new`] sanitizes every field, and the resolve paths clamp
/// `alpha` again defensively.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFragment {
    /// `front-to-back` sort key; smaller is nearer the eye.
    pub depth: f32,
    /// Fragment coverage/opacity in `0..=1`.
    pub alpha: f32,
    /// Optional colour/transmittance weight carried for shading, in `0..=1`.
    pub transmittance_weight: f32,
}

impl HairFragment {
    /// Builds a fragment, sanitizing `depth` to a finite value and clamping
    /// `alpha` and `transmittance_weight` to `0..=1`.
    #[must_use]
    pub fn new(depth: f32, alpha: f32, transmittance_weight: f32) -> Self {
        Self {
            depth: sanitize_depth(depth),
            alpha: sanitize_unit(alpha),
            transmittance_weight: sanitize_unit(transmittance_weight),
        }
    }
}

/// A pixel's fragment stack after `front-to-back` sorting and `k-layer` clip.
///
/// `layers` holds up to `k` nearest fragments in resolved order. `tail_alpha`
/// is the flattened opacity of every overflow fragment, accumulated as
/// `1 - product(1 - alpha)` so the deep tail composites like a single blended
/// occluder behind the resolved layers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayeredFragments {
    /// The nearest resolved fragments, `front-to-back`, at most `k` entries.
    pub layers: Vec<HairFragment>,
    /// Flattened opacity of the overflow fragments behind `layers`, in `0..=1`.
    pub tail_alpha: f32,
}

impl LayeredFragments {
    /// Returns `true` when no fragment was resolved and the tail is clear.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() && self.tail_alpha < DEPTH_EPS
    }
}

/// Sorts a pixel's captured fragments `front-to-back` and clips to `k` layers.
///
/// The input is the unordered fragment set a `PPLL` gathered for one pixel.
/// Fragments are sanitized, then **stably** sorted by ascending `depth` so that
/// coincident depths (within [`DEPTH_EPS`]) keep their capture order — this is
/// what makes the golden output deterministic. The nearest `k` fragments become
/// `layers`; everything deeper is flattened into `tail_alpha` via the
/// `1 - product(1 - alpha)` `alpha` composite. `k` is clamped to
/// [`MAX_OIT_LAYERS`]; `k == 0` sends the whole stack into the tail. An empty
/// input yields an empty result.
#[must_use]
pub fn sort_and_clip(fragments: &[HairFragment], k: usize) -> LayeredFragments {
    let k = k.min(MAX_OIT_LAYERS);

    let mut sorted: Vec<HairFragment> = Vec::with_capacity(fragments.len());
    for &fragment in fragments {
        // Re-sanitize so a hand-built struct (bypassing `new`) cannot inject a
        // NaN depth into the sort or an out-of-range `alpha` into the composite.
        sorted.push(HairFragment::new(
            fragment.depth,
            fragment.alpha,
            fragment.transmittance_weight,
        ));
    }

    // `sort_by` is a stable sort; sanitized depths are always finite so
    // `partial_cmp` never returns `None`. Depths within `DEPTH_EPS` compare
    // equal, so stability preserves their input order.
    sorted.sort_by(|a, b| {
        if (a.depth - b.depth).abs() < DEPTH_EPS {
            core::cmp::Ordering::Equal
        } else {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(core::cmp::Ordering::Equal)
        }
    });

    let split = k.min(sorted.len());
    let mut layers: Vec<HairFragment> = Vec::with_capacity(split);
    layers.extend_from_slice(&sorted[..split]);

    // Flatten the overflow into one tail opacity: transmittance of the deep
    // stack is the product of each `(1 - alpha)`, so the merged `alpha` is
    // `1 - that product`. Empty overflow leaves the tail fully clear.
    let mut tail_transmittance = 1.0_f32;
    for fragment in &sorted[split..] {
        tail_transmittance *= 1.0 - fragment.alpha;
    }
    let tail_alpha = (1.0 - tail_transmittance).clamp(0.0, 1.0);

    LayeredFragments { layers, tail_alpha }
}

/// `front-to-back` transmittance of a resolved pixel stack, in `0..=1`.
///
/// Transmittance is the fraction of background light that survives the stack:
/// the `alpha` composite `product(1 - alpha_i)` over the resolved layers, times
/// the tail's own `(1 - tail_alpha)`. No layers and a clear tail give full
/// transmittance (`1`); a fully opaque layer drives it to `0`.
#[must_use]
pub fn composite_transmittance(layered: &LayeredFragments) -> f32 {
    let mut transmittance = 1.0_f32;
    for fragment in &layered.layers {
        transmittance *= 1.0 - fragment.alpha;
    }
    transmittance *= 1.0 - layered.tail_alpha;
    transmittance.clamp(0.0, 1.0)
}

/// Accumulated coverage of a resolved pixel stack, `1 - transmittance`.
///
/// Coverage is the complement of [`composite_transmittance`]: the fraction of
/// the pixel the hair stack occludes. It stays in `0..=1`.
#[must_use]
pub fn composite_coverage(layered: &LayeredFragments) -> f32 {
    (1.0 - composite_transmittance(layered)).clamp(0.0, 1.0)
}

/// A batch of per-pixel resolves fanned out into one bucket per layer index.
///
/// `per_layer[i]` holds the `i`-th resolved fragment of every pixel that
/// reached that depth, in input-pixel order. A downstream pass can then
/// dispatch all pixels' layer `i` together instead of walking each pixel's
/// stack independently, mirroring the virtual-geometry bin pattern.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OitLayerBins {
    per_layer: Vec<Vec<HairFragment>>,
}

impl OitLayerBins {
    /// Total number of fragments across every layer bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.per_layer.iter().map(Vec::len).sum()
    }

    /// Returns `true` when no fragment landed in any layer bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_layer.iter().all(Vec::is_empty)
    }

    /// Read-only view of a given layer's bucket, or `None` when out of range.
    #[must_use]
    pub fn layer(&self, index: usize) -> Option<&[HairFragment]> {
        self.per_layer.get(index).map(Vec::as_slice)
    }

    /// Number of layer buckets (the effective `k`).
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.per_layer.len()
    }
}

/// Fans a batch of resolved pixels out into `k` per-layer buckets.
///
/// For each pixel's [`LayeredFragments`], its `i`-th layer is pushed into bucket
/// `i`, preserving the input-pixel order within each bucket so the fan-out stays
/// deterministic. A pixel shallower than `k` layers simply contributes to fewer
/// buckets. `k` is clamped to [`MAX_OIT_LAYERS`]; `k == 0` yields no buckets.
#[must_use]
pub fn bin_fragments_by_layer(per_pixel: &[LayeredFragments], k: usize) -> OitLayerBins {
    let k = k.min(MAX_OIT_LAYERS);
    let mut per_layer: Vec<Vec<HairFragment>> = Vec::with_capacity(k);
    for _ in 0..k {
        per_layer.push(Vec::new());
    }

    for pixel in per_pixel {
        for (index, &fragment) in pixel.layers.iter().enumerate() {
            let Some(bucket) = per_layer.get_mut(index) else {
                // Pixel carries more layers than requested buckets; the deeper
                // layers are not fanned out at this `k`.
                break;
            };
            bucket.push(fragment);
        }
    }

    OitLayerBins { per_layer }
}

/// Replaces a non-finite depth with `0.0` so the sort key is always orderable.
fn sanitize_depth(depth: f32) -> f32 {
    if depth.is_finite() {
        depth
    } else {
        0.0
    }
}

/// Clamps a value to `0..=1`, mapping any non-finite input to `0.0`.
fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn sorts_fragments_front_to_back() {
        let fragments = [
            HairFragment::new(0.9, 0.2, 1.0),
            HairFragment::new(0.1, 0.3, 1.0),
            HairFragment::new(0.5, 0.4, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        let depths: Vec<f32> = layered.layers.iter().map(|f| f.depth).collect();
        assert!(close(depths[0], 0.1));
        assert!(close(depths[1], 0.5));
        assert!(close(depths[2], 0.9));
    }

    #[test]
    fn stable_order_preserved_for_equal_depths() {
        // Three coincident depths with distinct alphas must keep input order.
        let fragments = [
            HairFragment::new(0.5, 0.1, 1.0),
            HairFragment::new(0.5, 0.2, 1.0),
            HairFragment::new(0.5, 0.3, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        assert!(close(layered.layers[0].alpha, 0.1));
        assert!(close(layered.layers[1].alpha, 0.2));
        assert!(close(layered.layers[2].alpha, 0.3));
    }

    #[test]
    fn clips_to_k_layers_and_flattens_overflow_into_tail() {
        let fragments = [
            HairFragment::new(0.1, 0.5, 1.0),
            HairFragment::new(0.2, 0.5, 1.0),
            HairFragment::new(0.3, 0.5, 1.0),
            HairFragment::new(0.4, 0.5, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 2);
        assert_eq!(layered.layers.len(), 2);
        assert!(close(layered.layers[0].depth, 0.1));
        assert!(close(layered.layers[1].depth, 0.2));
        // Overflow: two alpha=0.5 fragments -> 1 - 0.5*0.5 = 0.75.
        assert!(close(layered.tail_alpha, 0.75));
    }

    #[test]
    fn transmittance_is_alpha_product() {
        let fragments = [
            HairFragment::new(0.1, 0.5, 1.0),
            HairFragment::new(0.2, 0.5, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        // (1-0.5)*(1-0.5) = 0.25, no tail.
        assert!(close(composite_transmittance(&layered), 0.25));
    }

    #[test]
    fn fully_opaque_stack_has_zero_transmittance() {
        let fragments = [
            HairFragment::new(0.1, 1.0, 1.0),
            HairFragment::new(0.2, 0.4, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        assert!(close(composite_transmittance(&layered), 0.0));
    }

    #[test]
    fn fully_transparent_stack_has_full_transmittance() {
        let fragments = [
            HairFragment::new(0.1, 0.0, 1.0),
            HairFragment::new(0.2, 0.0, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        assert!(close(composite_transmittance(&layered), 1.0));
    }

    #[test]
    fn coverage_is_one_minus_transmittance() {
        let fragments = [
            HairFragment::new(0.1, 0.5, 1.0),
            HairFragment::new(0.2, 0.5, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 8);
        let trans = composite_transmittance(&layered);
        assert!(close(composite_coverage(&layered), 1.0 - trans));
        assert!(close(composite_coverage(&layered), 0.75));
    }

    #[test]
    fn empty_input_yields_empty_result() {
        let layered = sort_and_clip(&[], 8);
        assert!(layered.is_empty());
        assert!(layered.layers.is_empty());
        assert!(close(layered.tail_alpha, 0.0));
        assert!(close(composite_transmittance(&layered), 1.0));
        assert!(close(composite_coverage(&layered), 0.0));
    }

    #[test]
    fn zero_k_sends_everything_to_tail() {
        let fragments = [
            HairFragment::new(0.1, 0.5, 1.0),
            HairFragment::new(0.2, 0.5, 1.0),
        ];
        let layered = sort_and_clip(&fragments, 0);
        assert!(layered.layers.is_empty());
        // 1 - 0.5*0.5 = 0.75.
        assert!(close(layered.tail_alpha, 0.75));
        assert!(close(composite_transmittance(&layered), 0.25));
    }

    #[test]
    fn k_is_clamped_to_max_layers() {
        let fragments: Vec<HairFragment> = (0..MAX_OIT_LAYERS + 4)
            .map(|i| HairFragment::new(i as f32, 0.1, 1.0))
            .collect();
        // Request far more layers than allowed; only MAX_OIT_LAYERS resolve.
        let layered = sort_and_clip(&fragments, 999);
        assert_eq!(layered.layers.len(), MAX_OIT_LAYERS);
    }

    #[test]
    fn bad_alpha_and_nan_depth_are_sanitized() {
        // depth NaN -> 0.0 (sorts to front); alpha > 1 -> clamped to 1.0.
        let hostile = HairFragment {
            depth: f32::NAN,
            alpha: 5.0,
            transmittance_weight: f32::INFINITY,
        };
        let good = HairFragment::new(0.5, 0.2, 1.0);
        let layered = sort_and_clip(&[good, hostile], 8);
        // Sanitized NaN depth became 0.0, so the hostile fragment sorts first.
        assert!(close(layered.layers[0].depth, 0.0));
        assert!(close(layered.layers[0].alpha, 1.0));
        assert!(close(layered.layers[0].transmittance_weight, 0.0));
        // A clamped alpha=1 opaque fragment zeroes transmittance.
        assert!(close(composite_transmittance(&layered), 0.0));
    }

    #[test]
    fn negative_alpha_is_clamped_to_zero() {
        let fragments = [HairFragment::new(0.1, -3.0, 1.0)];
        let layered = sort_and_clip(&fragments, 8);
        assert!(close(layered.layers[0].alpha, 0.0));
        assert!(close(composite_transmittance(&layered), 1.0));
    }

    #[test]
    fn bins_preserve_pixel_order_within_a_layer() {
        let pixel_a = sort_and_clip(
            &[
                HairFragment::new(0.1, 0.1, 1.0),
                HairFragment::new(0.2, 0.2, 1.0),
            ],
            4,
        );
        let pixel_b = sort_and_clip(
            &[
                HairFragment::new(0.1, 0.3, 1.0),
                HairFragment::new(0.2, 0.4, 1.0),
            ],
            4,
        );
        let bins = bin_fragments_by_layer(&[pixel_a, pixel_b], 4);
        // Layer 0 holds the nearest fragment of pixel A then pixel B.
        let layer0 = bins.layer(0).unwrap();
        assert!(close(layer0[0].alpha, 0.1));
        assert!(close(layer0[1].alpha, 0.3));
        // Layer 1 holds each pixel's second fragment, same order.
        let layer1 = bins.layer(1).unwrap();
        assert!(close(layer1[0].alpha, 0.2));
        assert!(close(layer1[1].alpha, 0.4));
    }

    #[test]
    fn bins_total_counts_all_fanned_fragments() {
        let pixel_a = sort_and_clip(
            &[
                HairFragment::new(0.1, 0.1, 1.0),
                HairFragment::new(0.2, 0.2, 1.0),
            ],
            4,
        );
        let pixel_b = sort_and_clip(&[HairFragment::new(0.1, 0.3, 1.0)], 4);
        let bins = bin_fragments_by_layer(&[pixel_a, pixel_b], 4);
        // 2 + 1 = 3 fragments spread across the layer buckets.
        assert_eq!(bins.total(), 3);
        assert!(!bins.is_empty());
        assert_eq!(bins.layer_count(), 4);
    }

    #[test]
    fn empty_bins_are_empty() {
        let bins = bin_fragments_by_layer(&[], 4);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
        let zero = bin_fragments_by_layer(&[sort_and_clip(&[], 4)], 0);
        assert!(zero.is_empty());
        assert_eq!(zero.layer_count(), 0);
    }
}
