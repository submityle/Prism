//! Strand self-shadow transmittance accumulation (deep opacity / voxel).
//!
//! Hair self-shadowing is dominated by strand-on-strand occlusion: thousands of
//! thin strands overlap along the light direction, so a single shadow-map depth
//! test cannot answer "how much light survives to this receiver". Production
//! engines instead accumulate strand *opacity* along the light ray and expose a
//! layered transmittance curve — AMD `TressFX`'s approximate self-shadow and
//! UE5 Groom's *deep opacity map* both work this way. This module owns that
//! shared transmittance service: it turns a set of strand samples on one light
//! texel/ray into a monotone transmittance curve, which the shading side queries
//! by receiver depth and feeds into the shared shadow / OIT path
//! (`crate::transparency`'s `HairVisibility` route).
//!
//! Two complementary approximations are provided, both fully deterministic
//! (array in, array out; stable ordering; golden-comparable) and panic-free on
//! empty or out-of-range input:
//!
//! - **Deep opacity map** — sort samples by light-space depth, slice the depth
//!   range into layers, and accumulate the classic Beer/`alpha`-compositing
//!   product `T = product(1 - alpha_i)`. This is the recommended path: it is the
//!   exact optical transmittance of a stack of independent `alpha`-blended
//!   occluders and needs no exponential.
//! - **Voxel density** — bin sample opacity into a uniform slab of voxels and
//!   read transmittance as the same `alpha` composite `T = product(1 - sigma_j)`
//!   over the traversed voxels. This trades the depth-sorted layer layout for a
//!   fixed-resolution grid, which is convenient when many rays share one froxel
//!   volume, while staying exp-free for bit-exact determinism.
//!
//! Neither path reimplements the shared shadow or transparency subsystems; it
//! only produces the transmittance data those subsystems sample.

use alloc::vec::Vec;

/// One strand sample projected into the light's view: its light-space depth and
/// the opacity it contributes at that depth.
///
/// `opacity` is an `alpha` coverage in `0..=1` (`0` = fully transparent, `1` =
/// fully opaque); `depth` is a non-negative distance along the light ray.
/// [`TransmittanceSample::new`] clamps both invariants, and the accumulation
/// paths clamp opacity again defensively so a hand-built sample with a stray
/// out-of-range value can never drive transmittance outside `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmittanceSample {
    /// Light-space depth of the sample; larger is farther from the light.
    pub depth: f32,
    /// Opacity contribution in `0..=1`.
    pub opacity: f32,
}

impl TransmittanceSample {
    /// Builds a sample, clamping `depth` to `>= 0` and `opacity` to `0..=1`.
    #[must_use]
    pub fn new(depth: f32, opacity: f32) -> Self {
        Self {
            depth: depth.max(0.0),
            opacity: opacity.clamp(0.0, 1.0),
        }
    }
}

/// A layered deep opacity map for one light ray/texel.
///
/// `layer_depths[i]` is the far boundary of layer `i` in light space, and
/// `layer_transmittance[i]` is the fraction of light that survives to that
/// depth: the cumulative product `product(1 - alpha)` over every sample at or
/// in front of the boundary. The two vectors are parallel and equal length.
/// Because each layer multiplies in more factors in `0..=1`, transmittance is
/// monotonically non-increasing with depth. An empty map means "no occluders":
/// [`sample_transmittance`] reports fully transmissive (`1.0`) everywhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepOpacityLayers {
    /// Far-boundary depth of each layer, non-decreasing with index.
    pub layer_depths: Vec<f32>,
    /// Cumulative transmittance at each layer boundary, non-increasing.
    pub layer_transmittance: Vec<f32>,
}

/// Builds a deep opacity map from strand samples on a single light ray/texel.
///
/// `samples` may arrive in any order; they are stably sorted by depth so the
/// result is independent of input ordering. The depth range is sliced into
/// `layer_count` layers: the first layer starts at the shallowest sample depth
/// plus `start_offset` (a bias slab that keeps the frontmost strands from
/// self-shadowing at grazing depth precision), and layers extend in equal
/// depth steps to the deepest sample, which the last layer always reaches. Each
/// layer stores the cumulative transmittance `product(1 - alpha_i)` of all
/// samples at or in front of its far boundary — the classic deep opacity map
/// composite, which is the exact transmittance of a stack of independent
/// `alpha`-blended occluders.
///
/// Empty `samples` yield an empty (fully transmissive) map; `layer_count == 0`
/// is clamped to `1`; a negative `start_offset` is clamped to `0`. The function
/// never panics.
#[must_use]
pub fn build_deep_opacity(
    samples: &[TransmittanceSample],
    layer_count: u32,
    start_offset: f32,
) -> DeepOpacityLayers {
    let layer_count = layer_count.max(1) as usize;
    if samples.is_empty() {
        return DeepOpacityLayers::default();
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
    let mut layer_transmittance = Vec::with_capacity(layer_count);

    // Merge sorted samples into monotonically advancing layer boundaries: each
    // sample is composited exactly once, giving O(n + layers) with a strictly
    // non-increasing running transmittance.
    let mut running = 1.0_f32;
    let mut next = 0usize;
    for i in 0..layer_count {
        let boundary = if i + 1 == layer_count {
            // Pin the final boundary to the deepest sample so the last layer
            // always accumulates the full occluder stack.
            end
        } else if width > 0.0 {
            start + width * (i as f32 + 1.0)
        } else {
            end
        };

        while next < sorted.len() && sorted[next].depth <= boundary {
            running *= 1.0 - sorted[next].opacity.clamp(0.0, 1.0);
            next += 1;
        }

        layer_depths.push(boundary);
        layer_transmittance.push(running);
    }

    DeepOpacityLayers {
        layer_depths,
        layer_transmittance,
    }
}

/// Samples the transmittance a receiver at `depth` sees through the map.
///
/// Receivers in front of the frontmost layer are fully lit (`1.0`); receivers
/// at or beyond the deepest layer take the last (most occluded) value; in
/// between, the two bracketing layers are linearly interpolated. The result is
/// monotonically non-increasing in `depth` and always lies in `0..=1`. An empty
/// map returns `1.0`. Never panics.
#[must_use]
pub fn sample_transmittance(layers: &DeepOpacityLayers, depth: f32) -> f32 {
    let depths = &layers.layer_depths;
    let trans = &layers.layer_transmittance;
    if depths.is_empty() {
        return 1.0;
    }
    // In front of the frontmost boundary: unoccluded.
    if depth <= depths[0] {
        return 1.0;
    }
    let last = depths.len() - 1;
    if depth >= depths[last] {
        return trans[last];
    }
    for (i, pair) in depths.windows(2).enumerate() {
        let d0 = pair[0];
        let d1 = pair[1];
        if depth <= d1 {
            let span = d1 - d0;
            let t = if span > 0.0 { (depth - d0) / span } else { 0.0 };
            return trans[i] + (trans[i + 1] - trans[i]) * t;
        }
    }
    trans[last]
}

/// Accumulates strand opacity into a uniform slab of voxels along the light ray.
///
/// The light-space range `slab_start..slab_end` is divided into `voxel_count`
/// equal voxels; each sample's opacity is added into the voxel its depth falls
/// in, forming a per-voxel optical density `sigma`. Samples outside the slab
/// (depth `< slab_start` or `>= slab_end`) are skipped rather than clamped, so a
/// froxel volume never absorbs occluders that belong to a different slab.
/// `voxel_count == 0` is clamped to `1`; a degenerate slab (`slab_end <=
/// slab_start`) skips every sample and returns zeros. Never panics.
#[must_use]
pub fn accumulate_voxel_density(
    samples: &[TransmittanceSample],
    slab_start: f32,
    slab_end: f32,
    voxel_count: u32,
) -> Vec<f32> {
    let voxel_count = voxel_count.max(1) as usize;
    let mut densities = Vec::with_capacity(voxel_count);
    densities.resize(voxel_count, 0.0_f32);

    let span = slab_end - slab_start;
    if span <= 0.0 {
        return densities;
    }
    let width = span / voxel_count as f32;
    for sample in samples {
        if sample.depth < slab_start || sample.depth >= slab_end {
            continue;
        }
        let raw = ((sample.depth - slab_start) / width) as usize;
        let index = raw.min(voxel_count - 1);
        densities[index] += sample.opacity.clamp(0.0, 1.0);
    }
    densities
}

/// Reads voxel transmittance up to and including voxel `index`.
///
/// Composites the accumulated per-voxel opacity as `T = product(1 - sigma_j)`
/// over voxels `0..=index` (each `sigma_j` clamped to `0..=1`), which is
/// monotonically non-increasing in `index` and lies in `0..=1`. `index` past
/// the last voxel is clamped to the last voxel (full slab); an empty density
/// slab returns `1.0`. Never panics.
#[must_use]
pub fn voxel_transmittance(densities: &[f32], index: usize) -> f32 {
    if densities.is_empty() {
        return 1.0;
    }
    let end = index.min(densities.len() - 1);
    let mut transmittance = 1.0_f32;
    for &sigma in &densities[..=end] {
        transmittance *= 1.0 - sigma.clamp(0.0, 1.0);
    }
    transmittance
}

/// Per-texel strand sample: which light texel/ray a [`TransmittanceSample`]
/// belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexelSample {
    /// Index of the light texel/ray this sample lands on.
    pub texel: u32,
    /// The strand sample contributed to that texel.
    pub sample: TransmittanceSample,
}

/// Strand samples partitioned into one bucket per light texel/ray.
///
/// This follows the per-bucket-`Vec` binning pattern used across the geometry
/// pipeline: a fixed-width array of buckets, deterministic push order, and a
/// pure [`bin_samples`] pass that fans indexed samples out in a single loop.
/// Each bucket can then be handed to [`build_deep_opacity`] or
/// [`accumulate_voxel_density`] independently.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransmittanceBins {
    buckets: Vec<Vec<TransmittanceSample>>,
}

impl TransmittanceBins {
    /// Creates `texel_count` empty buckets (clamped to at least `1`).
    #[must_use]
    pub fn new(texel_count: u32) -> Self {
        let texel_count = texel_count.max(1) as usize;
        let mut buckets = Vec::with_capacity(texel_count);
        buckets.resize_with(texel_count, Vec::new);
        Self { buckets }
    }

    /// Number of texel buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Total number of samples across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    /// Returns `true` when no sample landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(Vec::is_empty)
    }

    /// Borrows the samples routed to `texel`, or `None` if out of range.
    #[must_use]
    pub fn bucket(&self, texel: u32) -> Option<&[TransmittanceSample]> {
        self.buckets.get(texel as usize).map(Vec::as_slice)
    }

    /// Appends `sample` to `texel`'s bucket; out-of-range texels are skipped.
    fn push(&mut self, texel: u32, sample: TransmittanceSample) {
        if let Some(bucket) = self.buckets.get_mut(texel as usize) {
            bucket.push(sample);
        }
    }
}

/// Routes indexed strand samples into per-texel buckets.
///
/// Preserves input order within each bucket for deterministic downstream
/// accumulation; a sample whose `texel` falls outside `0..texel_count` is
/// skipped rather than panicking, so a stale light-texel index cannot crash the
/// transmittance build.
#[must_use]
pub fn bin_samples(samples: &[TexelSample], texel_count: u32) -> TransmittanceBins {
    let mut bins = TransmittanceBins::new(texel_count);
    for entry in samples {
        bins.push(entry.texel, entry.sample);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn zero_opacity_sample_is_fully_transmissive() {
        let layers = build_deep_opacity(&[TransmittanceSample::new(3.0, 0.0)], 4, 0.0);
        for &t in &layers.layer_transmittance {
            assert!(close(t, 1.0), "expected full transmittance, got {t}");
        }
        assert!(close(sample_transmittance(&layers, 0.0), 1.0));
        assert!(close(sample_transmittance(&layers, 100.0), 1.0));
    }

    #[test]
    fn opaque_sample_blocks_behind_and_passes_in_front() {
        let d = 5.0;
        let layers = build_deep_opacity(&[TransmittanceSample::new(d, 1.0)], 2, 0.0);
        // Shallower than the occluder: fully lit.
        assert!(close(sample_transmittance(&layers, d - 1.0), 1.0));
        // Deeper than the occluder: fully shadowed.
        assert!(close(sample_transmittance(&layers, d + 1.0), 0.0));
    }

    #[test]
    fn two_samples_multiply_their_transmittance() {
        let a1 = 0.5;
        let a2 = 0.5;
        let layers = build_deep_opacity(
            &[
                TransmittanceSample::new(0.0, a1),
                TransmittanceSample::new(10.0, a2),
            ],
            2,
            0.0,
        );
        // Deepest receiver sees the full product T = (1 - a1)(1 - a2).
        let deep = sample_transmittance(&layers, 100.0);
        assert!(close(deep, (1.0 - a1) * (1.0 - a2)), "got {deep}");
    }

    #[test]
    fn stable_sort_makes_result_order_independent() {
        let ordered = [
            TransmittanceSample::new(0.0, 0.2),
            TransmittanceSample::new(4.0, 0.3),
            TransmittanceSample::new(9.0, 0.4),
        ];
        let shuffled = [
            TransmittanceSample::new(9.0, 0.4),
            TransmittanceSample::new(0.0, 0.2),
            TransmittanceSample::new(4.0, 0.3),
        ];
        let a = build_deep_opacity(&ordered, 3, 0.0);
        let b = build_deep_opacity(&shuffled, 3, 0.0);
        assert_eq!(a.layer_depths.len(), b.layer_depths.len());
        for (x, y) in a.layer_depths.iter().zip(&b.layer_depths) {
            assert!(close(*x, *y));
        }
        for (x, y) in a.layer_transmittance.iter().zip(&b.layer_transmittance) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn layer_transmittance_is_monotonically_non_increasing() {
        let layers = build_deep_opacity(
            &[
                TransmittanceSample::new(0.0, 0.3),
                TransmittanceSample::new(2.0, 0.4),
                TransmittanceSample::new(5.0, 0.2),
                TransmittanceSample::new(8.0, 0.6),
            ],
            4,
            0.0,
        );
        for pair in layers.layer_transmittance.windows(2) {
            assert!(
                pair[1] <= pair[0] + EPS,
                "transmittance rose: {} -> {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn sample_transmittance_boundaries_and_midpoint_golden() {
        // Two samples of alpha 0.5 at depth 0 and 10, two layers, no offset:
        //   layer 0 boundary = 5,  T = 0.5
        //   layer 1 boundary = 10, T = 0.25
        let layers = build_deep_opacity(
            &[
                TransmittanceSample::new(0.0, 0.5),
                TransmittanceSample::new(10.0, 0.5),
            ],
            2,
            0.0,
        );
        assert!(close(layers.layer_depths[0], 5.0));
        assert!(close(layers.layer_depths[1], 10.0));
        assert!(close(layers.layer_transmittance[0], 0.5));
        assert!(close(layers.layer_transmittance[1], 0.25));

        // In front of the frontmost boundary -> fully lit.
        assert!(close(sample_transmittance(&layers, 2.0), 1.0));
        // At/behind the last boundary -> last value.
        assert!(close(sample_transmittance(&layers, 10.0), 0.25));
        assert!(close(sample_transmittance(&layers, 50.0), 0.25));
        // Midpoint depth 7.5 between 5 and 10: lerp(0.5, 0.25, 0.5) = 0.375.
        assert!(close(sample_transmittance(&layers, 7.5), 0.375));
    }

    #[test]
    fn empty_samples_are_fully_transmissive_and_never_panic() {
        let layers = build_deep_opacity(&[], 4, 0.0);
        assert!(layers.layer_depths.is_empty());
        assert!(layers.layer_transmittance.is_empty());
        assert!(close(sample_transmittance(&layers, 0.0), 1.0));
        assert!(close(sample_transmittance(&layers, 123.0), 1.0));
    }

    #[test]
    fn zero_layer_count_is_clamped_to_one() {
        let layers = build_deep_opacity(&[TransmittanceSample::new(1.0, 0.5)], 0, 0.0);
        assert_eq!(layers.layer_depths.len(), 1);
        assert_eq!(layers.layer_transmittance.len(), 1);
        assert!(close(layers.layer_transmittance[0], 0.5));
    }

    #[test]
    fn voxel_density_accumulates_into_correct_cells() {
        let samples = [
            TransmittanceSample::new(1.0, 0.5),  // voxel 0 (width 2)
            TransmittanceSample::new(3.0, 0.5),  // voxel 1
            TransmittanceSample::new(20.0, 0.9), // out of slab -> skipped
        ];
        let densities = accumulate_voxel_density(&samples, 0.0, 10.0, 5);
        assert_eq!(densities.len(), 5);
        assert!(close(densities[0], 0.5));
        assert!(close(densities[1], 0.5));
        assert!(close(densities[2], 0.0));
        assert!(close(densities[3], 0.0));
        assert!(close(densities[4], 0.0));

        // T at voxel 0 = 1 - 0.5 = 0.5; at voxel 1 = 0.5 * 0.5 = 0.25; monotone.
        let t0 = voxel_transmittance(&densities, 0);
        let t1 = voxel_transmittance(&densities, 1);
        assert!(close(t0, 0.5));
        assert!(close(t1, 0.25));
        assert!(t1 <= t0 + EPS);
        // Index past the end clamps to the last voxel.
        assert!(close(voxel_transmittance(&densities, 999), t1));
    }

    #[test]
    fn voxel_zero_count_clamps_and_never_panics() {
        let densities =
            accumulate_voxel_density(&[TransmittanceSample::new(1.0, 0.5)], 0.0, 10.0, 0);
        assert_eq!(densities.len(), 1);
        assert!(close(densities[0], 0.5));
        // Empty slab and empty samples are also panic-free.
        let degenerate =
            accumulate_voxel_density(&[TransmittanceSample::new(1.0, 0.5)], 5.0, 5.0, 4);
        assert_eq!(degenerate.len(), 4);
        for &d in &degenerate {
            assert!(close(d, 0.0));
        }
        assert!(close(voxel_transmittance(&[], 0), 1.0));
    }

    #[test]
    fn binning_routes_by_texel_and_skips_out_of_range() {
        let samples = [
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(1.0, 0.5),
            },
            TexelSample {
                texel: 2,
                sample: TransmittanceSample::new(2.0, 0.25),
            },
            TexelSample {
                texel: 2,
                sample: TransmittanceSample::new(3.0, 0.75),
            },
            TexelSample {
                // Out of range: skipped, not panicked on.
                texel: 9,
                sample: TransmittanceSample::new(4.0, 1.0),
            },
        ];
        let bins = bin_samples(&samples, 3);
        assert_eq!(bins.len(), 3);
        assert!(!bins.is_empty());
        assert_eq!(bins.total(), 3);
        assert_eq!(bins.bucket(0).map(<[_]>::len), Some(1));
        assert_eq!(bins.bucket(1).map(<[_]>::len), Some(0));
        assert_eq!(bins.bucket(2).map(<[_]>::len), Some(2));
        assert_eq!(bins.bucket(3), None);

        // Preserves input order within the bucket.
        let bucket2 = bins.bucket(2).expect("texel 2 exists");
        assert!(close(bucket2[0].opacity, 0.25));
        assert!(close(bucket2[1].opacity, 0.75));
    }

    #[test]
    fn empty_bins_report_empty() {
        let bins = bin_samples(&[], 4);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
        assert_eq!(bins.len(), 4);
    }
}
