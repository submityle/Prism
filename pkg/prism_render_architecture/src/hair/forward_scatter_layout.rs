//! Host-side forward-scatter map layout: pack many per-texel strand buckets
//! into one flat, upload-ready slab of crossing counts.
//!
//! [`dual_scattering`](super::dual_scattering) turns the strand samples on *one*
//! light ray/texel into a monotone coverage-weighted crossing curve `n(d)`. A
//! whole light-space forward-scatter texture, however, needs the *same* fixed
//! layer count per texel so the result can be stored as a device-uploadable slab
//! (one array of near depths, one of layer steps, one flat crossing grid) that a
//! dual-scattering shading pass decodes with a constant stride. That cross-texel
//! packing is a host build step distinct from the per-ray accumulation, and is
//! the forward-scatter counterpart of [`super::deep_opacity_layout`]: sort per
//! texel, slice a fixed number of depth layers, and accumulate the *additive*
//! crossing count `sum(alpha)` rather than the deep-opacity product
//! `product(1 - alpha)`.
//!
//! This module owns only that layout/packing step. It reuses the exact
//! deterministic accumulation of
//! [`super::dual_scattering::build_forward_scatter`] (stable depth sort,
//! equal-width layers, additive `sum(alpha)`, last boundary pinned to the
//! deepest sample) so a single-texel row decodes bit-for-bit like the per-ray
//! path — the tests cross-check the two. Everything is array-in/array-out,
//! panic-free on empty or out-of-range input, and free of exponentials for
//! deterministic goldens. The *device-side* sort that would build this on the
//! GPU is separate scheduling and is not owned here.

use alloc::vec::Vec;

use super::deep_transmittance::{TransmittanceBins, TransmittanceSample};

/// A packed forward-scatter map covering every light texel with a fixed layer
/// count, ready to upload as three parallel buffers.
///
/// Layout is texel-major, layer-minor: the crossing count of layer `l` at texel
/// `t` lives at `crossings[t * layer_count + l]`, the cumulative sum
/// `sum(alpha)` of every strand sample at or in front of that layer's far
/// boundary. `near_depth[t]` is the front of texel `t`'s first layer and
/// `layer_step[t]` its uniform layer width, so layer `l`'s far boundary is
/// `near_depth[t] + layer_step[t] * (l + 1)`. Storing per-texel near/step (not a
/// global range) lets each texel bracket its own strand depth span while the
/// grid stays a constant stride.
///
/// A texel with no samples has zero crossings: `near_depth`/`layer_step` are
/// `0` and its whole crossing row is `0.0`. An empty map (`texel_count == 0`)
/// reports zero crossings everywhere via [`map_forward_scatter`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForwardScatterMap {
    /// Number of light texels (rows in the slab).
    pub texel_count: usize,
    /// Fixed layers per texel (columns in the slab), always `>= 1`.
    pub layer_count: usize,
    /// Per-texel front depth of the first layer; length `texel_count`.
    pub near_depth: Vec<f32>,
    /// Per-texel uniform layer width; length `texel_count`.
    pub layer_step: Vec<f32>,
    /// Flat texel-major crossing grid; length `texel_count * layer_count`.
    pub crossings: Vec<f32>,
}

impl ForwardScatterMap {
    /// Borrows the `layer_count` crossing values for `texel`, or `None` when
    /// `texel` is out of range.
    #[must_use]
    pub fn layers(&self, texel: usize) -> Option<&[f32]> {
        if texel >= self.texel_count {
            return None;
        }
        let base = texel * self.layer_count;
        Some(&self.crossings[base..base + self.layer_count])
    }
}

/// Packs one texel's samples into `layer_count` cumulative-crossing values,
/// appending them to `out`, and returns that texel's `(near, step)`.
///
/// This is the per-texel core of the host build; it repeats the exact
/// accumulation of [`super::dual_scattering::build_forward_scatter`] (stable
/// ascending sort by depth, equal-width layers over
/// `shallowest + start_offset ..= deepest`, running additive `sum(alpha)`, last
/// boundary pinned to the deepest sample) so the packed slab decodes identically
/// to the per-ray curve.
fn pack_bucket(
    samples: &[TransmittanceSample],
    layer_count: usize,
    start_offset: f32,
    out: &mut Vec<f32>,
) -> (f32, f32) {
    if samples.is_empty() {
        for _ in 0..layer_count {
            out.push(0.0);
        }
        return (0.0, 0.0);
    }

    // Stable total-order sort so equal-depth samples keep input order and the
    // layout is independent of arrival order (matches the per-ray path).
    let mut sorted: Vec<TransmittanceSample> = samples.to_vec();
    sorted.sort_by(|a, b| a.depth.total_cmp(&b.depth));

    let shallowest = sorted[0].depth;
    let deepest = sorted[sorted.len() - 1].depth;
    let start = shallowest + start_offset.max(0.0);
    let end = deepest.max(start);
    let width = (end - start) / layer_count as f32;

    let mut running = 0.0_f32;
    let mut next = 0usize;
    for i in 0..layer_count {
        let boundary = if i + 1 == layer_count {
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
        out.push(running);
    }

    (start, width)
}

/// Builds a packed [`ForwardScatterMap`] from per-texel strand buckets.
///
/// `bins` is the fan-out from [`super::deep_transmittance::bin_samples`]: one
/// bucket per light texel. Every texel is sliced into the same `layer_count`
/// depth layers (clamped to at least `1`) with the same near-bias `start_offset`
/// (clamped to `>= 0`), and its cumulative crossing row is appended to the flat
/// slab in texel order. Empty buckets pack as a zero-crossing row. The result's
/// `texel_count` equals `bins.len()`. Never panics.
#[must_use]
pub fn build_forward_scatter_map(
    bins: &TransmittanceBins,
    layer_count: u32,
    start_offset: f32,
) -> ForwardScatterMap {
    let layer_count = layer_count.max(1) as usize;
    let texel_count = bins.len();

    let mut near_depth = Vec::with_capacity(texel_count);
    let mut layer_step = Vec::with_capacity(texel_count);
    let mut crossings = Vec::with_capacity(texel_count * layer_count);

    for texel in 0..texel_count {
        let bucket = bins.bucket(texel as u32).unwrap_or(&[]);
        let (near, step) = pack_bucket(bucket, layer_count, start_offset, &mut crossings);
        near_depth.push(near);
        layer_step.push(step);
    }

    ForwardScatterMap {
        texel_count,
        layer_count,
        near_depth,
        layer_step,
        crossings,
    }
}

/// Samples the packed map: the coverage-weighted crossing count a receiver at
/// `depth` sees on `texel`.
///
/// Reconstructs the texel's layer boundaries from its `near`/`step` and mirrors
/// [`super::dual_scattering::sample_forward_scatter`]: receivers in front of the
/// first boundary see `0` crossings, receivers at or beyond the last boundary
/// take the largest (saturated) value, and in between the two bracketing layers
/// are linearly interpolated. Out-of-range `texel` (or a degenerate map) returns
/// `0.0`. The result is monotonically non-decreasing in `depth` and never
/// negative. Never panics.
#[must_use]
pub fn map_forward_scatter(map: &ForwardScatterMap, texel: usize, depth: f32) -> f32 {
    let Some(crossings) = map.layers(texel) else {
        return 0.0;
    };
    if map.layer_count == 0 {
        return 0.0;
    }
    let near = map.near_depth[texel];
    let step = map.layer_step[texel];
    let last = map.layer_count - 1;

    let boundary = |i: usize| near + step * (i as f32 + 1.0);

    if depth <= boundary(0) {
        return 0.0;
    }
    if depth >= boundary(last) {
        return crossings[last];
    }
    for i in 0..last {
        let b0 = boundary(i);
        let b1 = boundary(i + 1);
        if depth <= b1 {
            let span = b1 - b0;
            let t = if span > 0.0 { (depth - b0) / span } else { 0.0 };
            return crossings[i] + (crossings[i + 1] - crossings[i]) * t;
        }
    }
    crossings[last]
}

#[cfg(test)]
mod tests {
    use super::super::deep_transmittance::{bin_samples, TexelSample};
    use super::super::dual_scattering::{build_forward_scatter, sample_forward_scatter};
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn packs_row_major_with_fixed_stride() {
        let samples = [
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(0.0, 0.5),
            },
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(10.0, 0.5),
            },
            TexelSample {
                texel: 2,
                sample: TransmittanceSample::new(4.0, 1.0),
            },
        ];
        let bins = bin_samples(&samples, 3);
        let map = build_forward_scatter_map(&bins, 2, 0.0);

        assert_eq!(map.texel_count, 3);
        assert_eq!(map.layer_count, 2);
        assert_eq!(map.crossings.len(), 6);

        // Texel 0: two coverage-0.5 samples over two layers -> 0.5, 1.0.
        let t0 = map.layers(0).expect("texel 0");
        assert!(close(t0[0], 0.5));
        assert!(close(t0[1], 1.0));
        // Texel 1: empty -> zero crossings.
        let t1 = map.layers(1).expect("texel 1");
        assert!(close(t1[0], 0.0));
        assert!(close(t1[1], 0.0));
        // Texel 2: one full-coverage sample -> saturates at 1.0.
        let t2 = map.layers(2).expect("texel 2");
        assert!(close(t2[1], 1.0));
        // Out of range.
        assert!(map.layers(3).is_none());
    }

    #[test]
    fn single_texel_row_matches_per_ray_curve() {
        // The packed row must decode bit-for-bit like the per-ray dual_scattering
        // path across the whole depth domain.
        let ray = [
            TransmittanceSample::new(1.0, 0.3),
            TransmittanceSample::new(4.0, 0.4),
            TransmittanceSample::new(9.0, 0.2),
        ];
        let per_ray = build_forward_scatter(&ray, 4, 0.5);

        let texel_samples: Vec<TexelSample> = ray
            .iter()
            .map(|&sample| TexelSample { texel: 0, sample })
            .collect();
        let bins = bin_samples(&texel_samples, 1);
        let map = build_forward_scatter_map(&bins, 4, 0.5);

        for &d in &[0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 20.0] {
            let a = sample_forward_scatter(&per_ray, d);
            let b = map_forward_scatter(&map, 0, d);
            assert!(close(a, b), "depth {d}: per-ray {a} vs map {b}");
        }
    }

    #[test]
    fn rows_are_monotone_non_decreasing_and_saturate() {
        let samples = [
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(0.0, 0.25),
            },
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(3.0, 0.25),
            },
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(6.0, 0.25),
            },
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(9.0, 0.25),
            },
        ];
        let bins = bin_samples(&samples, 1);
        let map = build_forward_scatter_map(&bins, 4, 0.0);
        let row = map.layers(0).expect("texel 0");
        for pair in row.windows(2) {
            assert!(pair[1] >= pair[0] - EPS);
        }
        // Full stack counted by the last layer: 4 * 0.25 = 1.0.
        assert!(close(row[3], 1.0));
        // Behind the last boundary saturates; in front of the first is zero.
        assert!(close(map_forward_scatter(&map, 0, 100.0), 1.0));
        assert!(close(map_forward_scatter(&map, 0, -1.0), 0.0));
    }

    #[test]
    fn empty_map_and_out_of_range_are_zero_and_never_panic() {
        let empty = build_forward_scatter_map(&bin_samples(&[], 0), 4, 0.0);
        // bin_samples clamps texel_count to at least 1.
        assert_eq!(empty.texel_count, 1);
        assert!(close(map_forward_scatter(&empty, 0, 5.0), 0.0));
        // Out-of-range texel is fully unscattered.
        assert!(close(map_forward_scatter(&empty, 99, 5.0), 0.0));

        let truly_empty = ForwardScatterMap::default();
        assert!(close(map_forward_scatter(&truly_empty, 0, 5.0), 0.0));
    }

    #[test]
    fn zero_layer_count_is_clamped_to_one() {
        let samples = [TexelSample {
            texel: 0,
            sample: TransmittanceSample::new(1.0, 0.5),
        }];
        let map = build_forward_scatter_map(&bin_samples(&samples, 1), 0, 0.0);
        assert_eq!(map.layer_count, 1);
        assert_eq!(map.crossings.len(), 1);
        assert!(close(map.crossings[0], 0.5));
    }

    #[test]
    fn negative_start_offset_is_clamped_like_the_per_ray_path() {
        let samples = [
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(2.0, 0.5),
            },
            TexelSample {
                texel: 0,
                sample: TransmittanceSample::new(6.0, 0.5),
            },
        ];
        let bins = bin_samples(&samples, 1);
        let clamped = build_forward_scatter_map(&bins, 2, -50.0);
        let baseline = build_forward_scatter_map(&bins, 2, 0.0);
        assert!(close(clamped.near_depth[0], baseline.near_depth[0]));
        assert!(close(clamped.layer_step[0], baseline.layer_step[0]));
        for (a, b) in clamped.crossings.iter().zip(baseline.crossings.iter()) {
            assert!(close(*a, *b));
        }
    }
}
