//! Host-side deep opacity map layout: pack many per-texel strand buckets into
//! one flat, upload-ready slab.
//!
//! [`deep_transmittance`](super::deep_transmittance) turns the strand samples on
//! *one* light ray/texel into a monotone transmittance curve. A production deep
//! opacity map, however, covers a whole light-space texture: every texel needs
//! the *same* fixed layer count so the result can be stored as a
//! device-uploadable slab (one array of near depths, one of layer steps, one
//! flat transmittance grid) that a shading pass decodes with a constant stride.
//! That cross-texel packing is a host build step distinct from the per-ray
//! accumulation, and mirrors how `UE5` Groom and AMD `TressFX` bake their
//! self-shadow maps: sort per texel, slice a fixed number of depth layers, and
//! composite `T = product(1 - alpha)`.
//!
//! This module owns only that layout/packing step. It reuses the same
//! deterministic composite math as [`super::deep_transmittance::build_deep_opacity`]
//! (stable depth sort, equal-width layers, `alpha`-composite product) so a
//! single-texel map decodes bit-for-bit like the per-ray path — the tests
//! cross-check the two. Everything is array-in/array-out, panic-free on empty or
//! out-of-range input, and free of exponentials for deterministic goldens. The
//! *device-side* sort that would build this on the GPU is separate scheduling
//! and is not owned here.

use alloc::vec::Vec;

use super::deep_transmittance::{TransmittanceBins, TransmittanceSample};

/// A packed deep opacity map covering every light texel with a fixed layer
/// count, ready to upload as three parallel buffers.
///
/// Layout is texel-major, layer-minor: the transmittance of layer `l` at texel
/// `t` lives at `transmittance[t * layer_count + l]`, the cumulative product
/// `product(1 - alpha)` of every strand sample at or in front of that layer's
/// far boundary. `near_depth[t]` is the front of texel `t`'s first layer and
/// `layer_step[t]` its uniform layer width, so layer `l`'s far boundary is
/// `near_depth[t] + layer_step[t] * (l + 1)`. Storing per-texel near/step (not a
/// global range) lets each texel bracket its own strand depth span while the
/// grid stays a constant stride.
///
/// A texel with no samples is fully transmissive: `near_depth`/`layer_step` are
/// `0` and its whole transmittance row is `1.0`. An empty map (`texel_count ==
/// 0`) reports fully lit everywhere via [`map_transmittance`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepOpacityMap {
    /// Number of light texels (rows in the slab).
    pub texel_count: usize,
    /// Fixed layers per texel (columns in the slab), always `>= 1`.
    pub layer_count: usize,
    /// Per-texel front depth of the first layer; length `texel_count`.
    pub near_depth: Vec<f32>,
    /// Per-texel uniform layer width; length `texel_count`.
    pub layer_step: Vec<f32>,
    /// Flat texel-major transmittance grid; length `texel_count * layer_count`.
    pub transmittance: Vec<f32>,
}

impl DeepOpacityMap {
    /// Borrows the `layer_count` transmittance values for `texel`, or `None`
    /// when `texel` is out of range.
    #[must_use]
    pub fn layers(&self, texel: usize) -> Option<&[f32]> {
        if texel >= self.texel_count {
            return None;
        }
        let base = texel * self.layer_count;
        Some(&self.transmittance[base..base + self.layer_count])
    }
}

/// Packs one texel's samples into `layer_count` cumulative-transmittance values,
/// appending them to `out`, and returns that texel's `(near, step)`.
///
/// This is the per-texel core of the host build; it repeats the exact composite
/// of [`super::deep_transmittance::build_deep_opacity`] (stable ascending sort
/// by depth, equal-width layers over `shallowest + start_offset ..= deepest`,
/// running `product(1 - alpha)`, last boundary pinned to the deepest sample) so
/// the packed slab decodes identically to the per-ray map.
fn pack_bucket(
    samples: &[TransmittanceSample],
    layer_count: usize,
    start_offset: f32,
    out: &mut Vec<f32>,
) -> (f32, f32) {
    if samples.is_empty() {
        for _ in 0..layer_count {
            out.push(1.0);
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

    let mut running = 1.0_f32;
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
            running *= 1.0 - sorted[next].opacity.clamp(0.0, 1.0);
            next += 1;
        }
        out.push(running);
    }

    (start, width)
}

/// Builds a packed [`DeepOpacityMap`] from per-texel strand buckets.
///
/// `bins` is the fan-out from [`super::deep_transmittance::bin_samples`]: one
/// bucket per light texel. Every texel is sliced into the same `layer_count`
/// depth layers (clamped to at least `1`) with the same near-bias `start_offset`
/// (clamped to `>= 0`), and its cumulative transmittance row is appended to the
/// flat slab in texel order. Empty buckets pack as a fully transmissive row.
/// The result's `texel_count` equals `bins.len()`. Never panics.
#[must_use]
pub fn build_deep_opacity_map(
    bins: &TransmittanceBins,
    layer_count: u32,
    start_offset: f32,
) -> DeepOpacityMap {
    let layer_count = layer_count.max(1) as usize;
    let texel_count = bins.len();

    let mut near_depth = Vec::with_capacity(texel_count);
    let mut layer_step = Vec::with_capacity(texel_count);
    let mut transmittance = Vec::with_capacity(texel_count * layer_count);

    for texel in 0..texel_count {
        let bucket = bins.bucket(texel as u32).unwrap_or(&[]);
        let (near, step) = pack_bucket(bucket, layer_count, start_offset, &mut transmittance);
        near_depth.push(near);
        layer_step.push(step);
    }

    DeepOpacityMap {
        texel_count,
        layer_count,
        near_depth,
        layer_step,
        transmittance,
    }
}

/// Samples the packed map: the transmittance a receiver at `depth` sees on
/// `texel`.
///
/// Reconstructs the texel's layer boundaries from its `near`/`step` and mirrors
/// [`super::deep_transmittance::sample_transmittance`]: receivers in front of
/// the first boundary are fully lit (`1.0`), receivers at or beyond the last
/// boundary take the most-occluded value, and in between the two bracketing
/// layers are linearly interpolated. Out-of-range `texel` (or a degenerate map)
/// returns `1.0`. The result is monotonically non-increasing in `depth` and
/// always in `0..=1`. Never panics.
#[must_use]
pub fn map_transmittance(map: &DeepOpacityMap, texel: usize, depth: f32) -> f32 {
    let Some(trans) = map.layers(texel) else {
        return 1.0;
    };
    if map.layer_count == 0 {
        return 1.0;
    }
    let near = map.near_depth[texel];
    let step = map.layer_step[texel];
    let last = map.layer_count - 1;

    let boundary = |i: usize| near + step * (i as f32 + 1.0);

    if depth <= boundary(0) {
        return 1.0;
    }
    if depth >= boundary(last) {
        return trans[last];
    }
    for i in 0..last {
        let b0 = boundary(i);
        let b1 = boundary(i + 1);
        if depth <= b1 {
            let span = b1 - b0;
            let t = if span > 0.0 { (depth - b0) / span } else { 0.0 };
            return trans[i] + (trans[i + 1] - trans[i]) * t;
        }
    }
    trans[last]
}

#[cfg(test)]
mod tests {
    use super::super::deep_transmittance::{
        bin_samples, build_deep_opacity, sample_transmittance, TexelSample,
    };
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
        let map = build_deep_opacity_map(&bins, 2, 0.0);

        assert_eq!(map.texel_count, 3);
        assert_eq!(map.layer_count, 2);
        assert_eq!(map.near_depth.len(), 3);
        assert_eq!(map.layer_step.len(), 3);
        assert_eq!(map.transmittance.len(), 6);

        // Texel 0: two alpha-0.5 occluders at depth 0 and 10, two layers ->
        // layer 0 T = 0.5, layer 1 T = 0.25 (mirrors the per-ray golden).
        let row0 = map.layers(0).expect("texel 0");
        assert!(close(row0[0], 0.5));
        assert!(close(row0[1], 0.25));

        // Texel 1 has no samples: fully transmissive row.
        let row1 = map.layers(1).expect("texel 1");
        assert!(close(row1[0], 1.0));
        assert!(close(row1[1], 1.0));

        // Texel 2: single opaque occluder -> fully blocked once past it.
        let row2 = map.layers(2).expect("texel 2");
        assert!(close(row2[1], 0.0));

        assert!(map.layers(3).is_none());
    }

    #[test]
    fn decode_matches_per_ray_path_for_single_texel() {
        // A single texel's packed decode must equal the standalone per-ray
        // deep opacity map at every probe depth.
        let bucket = [
            TransmittanceSample::new(0.0, 0.3),
            TransmittanceSample::new(2.0, 0.4),
            TransmittanceSample::new(5.0, 0.2),
            TransmittanceSample::new(8.0, 0.6),
        ];
        let indexed: Vec<TexelSample> = bucket
            .iter()
            .map(|&sample| TexelSample { texel: 0, sample })
            .collect();
        let bins = bin_samples(&indexed, 1);
        let map = build_deep_opacity_map(&bins, 4, 0.5);
        let reference = build_deep_opacity(&bucket, 4, 0.5);

        for probe in [0.0_f32, 1.0, 2.5, 4.0, 6.0, 8.0, 12.0] {
            let packed = map_transmittance(&map, 0, probe);
            let per_ray = sample_transmittance(&reference, probe);
            assert!(
                close(packed, per_ray),
                "depth {probe}: packed {packed} vs per-ray {per_ray}",
            );
        }
    }

    #[test]
    fn transmittance_is_monotone_per_texel() {
        let indexed: Vec<TexelSample> = [
            TransmittanceSample::new(1.0, 0.3),
            TransmittanceSample::new(3.0, 0.5),
            TransmittanceSample::new(6.0, 0.2),
            TransmittanceSample::new(9.0, 0.7),
        ]
        .into_iter()
        .map(|sample| TexelSample { texel: 0, sample })
        .collect();
        let bins = bin_samples(&indexed, 1);
        let map = build_deep_opacity_map(&bins, 4, 0.0);

        let row = map.layers(0).expect("texel 0");
        for pair in row.windows(2) {
            assert!(pair[1] <= pair[0] + EPS, "rose: {} -> {}", pair[0], pair[1]);
        }
        // Sampling is also monotone non-increasing in depth.
        let mut prev = 1.0_f32;
        for step in 0..20 {
            let d = step as f32 * 0.5;
            let t = map_transmittance(&map, 0, d);
            assert!(t <= prev + EPS, "sample rose at depth {d}: {prev} -> {t}");
            assert!((0.0..=1.0).contains(&t));
            prev = t;
        }
    }

    #[test]
    fn empty_texel_and_out_of_range_are_fully_lit() {
        let bins = bin_samples(&[], 3);
        let map = build_deep_opacity_map(&bins, 4, 0.0);
        assert_eq!(map.texel_count, 3);
        // Every row is fully transmissive.
        for texel in 0..3 {
            assert!(close(map_transmittance(&map, texel, 0.0), 1.0));
            assert!(close(map_transmittance(&map, texel, 123.0), 1.0));
        }
        // Out-of-range texel never panics and reads fully lit.
        assert!(close(map_transmittance(&map, 99, 5.0), 1.0));
    }

    #[test]
    fn zero_layer_count_clamps_to_one() {
        let indexed = [TexelSample {
            texel: 0,
            sample: TransmittanceSample::new(2.0, 0.5),
        }];
        let bins = bin_samples(&indexed, 1);
        let map = build_deep_opacity_map(&bins, 0, 0.0);
        assert_eq!(map.layer_count, 1);
        assert_eq!(map.transmittance.len(), 1);
        assert!(close(map.transmittance[0], 0.5));
    }

    #[test]
    fn packing_is_order_independent() {
        let ordered = [
            TexelSample {
                texel: 1,
                sample: TransmittanceSample::new(0.0, 0.2),
            },
            TexelSample {
                texel: 1,
                sample: TransmittanceSample::new(4.0, 0.3),
            },
            TexelSample {
                texel: 1,
                sample: TransmittanceSample::new(9.0, 0.4),
            },
        ];
        let shuffled = [ordered[2], ordered[0], ordered[1]];
        let a = build_deep_opacity_map(&bin_samples(&ordered, 2), 3, 0.0);
        let b = build_deep_opacity_map(&bin_samples(&shuffled, 2), 3, 0.0);
        assert_eq!(a.transmittance.len(), b.transmittance.len());
        for (x, y) in a.transmittance.iter().zip(&b.transmittance) {
            assert!(close(*x, *y), "packing differed: {x} vs {y}");
        }
        for (x, y) in a.near_depth.iter().zip(&b.near_depth) {
            assert!(close(*x, *y));
        }
        for (x, y) in a.layer_step.iter().zip(&b.layer_step) {
            assert!(close(*x, *y));
        }
    }
}
