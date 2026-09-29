//! Sub-pixel hair strand rasterization path classification and binning.
//!
//! A render strand is *thin* — usually a fraction of a pixel wide — so the
//! hardware triangle path wastes almost all of its work on quad overshading and
//! setup for a primitive that barely covers a sample. Production hair engines
//! (UE5 Groom, AMD `TressFX`) instead rasterize thin strands in a compute
//! *visibility* pass: each strand segment is projected, its coverage is
//! accumulated into a per-sample vis-buffer with analytic sub-pixel weighting,
//! and shading is deferred to the resolved fragments. Only near, thick strands
//! (close-up hero hair, hair *cards*, or barrettes) are wide enough to earn the
//! hardware path.
//!
//! This module is the CPU-side classifier that decides, per strand segment,
//! which of those paths a segment takes and fans a frame's segments into one
//! bucket per path. It mirrors [`bin_cut`](crate::virtual_geometry) exactly:
//! a pure per-segment classification from screen-space statistics, a
//! deterministic single pass that preserves input order within every bucket,
//! and out-of-range indices skipped rather than panicking. The physical
//! rasterization lives in the backend; the compute software bucket feeds the
//! shared `virtual_geometry` software-raster path and, downstream, the
//! transparency subsystem's `HairVisibility` resolve.

use alloc::vec::Vec;

/// Default strand screen width, in pixels, at or below which a segment is
/// rasterized through the compute software path.
///
/// Two pixels keeps genuinely thin strands — the overwhelming majority of a
/// groom at any reasonable distance — on the sub-pixel software path while
/// letting only close-up thick strands escape to the hardware path.
pub const DEFAULT_HAIR_SOFTWARE_WIDTH_PX: f32 = 2.0;

/// Default minimum projected coverage below which a segment is culled.
///
/// Sub-8-bit coverage cannot survive an 8-bit composite and only contributes
/// shimmering noise as strands flicker in and out between frames, so it is
/// dropped before it reaches the vis-buffer.
pub const DEFAULT_HAIR_MIN_COVERAGE: f32 = 1.0 / 256.0;

/// Screen-space statistics for one projected strand segment.
///
/// All fields are non-negative by contract; the classifier is defensive and
/// treats a non-positive length or sub-threshold coverage as a cull rather than
/// trusting the caller. Coverage is the fraction of its footprint the segment
/// analytically covers after sub-pixel projection, in `0..=1`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HairStrandSegmentStats {
    /// Projected screen width of the strand at this segment, in pixels.
    pub screen_width_px: f32,
    /// Projected screen length of the segment along the strand, in pixels.
    pub screen_length_px: f32,
    /// Analytic sub-pixel coverage fraction contributed by the segment, `0..=1`.
    pub coverage: f32,
}

/// The rasterization path a single strand segment takes this frame.
///
/// This is hair-specific and deliberately distinct from
/// [`GeometryRasterPath`](crate::virtual_geometry::GeometryRasterPath): a strand
/// segment is a capsule, not a cluster of triangles, and its "hardware" case is
/// a thick-strand tessellation rather than a mesh-shader or indirect draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HairRasterPath {
    /// Thin strand rasterized in the compute software / visibility pass; its
    /// coverage is accumulated analytically into the vis-buffer.
    SubpixelSoftware,
    /// Near, thick strand (or a hair card / barrette) wide enough to earn the
    /// hardware triangle path.
    ThickHardware,
    /// Back-facing, sub-visible, or zero-length segment dropped before raster
    /// to prevent shimmering noise.
    Culled,
}

/// Tunables driving [`classify_hair_raster`].
///
/// Both fields are clamped to non-negative when the classifier reads them, so a
/// caller cannot make the thresholds nonsensical: a negative `min_coverage`
/// behaves like zero (cull only truly empty segments) and a negative
/// `software_width_threshold` behaves like zero (never take the software path).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairRasterParams {
    /// Coverage strictly below this value is culled. Clamped to `>= 0`.
    pub min_coverage: f32,
    /// Screen width at or below this value takes the software path. Clamped to
    /// `>= 0`.
    pub software_width_threshold: f32,
}

impl Default for HairRasterParams {
    fn default() -> Self {
        Self {
            min_coverage: DEFAULT_HAIR_MIN_COVERAGE,
            software_width_threshold: DEFAULT_HAIR_SOFTWARE_WIDTH_PX,
        }
    }
}

/// Deterministic CPU-side parameters handed to the compute software rasterizer.
///
/// This is an *ABI* description, not the rasterizer itself: it captures the
/// knobs the backend compute pass needs so that the CPU side (and its golden
/// tests) and the GPU side agree on behavior. The physical rasterization —
/// sample coverage evaluation, vis-buffer atomics, per-tile segment lists —
/// runs in the backend; this struct only names the deterministic parameters
/// that pass consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairSoftRasterAbi {
    /// Sub-pixel sample count per pixel used to evaluate analytic strand
    /// coverage. A power of two keeps the sample grid exact.
    pub subpixel_samples: u32,
    /// Coverage contributions at or below this epsilon are discarded inside the
    /// pass, matching the CPU-side [`HairRasterParams::min_coverage`] cull so
    /// the two sides route identically.
    pub coverage_epsilon: f32,
    /// Screen-space edge length, in pixels, of one binning tile.
    pub tile_size_px: u32,
    /// Maximum strand segments recorded per tile before overflow is spilled to
    /// the next pass; bounds the per-tile list so the compute dispatch stays
    /// within a fixed workgroup budget.
    pub max_segments_per_tile: u32,
}

impl Default for HairSoftRasterAbi {
    fn default() -> Self {
        Self {
            subpixel_samples: 8,
            coverage_epsilon: DEFAULT_HAIR_MIN_COVERAGE,
            tile_size_px: 16,
            max_segments_per_tile: 256,
        }
    }
}

/// A reference to one strand segment, carrying the index of its statistics.
///
/// `strand` and `segment` identify the geometry; `stats` indexes the parallel
/// [`HairStrandSegmentStats`] slice passed to [`bin_hair_raster`]. Carrying an
/// explicit statistics index (rather than reusing the position in the segment
/// slice) lets several segments share one projected statistic and lets a stale
/// reference point past the slice, which is skipped rather than panicking —
/// exactly as [`CutCluster::node`](crate::virtual_geometry) indexes its own
/// parallel statistics slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HairSegmentRef {
    /// Index of the owning render strand.
    pub strand: u32,
    /// Index of the segment within its strand.
    pub segment: u32,
    /// Index into the parallel statistics slice.
    pub stats: u32,
}

impl HairSegmentRef {
    /// Builds a segment reference from its strand, segment, and statistics
    /// indices.
    #[must_use]
    pub const fn new(strand: u32, segment: u32, stats: u32) -> Self {
        Self {
            strand,
            segment,
            stats,
        }
    }
}

/// A frame's strand segments partitioned by the raster path each one takes.
///
/// The backend consumes one bucket at a time: `subpixel_software` seeds the
/// compute visibility pass, `thick_hardware` becomes a hardware strand draw,
/// and `culled` is retained (rather than discarded) so callers can account for
/// every input segment. Per-bucket order matches the input order, keeping
/// submission deterministic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HairRasterBins {
    /// Thin strands routed to the compute software / visibility pass.
    pub subpixel_software: Vec<HairSegmentRef>,
    /// Thick strands routed to the hardware triangle path.
    pub thick_hardware: Vec<HairSegmentRef>,
    /// Segments culled before rasterization.
    pub culled: Vec<HairSegmentRef>,
}

impl HairRasterBins {
    /// Total number of segments across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.subpixel_software.len() + self.thick_hardware.len() + self.culled.len()
    }

    /// Returns `true` when no segment landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.subpixel_software.is_empty()
            && self.thick_hardware.is_empty()
            && self.culled.is_empty()
    }

    /// Immutable view of the bucket backing a given raster path.
    #[must_use]
    pub fn bucket(&self, path: HairRasterPath) -> &[HairSegmentRef] {
        match path {
            HairRasterPath::SubpixelSoftware => &self.subpixel_software,
            HairRasterPath::ThickHardware => &self.thick_hardware,
            HairRasterPath::Culled => &self.culled,
        }
    }

    /// Appends a segment to the bucket for its resolved raster path.
    pub fn push(&mut self, segment: HairSegmentRef, path: HairRasterPath) {
        self.bucket_mut(path).push(segment);
    }

    /// Mutable handle to the bucket backing a given raster path.
    fn bucket_mut(&mut self, path: HairRasterPath) -> &mut Vec<HairSegmentRef> {
        match path {
            HairRasterPath::SubpixelSoftware => &mut self.subpixel_software,
            HairRasterPath::ThickHardware => &mut self.thick_hardware,
            HairRasterPath::Culled => &mut self.culled,
        }
    }
}

/// Classifies the raster path for one strand segment.
///
/// A segment is [`HairRasterPath::Culled`] when it has no positive screen
/// length (back-facing or degenerate) or its coverage falls below the clamped
/// `min_coverage`. A surviving segment at or below the clamped
/// `software_width_threshold` takes [`HairRasterPath::SubpixelSoftware`];
/// anything wider takes [`HairRasterPath::ThickHardware`]. The width test uses
/// `<=` so the threshold itself resolves to the software path.
#[must_use]
pub fn classify_hair_raster(
    stats: HairStrandSegmentStats,
    params: HairRasterParams,
) -> HairRasterPath {
    let min_coverage = params.min_coverage.max(0.0);
    let width_threshold = params.software_width_threshold.max(0.0);
    if stats.screen_length_px <= 0.0 || stats.coverage < min_coverage {
        return HairRasterPath::Culled;
    }
    if stats.screen_width_px <= width_threshold {
        HairRasterPath::SubpixelSoftware
    } else {
        HairRasterPath::ThickHardware
    }
}

/// Partitions a frame's strand segments into per-path raster buckets.
///
/// Each [`HairSegmentRef::stats`] indexes the parallel `stats` slice. A segment
/// whose statistics index falls outside `stats` is skipped rather than
/// panicking, so a stale segment list cannot crash draw submission; every
/// in-range segment is routed via [`classify_hair_raster`]. Input order is
/// preserved within each bucket.
#[must_use]
pub fn bin_hair_raster(
    segments: &[HairSegmentRef],
    stats: &[HairStrandSegmentStats],
    params: HairRasterParams,
) -> HairRasterBins {
    let mut bins = HairRasterBins::default();
    for &segment in segments {
        let Some(&segment_stats) = stats.get(segment.stats as usize) else {
            continue;
        };
        bins.push(segment, classify_hair_raster(segment_stats, params));
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn stats(screen_width_px: f32, screen_length_px: f32, coverage: f32) -> HairStrandSegmentStats {
        HairStrandSegmentStats {
            screen_width_px,
            screen_length_px,
            coverage,
        }
    }

    #[test]
    fn thin_strand_takes_software_path() {
        let path = classify_hair_raster(stats(0.4, 20.0, 0.9), HairRasterParams::default());
        assert_eq!(path, HairRasterPath::SubpixelSoftware);
    }

    #[test]
    fn thick_strand_takes_hardware_path() {
        let path = classify_hair_raster(stats(6.0, 20.0, 0.9), HairRasterParams::default());
        assert_eq!(path, HairRasterPath::ThickHardware);
    }

    #[test]
    fn low_coverage_segment_is_culled() {
        let path = classify_hair_raster(stats(0.4, 20.0, 0.0), HairRasterParams::default());
        assert_eq!(path, HairRasterPath::Culled);
    }

    #[test]
    fn zero_length_segment_is_culled_even_when_thin_and_covered() {
        let path = classify_hair_raster(stats(0.4, 0.0, 1.0), HairRasterParams::default());
        assert_eq!(path, HairRasterPath::Culled);
    }

    #[test]
    fn negative_length_segment_is_culled() {
        let path = classify_hair_raster(stats(0.4, -1.0, 1.0), HairRasterParams::default());
        assert_eq!(path, HairRasterPath::Culled);
    }

    #[test]
    fn width_threshold_boundary_takes_software_path() {
        // Exactly at the threshold resolves to software because the test is `<=`.
        let params = HairRasterParams {
            min_coverage: 0.0,
            software_width_threshold: 2.0,
        };
        assert_eq!(
            classify_hair_raster(stats(2.0, 5.0, 1.0), params),
            HairRasterPath::SubpixelSoftware
        );
        // A hair past the threshold flips to hardware.
        assert_eq!(
            classify_hair_raster(stats(2.000_001, 5.0, 1.0), params),
            HairRasterPath::ThickHardware
        );
    }

    #[test]
    fn coverage_at_threshold_survives_but_below_is_culled() {
        let params = HairRasterParams {
            min_coverage: 0.25,
            software_width_threshold: 2.0,
        };
        // Coverage equal to the minimum is kept (cull test is strict `<`).
        assert_eq!(
            classify_hair_raster(stats(0.5, 5.0, 0.25), params),
            HairRasterPath::SubpixelSoftware
        );
        // Just below the minimum is culled.
        assert_eq!(
            classify_hair_raster(stats(0.5, 5.0, 0.249), params),
            HairRasterPath::Culled
        );
    }

    #[test]
    fn negative_params_clamp_to_zero() {
        // Negative min_coverage acts as zero: only truly empty coverage culls.
        // Negative width threshold acts as zero: nothing takes the software path.
        let params = HairRasterParams {
            min_coverage: -1.0,
            software_width_threshold: -1.0,
        };
        assert_eq!(
            classify_hair_raster(stats(0.0, 5.0, 0.0), params),
            HairRasterPath::SubpixelSoftware
        );
        assert_eq!(
            classify_hair_raster(stats(0.1, 5.0, 0.5), params),
            HairRasterPath::ThickHardware
        );
    }

    #[test]
    fn bins_route_each_segment_to_its_path() {
        // stat 0: thin -> software; stat 1: thick -> hardware; stat 2: culled.
        let segments = [
            HairSegmentRef::new(0, 0, 0),
            HairSegmentRef::new(0, 1, 1),
            HairSegmentRef::new(1, 0, 2),
        ];
        let stats = [
            stats(0.5, 10.0, 0.9),
            stats(8.0, 10.0, 0.9),
            stats(0.5, 10.0, 0.0),
        ];
        let bins = bin_hair_raster(&segments, &stats, HairRasterParams::default());
        assert_eq!(bins.subpixel_software, vec![HairSegmentRef::new(0, 0, 0)]);
        assert_eq!(bins.thick_hardware, vec![HairSegmentRef::new(0, 1, 1)]);
        assert_eq!(bins.culled, vec![HairSegmentRef::new(1, 0, 2)]);
        assert_eq!(bins.total(), 3);
    }

    #[test]
    fn bins_preserve_input_order_within_a_bucket() {
        let segments = [
            HairSegmentRef::new(2, 0, 0),
            HairSegmentRef::new(0, 0, 0),
            HairSegmentRef::new(1, 0, 0),
        ];
        let stats = [stats(0.5, 10.0, 0.9)];
        let bins = bin_hair_raster(&segments, &stats, HairRasterParams::default());
        assert_eq!(
            bins.subpixel_software,
            vec![
                HairSegmentRef::new(2, 0, 0),
                HairSegmentRef::new(0, 0, 0),
                HairSegmentRef::new(1, 0, 0),
            ]
        );
        assert_eq!(bins.bucket(HairRasterPath::SubpixelSoftware).len(), 3);
    }

    #[test]
    fn bins_skip_out_of_range_stats_indices() {
        // stat index 5 has no entry and must be dropped, not panic.
        let segments = [HairSegmentRef::new(0, 0, 0), HairSegmentRef::new(0, 1, 5)];
        let stats = [stats(0.5, 10.0, 0.9)];
        let bins = bin_hair_raster(&segments, &stats, HairRasterParams::default());
        assert_eq!(bins.total(), 1);
        assert_eq!(bins.subpixel_software, vec![HairSegmentRef::new(0, 0, 0)]);
    }

    #[test]
    fn empty_input_is_empty_bins() {
        let bins = bin_hair_raster(&[], &[], HairRasterParams::default());
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn bucket_view_matches_pushed_segments() {
        let mut bins = HairRasterBins::default();
        let seg = HairSegmentRef::new(3, 4, 0);
        bins.push(seg, HairRasterPath::ThickHardware);
        assert_eq!(bins.bucket(HairRasterPath::ThickHardware), &[seg]);
        assert!(bins.bucket(HairRasterPath::SubpixelSoftware).is_empty());
        assert!(bins.bucket(HairRasterPath::Culled).is_empty());
    }

    #[test]
    fn default_abi_is_deterministic_and_documented() {
        let abi = HairSoftRasterAbi::default();
        assert_eq!(abi.subpixel_samples, 8);
        assert_eq!(abi.tile_size_px, 16);
        assert_eq!(abi.max_segments_per_tile, 256);
    }
}
