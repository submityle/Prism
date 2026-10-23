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
use core::fmt;

/// GPU storage-buffer record consumed by `HairRasterSegment` in
/// `pkg/prism_render_scene/src/shaders/hair_raster.wesl`.
///
/// The two `vec2<f32>` fields have WGSL alignment 8; the remaining scalar
/// fields have alignment 4. This representation therefore has offsets
/// `0, 8, 16, 20, 24, 28` and a 32-byte array stride. Construct records with
/// [`ProjectedHairRasterSegment::try_new`] so coordinates and scalar values
/// satisfy the shader's viewport and numeric assumptions. Segments may have
/// endpoints outside the viewport; the shader clips their expanded bounds,
/// which is required for strands crossing a view edge.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(8))]
pub struct ProjectedHairRasterSegment {
    /// Projected start point in pixel coordinates.
    p0: [f32; 2],
    /// Projected end point in pixel coordinates.
    p1: [f32; 2],
    /// Normalized depth at `p0`, where zero is near and one is far.
    depth0: f32,
    /// Normalized depth at `p1`, where zero is near and one is far.
    depth1: f32,
    /// Projected strand diameter in pixels.
    width_px: f32,
    /// Packed strand/segment identity used by deferred shading.
    seg_id: u32,
}

impl ProjectedHairRasterSegment {
    /// Builds a GPU record after validating it against the target viewport.
    /// Endpoints may lie outside the viewport so segments crossing a view edge
    /// are retained; the shader clips its radius-expanded bounds. Coordinates
    /// and widths are bounded to keep the shader's float-to-i32 AABB conversion
    /// away from saturation.
    pub fn try_new(
        p0: [f32; 2],
        p1: [f32; 2],
        depth0: f32,
        depth1: f32,
        width_px: f32,
        seg_id: u32,
        viewport: [u32; 2],
    ) -> Result<Self, ProjectedHairRasterSegmentError> {
        if viewport[0] == 0 || viewport[1] == 0 {
            return Err(ProjectedHairRasterSegmentError::EmptyViewport);
        }
        // Keep dimensions exactly representable in f32: the shader uses them
        // for pixel-space clipping and u32 pixel addressing.
        if viewport[0] > (1 << 24) || viewport[1] > (1 << 24) {
            return Err(ProjectedHairRasterSegmentError::ViewportOutOfRange);
        }
        if !p0[0].is_finite()
            || !p0[1].is_finite()
            || !p1[0].is_finite()
            || !p1[1].is_finite()
            || !depth0.is_finite()
            || !depth1.is_finite()
            || !width_px.is_finite()
        {
            return Err(ProjectedHairRasterSegmentError::NonFiniteValue);
        }
        // hair_segment_bounds converts floor/ceil of endpoint +/- radius to
        // i32 before clipping. Leave generous headroom for the radius and its
        // one-pixel filter expansion, while retaining offscreen endpoints.
        let coordinate_limit = (1 << 30) as f32;
        if [p0, p1]
            .iter()
            .any(|p| p[0].abs() > coordinate_limit || p[1].abs() > coordinate_limit)
        {
            return Err(ProjectedHairRasterSegmentError::CoordinateOutOfRange);
        }
        if !(0.0..=1.0).contains(&depth0) || !(0.0..=1.0).contains(&depth1) {
            return Err(ProjectedHairRasterSegmentError::DepthOutOfRange);
        }
        if width_px < 0.0 {
            return Err(ProjectedHairRasterSegmentError::NegativeWidth);
        }
        // Bounds expand by half the diameter plus one pixel before conversion
        // to i32; reject giant radii instead of relying on shader saturation.
        if width_px > (1 << 24) as f32 {
            return Err(ProjectedHairRasterSegmentError::WidthOutOfRange);
        }
        Ok(Self {
            p0,
            p1,
            depth0,
            depth1,
            width_px,
            seg_id,
        })
    }

    /// Projected start point in pixel coordinates.
    #[must_use]
    pub const fn p0(&self) -> [f32; 2] {
        self.p0
    }

    /// Projected end point in pixel coordinates.
    #[must_use]
    pub const fn p1(&self) -> [f32; 2] {
        self.p1
    }

    /// Normalized depth at `p0`.
    #[must_use]
    pub const fn depth0(&self) -> f32 {
        self.depth0
    }

    /// Normalized depth at `p1`.
    #[must_use]
    pub const fn depth1(&self) -> f32 {
        self.depth1
    }

    /// Projected strand diameter in pixels.
    #[must_use]
    pub const fn width_px(&self) -> f32 {
        self.width_px
    }

    /// Packed strand/segment identity.
    #[must_use]
    pub const fn seg_id(&self) -> u32 {
        self.seg_id
    }

    /// Encodes the record as the shader's 32-byte storage-buffer layout.
    /// Float bit patterns and the identity are written little-endian, matching
    /// the WebGPU/WGSL host-shareable buffer representation.
    #[must_use]
    pub fn encode_le(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        bytes[0..4].copy_from_slice(&self.p0[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.p0[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.p1[0].to_le_bytes());
        bytes[12..16].copy_from_slice(&self.p1[1].to_le_bytes());
        bytes[16..20].copy_from_slice(&self.depth0.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.depth1.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.width_px.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.seg_id.to_le_bytes());
        bytes
    }
}

/// Projects a world-space strand segment into the compute-raster ABI.
///
/// The matrix is column-major and follows WGSL `clip_from_world * vec4(p, 1)`
/// convention. The segment is clipped against the homogeneous near/far planes
/// and positive-`w` camera plane before perspective division. Screen coordinates
/// are y-down pixels; NDC depth is preserved in the raster shader's
/// `0=near..1=far` convention. `radius_*` are world-space radii. Width uses a
/// conservative bound over the entire clipped segment, which avoids
/// under-covering when radius and perspective scale vary along its length.
#[must_use]
pub fn project_hair_raster_segment(
    clip_from_world: &[[f32; 4]; 4],
    world_a: [f32; 3],
    radius_a: f32,
    world_b: [f32; 3],
    radius_b: f32,
    viewport: [u32; 2],
    seg_id: u32,
) -> Result<Option<ProjectedHairRasterSegment>, ProjectedHairRasterSegmentError> {
    if viewport[0] == 0 || viewport[1] == 0 {
        return Err(ProjectedHairRasterSegmentError::EmptyViewport);
    }
    if viewport[0] > (1 << 24) || viewport[1] > (1 << 24) {
        return Err(ProjectedHairRasterSegmentError::ViewportOutOfRange);
    }
    if !radius_a.is_finite() || !radius_b.is_finite() {
        return Err(ProjectedHairRasterSegmentError::NonFiniteValue);
    }
    if radius_a > (1 << 24) as f32 || radius_b > (1 << 24) as f32 {
        return Err(ProjectedHairRasterSegmentError::WidthOutOfRange);
    }
    if radius_a < 0.0 || radius_b < 0.0 {
        return Err(ProjectedHairRasterSegmentError::NegativeWidth);
    }
    if !world_a.into_iter().chain(world_b).all(f32::is_finite)
        || !clip_from_world
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    {
        return Err(ProjectedHairRasterSegmentError::NonFiniteValue);
    }
    let project = |point: [f32; 3]| {
        let p = [point[0], point[1], point[2], 1.0];
        core::array::from_fn::<_, 4, _>(|row| {
            clip_from_world[0][row] * p[0]
                + clip_from_world[1][row] * p[1]
                + clip_from_world[2][row] * p[2]
                + clip_from_world[3][row] * p[3]
        })
    };
    let mut a = project(world_a);
    let mut b = project(world_b);
    let mut ra = radius_a;
    let mut rb = radius_b;
    // Clip against w>epsilon, z>=0, and z<=w. This is a parametric segment
    // clip in homogeneous space; applying each plane to the current endpoints
    // preserves perspective-correct endpoint depth after division.
    for plane in 0..3 {
        let distance = |v: [f32; 4]| match plane {
            0 => v[3] - 1.0e-6,
            1 => v[2],
            _ => v[3] - v[2],
        };
        let da = distance(a);
        let db = distance(b);
        if da < 0.0 && db < 0.0 {
            return Ok(None);
        }
        if da < 0.0 || db < 0.0 {
            let t = da / (da - db);
            let clipped = core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
            let clipped_radius = ra + (rb - ra) * t;
            if da < 0.0 {
                a = clipped;
                ra = clipped_radius;
            } else {
                b = clipped;
                rb = clipped_radius;
            }
        }
    }
    let to_screen = |v: [f32; 4]| {
        let inv_w = 1.0 / v[3];
        [
            (v[0] * inv_w * 0.5 + 0.5) * viewport[0] as f32,
            (0.5 - v[1] * inv_w * 0.5) * viewport[1] as f32,
            v[2] * inv_w,
        ]
    };
    let screen_a = to_screen(a);
    let screen_b = to_screen(b);
    // Each Jacobian numerator is affine along the clipped world-space segment,
    // so its absolute maximum is bounded by its endpoint maxima. `w` is also
    // affine and positive after clipping; its minimum is at an endpoint. Using
    // those bounds together with the maximum endpoint radius gives a valid
    // screen-space upper bound even when radius and perspective scale peak at
    // different points in the segment.
    let jacobian_numerator = |v: [f32; 4], axis: usize, row: usize| {
        clip_from_world[axis][row] * v[3] - clip_from_world[axis][3] * v[row]
    };
    let mut jacobian_bound_sq = 0.0;
    for axis in 0..3 {
        let nx = jacobian_numerator(a, axis, 0)
            .abs()
            .max(jacobian_numerator(b, axis, 0).abs())
            * viewport[0] as f32
            * 0.5;
        let ny = jacobian_numerator(a, axis, 1)
            .abs()
            .max(jacobian_numerator(b, axis, 1).abs())
            * viewport[1] as f32
            * 0.5;
        jacobian_bound_sq += nx * nx + ny * ny;
    }
    // Frobenius norm bounds the largest singular value of the 2x3 screen
    // Jacobian, including diagonal camera orientations.
    let min_w = a[3].min(b[3]);
    let max_radius = ra.max(rb);
    let width = max_radius * 2.0 * jacobian_bound_sq.sqrt() / (min_w * min_w);
    if !screen_a.into_iter().chain(screen_b).all(f32::is_finite) || !width.is_finite() {
        return Err(ProjectedHairRasterSegmentError::NonFiniteValue);
    }
    ProjectedHairRasterSegment::try_new(
        [screen_a[0], screen_a[1]],
        [screen_b[0], screen_b[1]],
        screen_a[2],
        screen_b[2],
        width,
        seg_id,
        viewport,
    )
    .map(Some)
}

/// Why a projected hair segment could not be encoded for the GPU rasterizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectedHairRasterSegmentError {
    /// A zero-sized render target cannot contain projected coordinates.
    EmptyViewport,
    /// Pixel dimensions exceed the exact integer range of f32 coordinates.
    ViewportOutOfRange,
    /// A coordinate, depth, or width was NaN or infinite.
    NonFiniteValue,
    /// An endpoint exceeds the safe finite range for shader pixel-bound math.
    CoordinateOutOfRange,
    /// Normalized depth was not in `0..=1`.
    DepthOutOfRange,
    /// Projected strand diameter was negative.
    NegativeWidth,
    /// Projected strand diameter exceeds the safe shader pixel-bound range.
    WidthOutOfRange,
}

impl fmt::Display for ProjectedHairRasterSegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyViewport => f.write_str("hair raster viewport must be non-zero"),
            Self::ViewportOutOfRange => {
                f.write_str("hair raster viewport exceeds shader i32 bounds")
            }
            Self::NonFiniteValue => f.write_str("hair raster segment contains a non-finite value"),
            Self::CoordinateOutOfRange => {
                f.write_str("hair raster endpoint exceeds shader pixel-bound range")
            }
            Self::DepthOutOfRange => f.write_str("hair raster depth must be in 0..=1"),
            Self::NegativeWidth => f.write_str("hair raster width must be non-negative"),
            Self::WidthOutOfRange => {
                f.write_str("hair raster width exceeds shader pixel-bound range")
            }
        }
    }
}

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
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn projected_segment_matches_wgsl_storage_layout() {
        assert_eq!(align_of::<ProjectedHairRasterSegment>(), 8);
        assert_eq!(size_of::<ProjectedHairRasterSegment>(), 32);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, p0), 0);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, p1), 8);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, depth0), 16);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, depth1), 20);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, width_px), 24);
        assert_eq!(offset_of!(ProjectedHairRasterSegment, seg_id), 28);
    }

    #[test]
    fn world_projection_maps_identity_and_computes_screen_width() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let segment = project_hair_raster_segment(
            &identity,
            [-0.5, 0.0, 0.25],
            0.1,
            [0.5, 0.0, 0.75],
            0.1,
            [100, 50],
            9,
        )
        .unwrap()
        .unwrap();
        assert_eq!(segment.p0(), [25.0, 25.0]);
        assert_eq!(segment.p1(), [75.0, 25.0]);
        assert_eq!(segment.depth0(), 0.25);
        assert_eq!(segment.depth1(), 0.75);
        assert!((segment.width_px() - 10.0_f32 * 1.25_f32.sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn world_projection_clips_near_plane_and_rejects_hidden_segments() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let clipped = project_hair_raster_segment(
            &identity,
            [0.0, 0.0, -1.0],
            0.1,
            [0.0, 0.0, 0.5],
            0.1,
            [64, 64],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(clipped.depth0(), 0.0);
        assert_eq!(clipped.depth1(), 0.5);
        assert_eq!(
            project_hair_raster_segment(
                &identity,
                [0.0, 0.0, -1.0],
                0.1,
                [0.0, 0.0, -0.5],
                0.1,
                [64, 64],
                1,
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn world_projection_applies_perspective_divide_and_depth_clipping() {
        // clip = (x, y, z - 1, z), so the near plane is z=1 and
        // perspective division makes equal world-space x values differ.
        let perspective = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ];
        let segment = project_hair_raster_segment(
            &perspective,
            [0.5, 0.0, 1.0],
            0.05,
            [0.5, 0.0, 2.0],
            0.05,
            [100, 100],
            3,
        )
        .unwrap()
        .unwrap();
        assert_eq!(segment.p0(), [75.0, 50.0]);
        assert_eq!(segment.p1(), [62.5, 50.0]);
        assert_eq!(segment.depth0(), 0.0);
        assert_eq!(segment.depth1(), 0.5);
        assert!(segment.width_px() > 2.5);
    }

    #[test]
    fn world_projection_rejects_viewport_dimensions_outside_shader_range() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            project_hair_raster_segment(
                &identity,
                [0.0; 3],
                0.1,
                [0.5, 0.0, 0.5],
                0.1,
                [(1 << 24) + 1, 64],
                0,
            ),
            Err(ProjectedHairRasterSegmentError::ViewportOutOfRange)
        );
    }

    #[test]
    fn projected_segment_constructor_accepts_viewport_boundary_and_offscreen_values() {
        let segment = ProjectedHairRasterSegment::try_new(
            [0.0, 0.0],
            [1920.0, 1080.0],
            0.0,
            1.0,
            0.75,
            0x1234_5678,
            [1920, 1080],
        )
        .unwrap();
        assert_eq!(segment.p0, [0.0, 0.0]);
        assert_eq!(segment.p1, [1920.0, 1080.0]);
        assert_eq!(segment.seg_id, 0x1234_5678);
        let crossing = ProjectedHairRasterSegment::try_new(
            [-5.0, 20.0],
            [30.0, 20.0],
            0.25,
            0.5,
            2.0,
            7,
            [16, 32],
        )
        .unwrap();
        assert_eq!(crossing.p0(), [-5.0, 20.0]);
        let encoded = segment.encode_le();
        assert_eq!(encoded.len(), 32);
        assert_eq!(&encoded[0..4], &0.0_f32.to_le_bytes());
        assert_eq!(&encoded[8..12], &1920.0_f32.to_le_bytes());
        assert_eq!(&encoded[16..20], &0.0_f32.to_le_bytes());
        assert_eq!(&encoded[20..24], &1.0_f32.to_le_bytes());
        assert_eq!(&encoded[24..28], &0.75_f32.to_le_bytes());
        assert_eq!(&encoded[28..32], &0x1234_5678_u32.to_le_bytes());
    }

    #[test]
    fn projected_segment_constructor_rejects_invalid_viewports_values_and_coordinates() {
        let make = |p0, p1, depth0, depth1, width, viewport| {
            ProjectedHairRasterSegment::try_new(p0, p1, depth0, depth1, width, 0, viewport)
        };
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 0.5, 1.0, [0, 1]),
            Err(ProjectedHairRasterSegmentError::EmptyViewport)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 0.5, 1.0, [(1 << 24) + 1, 1]),
            Err(ProjectedHairRasterSegmentError::ViewportOutOfRange)
        );
        assert_eq!(
            make([f32::NAN, 0.0], [1.0; 2], 0.5, 0.5, 1.0, [10, 10]),
            Err(ProjectedHairRasterSegmentError::NonFiniteValue)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], f32::INFINITY, 0.5, 1.0, [10, 10]),
            Err(ProjectedHairRasterSegmentError::NonFiniteValue)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 0.5, f32::NEG_INFINITY, [10, 10]),
            Err(ProjectedHairRasterSegmentError::NonFiniteValue)
        );
        assert_eq!(
            make(
                [-((1 << 30) as f32) - 256.0, 0.0],
                [1.0; 2],
                0.5,
                0.5,
                1.0,
                [10, 10]
            ),
            Err(ProjectedHairRasterSegmentError::CoordinateOutOfRange)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], -0.001, 0.5, 1.0, [10, 10]),
            Err(ProjectedHairRasterSegmentError::DepthOutOfRange)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 1.001, 1.0, [10, 10]),
            Err(ProjectedHairRasterSegmentError::DepthOutOfRange)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 0.5, -0.001, [10, 10]),
            Err(ProjectedHairRasterSegmentError::NegativeWidth)
        );
        assert_eq!(
            make([0.0; 2], [1.0; 2], 0.5, 0.5, (1 << 25) as f32, [10, 10]),
            Err(ProjectedHairRasterSegmentError::WidthOutOfRange)
        );
    }

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
