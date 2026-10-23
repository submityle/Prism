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

use super::groom_import::ResampledGroom;

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
        checked_hair_raster_pixel_count(viewport)?;
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
    checked_hair_raster_pixel_count(viewport)?;
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

/// Projects every adjacent control-point pair in a resampled groom.
///
/// Segment IDs are assigned in stable strand-major order, including segments
/// clipped from the view, so IDs remain deterministic as the camera moves.
/// The IDs address this returned vector's logical input order; callers that
/// compact `None` results must retain each record's `seg_id` rather than use
/// its compacted index. Publicly mutable groom fields are checked before any
/// indexing. The default empty groom is valid and produces no records.
pub fn project_resampled_groom(
    clip_from_world: &[[f32; 4]; 4],
    groom: &ResampledGroom,
    viewport: [u32; 2],
) -> Result<Vec<Option<ProjectedHairRasterSegment>>, ProjectedHairRasterSegmentError> {
    project_resampled_groom_with_id_base(clip_from_world, groom, viewport, 0)
}

/// Projects one groom while assigning IDs from a caller-owned global range.
///
/// Use a disjoint `first_segment_id` range when multiple grooms share one
/// visibility buffer; this prevents their strand-major local indices from
/// aliasing in the deferred-shading lookup. The caller owns range allocation
/// and must retain the same ID-to-groom/segment mapping through shading.
pub fn project_resampled_groom_with_id_base(
    clip_from_world: &[[f32; 4]; 4],
    groom: &ResampledGroom,
    viewport: [u32; 2],
    first_segment_id: u32,
) -> Result<Vec<Option<ProjectedHairRasterSegment>>, ProjectedHairRasterSegmentError> {
    if viewport[0] == 0 || viewport[1] == 0 {
        return Err(ProjectedHairRasterSegmentError::EmptyViewport);
    }
    if viewport[0] > (1 << 24) || viewport[1] > (1 << 24) {
        return Err(ProjectedHairRasterSegmentError::ViewportOutOfRange);
    }
    checked_hair_raster_pixel_count(viewport)?;
    let strand_count = groom.attributes.len();
    if strand_count == 0 {
        if !groom.positions.is_empty() {
            return Err(ProjectedHairRasterSegmentError::InvalidGroomLayout);
        }
        return Ok(Vec::new());
    }
    if groom.points_per_strand < 2
        || strand_count
            .checked_mul(groom.points_per_strand)
            .filter(|&expected| expected == groom.positions.len())
            .is_none()
    {
        return Err(ProjectedHairRasterSegmentError::InvalidGroomLayout);
    }
    let segments_per_strand = groom.points_per_strand - 1;
    let segment_count = strand_count
        .checked_mul(segments_per_strand)
        .ok_or(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)?;
    validate_projected_segment_range(segment_count, first_segment_id)?;
    let mut projected = Vec::new();
    projected
        .try_reserve_exact(segment_count)
        .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
    for (strand_index, attributes) in groom.attributes.iter().enumerate() {
        let base = strand_index * groom.points_per_strand;
        for segment_index in 0..segments_per_strand {
            let t0 = segment_index as f32 / segments_per_strand as f32;
            let t1 = (segment_index + 1) as f32 / segments_per_strand as f32;
            let local_segment_id = strand_index * segments_per_strand + segment_index;
            let seg_id = first_segment_id
                .checked_add(
                    u32::try_from(local_segment_id)
                        .map_err(|_| ProjectedHairRasterSegmentError::SegmentIdOutOfRange)?,
                )
                .ok_or(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)?;
            let a = groom.positions[base + segment_index];
            let b = groom.positions[base + segment_index + 1];
            projected.push(project_hair_raster_segment(
                clip_from_world,
                [a.x, a.y, a.z],
                attributes.radius_at(t0),
                [b.x, b.y, b.z],
                attributes.radius_at(t1),
                viewport,
                seg_id,
            )?);
        }
    }
    Ok(projected)
}

fn validate_projected_segment_range(
    segment_count: usize,
    first_segment_id: u32,
) -> Result<(), ProjectedHairRasterSegmentError> {
    // The dispatch ABI carries the record count in one u32. Even when the
    // final stable ID would fit (for example IDs 0..=u32::MAX), that many
    // records cannot be described by a u32 segment_count.
    u32::try_from(segment_count)
        .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
    if segment_count > 0
        && first_segment_id
            .checked_add(
                u32::try_from(segment_count - 1)
                    .map_err(|_| ProjectedHairRasterSegmentError::SegmentIdOutOfRange)?,
            )
            .is_none_or(|last_segment_id| last_segment_id == u32::MAX)
    {
        return Err(ProjectedHairRasterSegmentError::SegmentIdOutOfRange);
    }
    Ok(())
}

fn checked_hair_raster_pixel_count(
    viewport: [u32; 2],
) -> Result<u32, ProjectedHairRasterSegmentError> {
    viewport[0]
        .checked_mul(viewport[1])
        .filter(|&pixel_count| pixel_count > 0)
        .ok_or(ProjectedHairRasterSegmentError::ViewportPixelCountOutOfRange)
}

/// Packs projected Hair records into one contiguous little-endian upload slab.
///
/// The exact 32-byte shader ABI is preserved for every record. Capacity is
/// reserved once, avoiding per-record staging allocations; allocation or
/// address-space overflow is reported instead of panicking. Clipped records
/// are intentionally not accepted here: callers compact the `Some` results
/// from [`project_resampled_groom`] first, while retaining each record's stable
/// `seg_id`.
pub fn encode_projected_hair_segments_le(
    segments: &[ProjectedHairRasterSegment],
) -> Result<Vec<u8>, ProjectedHairRasterSegmentError> {
    let byte_len = segments
        .len()
        .checked_mul(32)
        .ok_or(ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
    for segment in segments {
        bytes.extend_from_slice(&segment.encode_le());
    }
    Ok(bytes)
}

/// One groom and its object-to-clip transform for a shared Hair visibility
/// buffer.
#[derive(Clone, Copy, Debug)]
pub struct HairRasterGroomInput<'a> {
    pub clip_from_world: &'a [[f32; 4]; 4],
    pub groom: &'a ResampledGroom,
}

/// Address range for one groom in a compact multi-groom segment upload.
///
/// `first_segment_id..first_segment_id + segment_count` describes the groom's
/// stable logical segment IDs, including segments clipped out of the view.
/// `upload_offset_bytes` and `visible_segment_count` address only records that
/// survived projection and are present in [`HairRasterUploadBatch::bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HairRasterGroomUpload {
    pub first_segment_id: u32,
    pub segment_count: u32,
    pub upload_offset_bytes: u64,
    pub visible_segment_count: u32,
}

/// Compact little-endian upload for multiple grooms sharing one visibility
/// buffer, with stable non-overlapping IDs and per-groom byte ranges.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HairRasterUploadBatch {
    pub bytes: Vec<u8>,
    pub grooms: Vec<HairRasterGroomUpload>,
}

fn next_hair_segment_id(
    first_segment_id: u32,
    segment_count: u32,
) -> Result<u32, ProjectedHairRasterSegmentError> {
    first_segment_id
        .checked_add(segment_count)
        .ok_or(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)
}

/// Projects and packs a deterministic multi-groom visibility input.
///
/// IDs are assigned in input order using each groom's full logical segment
/// count, so clipping never renumbers later segments or grooms. Visible records
/// are compacted into one upload slab while retaining their assigned IDs. This
/// avoids per-groom GPU buffers and gives downstream shading an exact mapping
/// from each groom's logical segment range to its compact upload subrange.
pub fn project_hair_raster_upload_batch(
    inputs: &[HairRasterGroomInput<'_>],
    viewport: [u32; 2],
) -> Result<HairRasterUploadBatch, ProjectedHairRasterSegmentError> {
    let mut batch = HairRasterUploadBatch {
        bytes: Vec::new(),
        grooms: Vec::new(),
    };
    batch
        .grooms
        .try_reserve_exact(inputs.len())
        .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;

    let mut first_segment_id = 0_u32;
    for input in inputs {
        let projected = project_resampled_groom_with_id_base(
            input.clip_from_world,
            input.groom,
            viewport,
            first_segment_id,
        )?;
        let segment_count = u32::try_from(projected.len())
            .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
        let visible_segment_count = u32::try_from(projected.iter().flatten().count())
            .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
        let upload_offset_bytes = u64::try_from(batch.bytes.len())
            .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
        let encoded_byte_len = usize::try_from(visible_segment_count)
            .ok()
            .and_then(|count| count.checked_mul(32))
            .ok_or(ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
        batch
            .bytes
            .try_reserve(encoded_byte_len)
            .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
        for segment in projected.into_iter().flatten() {
            batch.bytes.extend_from_slice(&segment.encode_le());
        }
        batch.grooms.push(HairRasterGroomUpload {
            first_segment_id,
            segment_count,
            upload_offset_bytes,
            visible_segment_count,
        });
        first_segment_id = next_hair_segment_id(first_segment_id, segment_count)?;
    }
    Ok(batch)
}

/// Why a projected hair segment could not be encoded for the GPU rasterizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectedHairRasterSegmentError {
    /// A zero-sized render target cannot contain projected coordinates.
    EmptyViewport,
    /// Pixel dimensions exceed the exact integer range of f32 coordinates.
    ViewportOutOfRange,
    /// Viewport area exceeds the shader's u32 pixel-address space.
    ViewportPixelCountOutOfRange,
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
    /// Resampled groom point and per-strand attribute arrays are inconsistent.
    InvalidGroomLayout,
    /// Stable strand-major segment identity does not fit in the GPU ABI.
    SegmentIdOutOfRange,
    /// Segment count cannot be represented by the immediate dispatch ABI.
    SegmentCountOutOfRange,
    /// The segment dispatch exceeds the device's one-dimensional workgroup limit.
    DispatchExceedsDeviceLimit,
    /// Raster parameters cannot be represented by the shader contract.
    InvalidRasterParams,
    /// Packed upload byte length overflows or cannot be reserved.
    UploadSizeOutOfRange,
}

impl fmt::Display for ProjectedHairRasterSegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyViewport => f.write_str("hair raster viewport must be non-zero"),
            Self::ViewportOutOfRange => {
                f.write_str("hair raster viewport exceeds shader i32 bounds")
            }
            Self::ViewportPixelCountOutOfRange => {
                f.write_str("hair raster viewport area exceeds u32 pixel addressing")
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
            Self::InvalidGroomLayout => {
                f.write_str("resampled groom point and attribute layout is inconsistent")
            }
            Self::SegmentIdOutOfRange => {
                f.write_str("hair raster segment identity exceeds the GPU ABI")
            }
            Self::SegmentCountOutOfRange => {
                f.write_str("hair raster segment count exceeds the immediate ABI")
            }
            Self::DispatchExceedsDeviceLimit => {
                f.write_str("hair raster segment dispatch exceeds the device workgroup limit")
            }
            Self::InvalidRasterParams => {
                f.write_str("hair raster parameters violate the shader ABI")
            }
            Self::UploadSizeOutOfRange => {
                f.write_str("hair raster upload size exceeds addressable memory")
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
/// Maximum sub-pixel samples accepted by one hair raster dispatch.
///
/// Each sample evaluates a point-to-segment distance in both winner and
/// coverage passes. Bounding this count prevents malformed runtime settings
/// from multiplying per-segment pixel work without limit.
pub const MAX_HAIR_RASTER_SUBPIXEL_SAMPLES: u32 = 64;

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

/// Five-word immediate parameter block consumed by `hair_raster.wesl`.
///
/// Its 20-byte little-endian representation is viewport width/height, coverage
/// epsilon, sample count, and segment count, in that order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct HairRasterDispatchParams {
    viewport: [u32; 2],
    coverage_epsilon: f32,
    subpixel_samples: u32,
    segment_count: u32,
}

impl HairRasterDispatchParams {
    /// Validates and constructs the immediate block for a raster dispatch.
    pub fn try_new(
        viewport: [u32; 2],
        abi: HairSoftRasterAbi,
        segment_count: usize,
    ) -> Result<Self, ProjectedHairRasterSegmentError> {
        if viewport[0] == 0 || viewport[1] == 0 {
            return Err(ProjectedHairRasterSegmentError::EmptyViewport);
        }
        if viewport[0] > (1 << 24) || viewport[1] > (1 << 24) {
            return Err(ProjectedHairRasterSegmentError::ViewportOutOfRange);
        }
        checked_hair_raster_pixel_count(viewport)?;
        let segment_count = u32::try_from(segment_count)
            .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
        if !abi.coverage_epsilon.is_finite()
            || !(0.0..=1.0).contains(&abi.coverage_epsilon)
            || abi.subpixel_samples == 0
            || !abi.subpixel_samples.is_power_of_two()
            || abi.subpixel_samples > MAX_HAIR_RASTER_SUBPIXEL_SAMPLES
        {
            return Err(ProjectedHairRasterSegmentError::InvalidRasterParams);
        }
        Ok(Self {
            viewport,
            coverage_epsilon: abi.coverage_epsilon,
            subpixel_samples: abi.subpixel_samples,
            segment_count,
        })
    }

    /// Encodes the WGSL immediate block as five little-endian 32-bit words.
    #[must_use]
    pub fn encode_le(self) -> [u8; 20] {
        let mut bytes = [0; 20];
        bytes[0..4].copy_from_slice(&self.viewport[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.viewport[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.coverage_epsilon.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.subpixel_samples.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.segment_count.to_le_bytes());
        bytes
    }

    /// Returns the one-dimensional workgroup count after checking the device
    /// limit. An empty groom requires no dispatch and returns zero.
    pub fn dispatch_workgroups(
        self,
        max_workgroups_per_dimension: u32,
    ) -> Result<u32, ProjectedHairRasterSegmentError> {
        let workgroups = self.segment_count.div_ceil(64);
        if workgroups > max_workgroups_per_dimension {
            return Err(ProjectedHairRasterSegmentError::DispatchExceedsDeviceLimit);
        }
        Ok(workgroups)
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
    fn raster_abi_rejects_viewport_area_outside_u32_pixel_addressing() {
        assert_eq!(
            ProjectedHairRasterSegment::try_new(
                [0.0, 0.0],
                [1.0, 1.0],
                0.0,
                1.0,
                1.0,
                0,
                [65_536, 65_536],
            ),
            Err(ProjectedHairRasterSegmentError::ViewportPixelCountOutOfRange)
        );
    }

    #[test]
    fn resampled_groom_projects_segments_with_stable_ids_and_radius_taper() {
        use super::super::dynamics::Vec3;
        use super::super::groom_import::StrandAttributes;

        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let groom = ResampledGroom {
            points_per_strand: 3,
            positions: vec![
                Vec3::new(-0.5, 0.0, 0.5),
                Vec3::new(0.0, 0.0, 0.5),
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(0.0, 0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
            ],
            attributes: vec![
                StrandAttributes::new(0.1, 0.05, [0.0; 2], 11),
                StrandAttributes::new(0.08, 0.02, [0.0; 2], 22),
            ],
        };
        let projected = project_resampled_groom(&identity, &groom, [100, 100]).unwrap();
        assert_eq!(projected.len(), 4);
        let records: Vec<_> = projected.into_iter().map(Option::unwrap).collect();
        assert_eq!(
            records.iter().map(|r| r.seg_id()).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!((records[0].width_px() - 2.0_f32.sqrt() * 10.0).abs() < 1.0e-4);
        assert!((records[1].width_px() - 2.0_f32.sqrt() * 7.5).abs() < 1.0e-4);
        assert!((records[2].width_px() - 2.0_f32.sqrt() * 8.0).abs() < 1.0e-4);
        assert!((records[3].width_px() - 2.0_f32.sqrt() * 5.0).abs() < 1.0e-4);
        let upload = encode_projected_hair_segments_le(&records).unwrap();
        assert_eq!(upload.len(), records.len() * 32);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(&upload[index * 32..(index + 1) * 32], &record.encode_le());
        }
        let rebased =
            project_resampled_groom_with_id_base(&identity, &groom, [100, 100], 40).unwrap();
        assert_eq!(
            rebased
                .iter()
                .map(|record| record.as_ref().unwrap().seg_id())
                .collect::<Vec<_>>(),
            [40, 41, 42, 43]
        );
        assert_eq!(
            project_resampled_groom_with_id_base(&identity, &groom, [100, 100], u32::MAX - 2,),
            Err(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)
        );
    }

    #[test]
    fn multi_groom_upload_keeps_stable_ids_when_segments_are_clipped() {
        use super::super::dynamics::Vec3;
        use super::super::groom_import::StrandAttributes;

        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let visible = ResampledGroom {
            points_per_strand: 3,
            positions: vec![
                Vec3::new(-0.5, 0.0, 0.5),
                Vec3::new(0.0, 0.0, 0.5),
                Vec3::new(0.5, 0.0, 0.5),
            ],
            attributes: vec![StrandAttributes::new(0.1, 0.05, [0.0; 2], 1)],
        };
        let clipped = ResampledGroom {
            points_per_strand: 3,
            positions: vec![
                Vec3::new(0.0, 0.0, -0.5),
                Vec3::new(0.0, 0.0, -0.25),
                Vec3::new(0.0, 0.0, 0.5),
            ],
            attributes: vec![StrandAttributes::new(0.1, 0.05, [0.0; 2], 2)],
        };
        let batch = project_hair_raster_upload_batch(
            &[
                HairRasterGroomInput {
                    clip_from_world: &identity,
                    groom: &visible,
                },
                HairRasterGroomInput {
                    clip_from_world: &identity,
                    groom: &clipped,
                },
            ],
            [64, 64],
        )
        .unwrap();

        assert_eq!(batch.grooms.len(), 2);
        assert_eq!(batch.grooms[0].first_segment_id, 0);
        assert_eq!(batch.grooms[0].segment_count, 2);
        assert_eq!(batch.grooms[0].upload_offset_bytes, 0);
        assert_eq!(batch.grooms[0].visible_segment_count, 2);
        assert_eq!(batch.grooms[1].first_segment_id, 2);
        assert_eq!(batch.grooms[1].segment_count, 2);
        assert_eq!(batch.grooms[1].upload_offset_bytes, 64);
        assert_eq!(batch.grooms[1].visible_segment_count, 1);
        assert_eq!(batch.bytes.len(), 96);
        assert_eq!(
            u32::from_le_bytes(batch.bytes[64 + 28..64 + 32].try_into().unwrap()),
            3
        );
    }

    #[test]
    fn multi_groom_upload_handles_empty_batch_and_checks_global_id_overflow() {
        use super::next_hair_segment_id;

        assert_eq!(
            project_hair_raster_upload_batch(&[], [64, 64]).unwrap(),
            HairRasterUploadBatch::default()
        );
        assert_eq!(next_hair_segment_id(u32::MAX - 1, 1), Ok(u32::MAX));
        assert_eq!(
            next_hair_segment_id(u32::MAX, 1),
            Err(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)
        );
    }

    #[test]
    fn resampled_groom_projection_rejects_malformed_public_layout() {
        let malformed = ResampledGroom {
            points_per_strand: 3,
            positions: vec![super::super::dynamics::Vec3::ZERO; 2],
            attributes: vec![super::super::groom_import::StrandAttributes::default()],
        };
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            project_resampled_groom(&identity, &malformed, [64, 64]),
            Err(ProjectedHairRasterSegmentError::InvalidGroomLayout)
        );
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn projected_segment_range_matches_count_and_id_abis() {
        assert_eq!(
            validate_projected_segment_range(u32::MAX as usize, 0),
            Ok(())
        );
        assert_eq!(
            validate_projected_segment_range(u32::MAX as usize + 1, 0),
            Err(ProjectedHairRasterSegmentError::SegmentCountOutOfRange)
        );
        assert_eq!(
            validate_projected_segment_range(2, u32::MAX),
            Err(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)
        );
        assert_eq!(
            validate_projected_segment_range(1, u32::MAX),
            Err(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)
        );
        assert_eq!(validate_projected_segment_range(1, u32::MAX - 1), Ok(()));
        assert_eq!(validate_projected_segment_range(0, u32::MAX), Ok(()));
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

    #[test]
    fn dispatch_params_match_the_twenty_byte_wgsl_immediate_layout() {
        let params = HairRasterDispatchParams::try_new(
            [1280, 720],
            HairSoftRasterAbi::default(),
            0x0102_0304,
        )
        .unwrap();
        assert_eq!(core::mem::size_of::<HairRasterDispatchParams>(), 20);
        assert_eq!(
            params.encode_le(),
            [
                0x00, 0x05, 0x00, 0x00, // viewport width
                0xD0, 0x02, 0x00, 0x00, // viewport height
                0x00, 0x00, 0x80, 0x3B, // 1/256 coverage epsilon
                0x08, 0x00, 0x00, 0x00, // sample count
                0x04, 0x03, 0x02, 0x01, // segment count
            ]
        );
        assert_eq!(params.dispatch_workgroups(300_000), Ok(264_205));
        assert_eq!(
            params.dispatch_workgroups(1_000),
            Err(ProjectedHairRasterSegmentError::DispatchExceedsDeviceLimit)
        );
    }

    #[test]
    fn dispatch_params_reject_invalid_epsilon_and_unaddressable_viewport() {
        let mut abi = HairSoftRasterAbi::default();
        abi.coverage_epsilon = f32::NAN;
        assert_eq!(
            HairRasterDispatchParams::try_new([1, 1], abi, 1),
            Err(ProjectedHairRasterSegmentError::InvalidRasterParams)
        );
        assert_eq!(
            HairRasterDispatchParams::try_new([65_536, 65_536], HairSoftRasterAbi::default(), 1),
            Err(ProjectedHairRasterSegmentError::ViewportPixelCountOutOfRange)
        );
        for samples in [0, 3, MAX_HAIR_RASTER_SUBPIXEL_SAMPLES * 2] {
            let mut abi = HairSoftRasterAbi::default();
            abi.subpixel_samples = samples;
            assert_eq!(
                HairRasterDispatchParams::try_new([1, 1], abi, 1),
                Err(ProjectedHairRasterSegmentError::InvalidRasterParams)
            );
        }
    }
}
