//! Sorting and culling decisions: sort-key quantization, the blend-driven
//! strategy matrix, frustum/distance/`HZB` culling, bounds reduction, and
//! significance sleeping.
//!
//! Before a renderer composites its particles it decides *whether* to sort and
//! *how* (design §12), *which* particles survive frustum, distance, and `HZB`
//! culling (design §13), and *whether* a barely-visible, barely-moving emitter
//! can sleep to save simulation. This module is the deterministic `CPU`
//! reference for those decisions and the small math they need — a 16-bit depth
//! key, an axis-aligned bounds reduction, and inward-pointing frustum planes.
//!
//! The actual radix/bitonic sort kernels and the `HZB` depth-pyramid sample run
//! on the `GPU` and are pending the backend; this layer picks the strategy and
//! evaluates the `CPU`-checkable geometry, taking the `GPU` occlusion result as
//! an input where it cannot compute it. Only `sqrt` (through [`Vec3`]) and
//! ordinary arithmetic are used.

use super::{SortStrategy, Vec3};

/// GPU-visible sort item ABI. `depth_key` is the sortable quantized depth;
/// `particle_index` is the stable payload used to gather SoA attributes. The
/// explicit reserved word keeps the record 8-byte aligned for storage-buffer
/// implementations while remaining exactly 8 bytes on CPU and GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSortItem {
    /// Stable original pool slot.
    pub particle_index: u32,
    /// Quantized depth key, already oriented for the requested order.
    pub depth_key: u16,
    /// Must be zero; reserved for deterministic tie-break policy evolution.
    pub reserved: u16,
}

/// GPU cull input ABI. This is 32 bytes and uses scalar fields rather than
/// backend-specific vector alignment, making offsets portable across WGSL
/// storage layouts and host uploads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuCullSphere {
    /// World-space center.
    pub center: [f32; 3],
    /// Non-negative sphere radius.
    pub radius: f32,
    /// Original pool slot, preserved in the compacted visible list.
    pub particle_index: u32,
    /// Reserved flags, required to be zero in ABI v1.
    pub flags: u32,
    /// Reserved words keep 16-byte array stride.
    pub reserved: [u32; 2],
}

/// Checked upper-bound resource layout for sort/cull passes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SortCullBufferLayout {
    /// Capacity of source and destination key arrays.
    pub capacity: u32,
    /// Number of radix histogram bins per workgroup.
    pub radix_bins: u32,
    /// Workgroup width used to compute histogram count.
    pub workgroup_size: u32,
    /// Bytes in one input or output item array.
    pub item_bytes: u64,
    /// Bytes in one radix histogram scratch allocation.
    pub histogram_bytes: u64,
    /// Bytes for compacted visible indices.
    pub visible_index_bytes: u64,
}

/// Layout validation failure, reported before GPU allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SortCullLayoutError {
    /// Zero capacity cannot describe a dispatchable buffer.
    ZeroCapacity,
    /// Workgroup width or radix bin count must be nonzero and radix bins must
    /// be a power of two.
    InvalidRadixConfiguration,
    /// Byte arithmetic overflowed the host/GPU addressable range.
    SizeOverflow,
    /// Allocation exceeds the device's per-buffer limit.
    ExceedsDeviceLimit,
}

/// Derives checked storage requirements for stable sort and cull compaction.
/// The histogram scratch bound assumes one histogram per workgroup and one
/// 32-bit counter per bin; callers may reuse it between radix digit passes.
pub fn sort_cull_buffer_layout(
    capacity: u32,
    radix_bins: u32,
    workgroup_size: u32,
    max_buffer_bytes: u64,
) -> Result<SortCullBufferLayout, SortCullLayoutError> {
    if capacity == 0 {
        return Err(SortCullLayoutError::ZeroCapacity);
    }
    if radix_bins == 0 || !radix_bins.is_power_of_two() || workgroup_size == 0 {
        return Err(SortCullLayoutError::InvalidRadixConfiguration);
    }
    let capacity = u64::from(capacity);
    let groups = capacity
        .checked_add(u64::from(workgroup_size) - 1)
        .ok_or(SortCullLayoutError::SizeOverflow)?
        / u64::from(workgroup_size);
    let item_bytes = capacity
        .checked_mul(core::mem::size_of::<GpuSortItem>() as u64)
        .ok_or(SortCullLayoutError::SizeOverflow)?;
    let histogram_bytes = groups
        .checked_mul(u64::from(radix_bins))
        .and_then(|v| v.checked_mul(core::mem::size_of::<u32>() as u64))
        .ok_or(SortCullLayoutError::SizeOverflow)?;
    let visible_index_bytes = capacity
        .checked_mul(core::mem::size_of::<u32>() as u64)
        .ok_or(SortCullLayoutError::SizeOverflow)?;
    if item_bytes > max_buffer_bytes
        || histogram_bytes > max_buffer_bytes
        || visible_index_bytes > max_buffer_bytes
    {
        return Err(SortCullLayoutError::ExceedsDeviceLimit);
    }
    Ok(SortCullBufferLayout {
        capacity: capacity as u32,
        radix_bins,
        workgroup_size,
        item_bytes,
        histogram_bytes,
        visible_index_bytes,
    })
}

/// Rejects non-finite or negative bounds before culling or GPU upload.
#[must_use]
pub fn valid_cull_sphere(center: Vec3, radius: f32) -> bool {
    center.x.is_finite()
        && center.y.is_finite()
        && center.z.is_finite()
        && radius.is_finite()
        && radius >= 0.0
}

/// Quantizes a view-space depth to a 16-bit sort key.
///
/// `depth` is clamped to `[near, far]` and mapped linearly to `0..=65535`; a
/// degenerate range (`far <= near`) yields `0` so the key is always defined.
/// Nearer particles get smaller keys, so an ascending sort is front-to-back.
#[must_use]
pub fn quantize_depth(depth: f32, near: f32, far: f32) -> u16 {
    if far <= near {
        return 0;
    }
    let clamped = depth.max(near).min(far);
    let t = (clamped - near) / (far - near);
    (t * 65535.0 + 0.5) as u16
}

/// Depth sort key oriented for a blend mode.
///
/// Front-to-back (`back_to_front = false`) keeps the quantized depth so an
/// ascending sort draws near particles first (opaque/depth-tested). Back-to-
/// front inverts the key so the same ascending sort draws far particles first,
/// which is correct for alpha blending.
#[must_use]
pub fn sort_key(depth: f32, near: f32, far: f32, back_to_front: bool) -> u16 {
    let q = quantize_depth(depth, near, far);
    if back_to_front {
        u16::MAX - q
    } else {
        q
    }
}

/// How a renderer's particles blend into the frame, which decides whether a
/// depth sort is even needed (design §12).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendMode {
    /// Opaque: depth-tested, order-independent.
    Opaque,
    /// Alpha-tested cutout: depth-tested and order-independent.
    AlphaMask,
    /// Additive: commutative, order-independent.
    Additive,
    /// Premultiplied alpha over an additive-safe pipeline: order-independent.
    Premultiplied,
    /// Straight alpha blending: order-*dependent*, needs a back-to-front sort.
    AlphaBlend,
}

impl BlendMode {
    /// Returns `true` when correct compositing requires an explicit depth sort.
    #[must_use]
    pub fn needs_sort(self) -> bool {
        matches!(self, BlendMode::AlphaBlend)
    }
}

/// Inputs to the sort-strategy decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SortDecision {
    /// The renderer's blend mode.
    pub blend: BlendMode,
    /// Live particle count to sort.
    pub particle_count: u32,
    /// At or above this count a standalone sort uses radix; below it, bitonic.
    pub radix_min_count: u32,
    /// Route order-dependent blends through the scene's shared `OIT` path
    /// instead of a standalone per-emitter sort.
    pub prefer_shared_oit: bool,
}

/// Chooses the sort strategy for a renderer (design §12).
///
/// Order-independent blends (opaque/additive/premultiplied) never sort and
/// return [`SortStrategy::None`]. An order-dependent blend routes to
/// [`SortStrategy::SharedOit`] when the caller prefers the shared path, else to
/// a standalone [`SortStrategy::ViewDepthRadix`] for large counts or
/// [`SortStrategy::ViewDepthBitonic`] for small ones. An empty emitter also
/// needs no sort.
#[must_use]
pub fn choose_sort_strategy(decision: SortDecision) -> SortStrategy {
    if !decision.blend.needs_sort() || decision.particle_count <= 1 {
        return SortStrategy::None;
    }
    if decision.prefer_shared_oit {
        return SortStrategy::SharedOit;
    }
    if decision.particle_count >= decision.radix_min_count {
        SortStrategy::ViewDepthRadix
    } else {
        SortStrategy::ViewDepthBitonic
    }
}

/// An axis-aligned bounding box over particle positions.
///
/// The empty box has `min = +inf` and `max = -inf`, so [`Aabb::expand`] with the
/// first point yields a degenerate-but-valid box at that point and any union
/// with the empty box is the other box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// The empty box (inverted infinities), the identity for [`Aabb::union`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// A zero-volume box at a single point.
    #[must_use]
    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// Grows the box to include `p`.
    #[must_use]
    pub fn expand(self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    /// The smallest box containing both boxes.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns `true` when the box encloses a non-negative volume (i.e. it has
    /// received at least one point).
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    /// Center point (defined only for a valid box).
    #[must_use]
    pub fn center(self) -> Vec3 {
        self.min.add(self.max).scale(0.5)
    }

    /// Half the diagonal extent (defined only for a valid box).
    #[must_use]
    pub fn half_extents(self) -> Vec3 {
        self.max.sub(self.min).scale(0.5)
    }

    /// Radius of the bounding sphere sharing the box center.
    #[must_use]
    pub fn bounding_radius(self) -> f32 {
        self.half_extents().length()
    }
}

/// Reduces particle positions to their axis-aligned bounds.
///
/// Returns [`Aabb::empty`] for an empty slice, so a downstream union with a
/// per-emitter box is a no-op. This is the `CPU` reference for the parallel
/// bounds reduction a `GPU` pass performs (design §13).
#[must_use]
pub fn reduce_bounds(positions: &[Vec3]) -> Aabb {
    let mut bounds = Aabb::empty();
    for &p in positions {
        bounds = bounds.expand(p);
    }
    bounds
}

/// An inward-pointing frustum plane: points with `normal · p + d >= 0` are on
/// the inside half-space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit-length inward normal.
    pub normal: Vec3,
    /// Signed offset so `normal · p + d` is the signed inside distance.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from an inward normal and offset.
    #[must_use]
    pub fn new(normal: Vec3, d: f32) -> Self {
        Self { normal, d }
    }

    /// Signed distance from `p` to the plane; positive is inside.
    #[must_use]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }
}

/// A view frustum as six inward-pointing planes (left/right/bottom/top/near/far).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    /// The six bounding planes.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Builds a frustum from six inward-pointing planes.
    #[must_use]
    pub fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Verifies finite plane coefficients and approximately unit-length
    /// normals. Construction remains cheap; validation is performed once at
    /// the frame boundary before the frustum is uploaded or used for culling.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.planes.iter().all(|plane| {
            let n = plane.normal;
            let len2 = n.length_squared();
            n.x.is_finite()
                && n.y.is_finite()
                && n.z.is_finite()
                && plane.d.is_finite()
                && (len2 - 1.0).abs() <= 1e-3
        })
    }

    /// Returns `true` when the sphere is not fully outside any plane.
    ///
    /// A sphere is culled only when it lies entirely beyond one plane
    /// (`signed_distance < -radius`); a sphere straddling a plane is kept, which
    /// is the conservative choice culling requires (never drop a visible
    /// particle).
    #[must_use]
    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        for plane in self.planes {
            if plane.signed_distance(center) < -radius {
                return false;
            }
        }
        true
    }
}

/// Why a particle (or emitter bounds) was kept or culled.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CullReason {
    /// Passed every enabled test.
    Visible,
    /// Fully outside the view frustum.
    OutsideFrustum,
    /// Farther than the cull distance.
    BeyondDistance,
    /// Reported occluded by the `HZB` depth pyramid.
    HzbOccluded,
    /// Invalid/non-finite sphere data; rejected before GPU upload.
    InvalidBounds,
}

/// The result of a cull test.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CullDecision {
    /// Whether the particle survives.
    pub visible: bool,
    /// The first test that rejected it, or [`CullReason::Visible`].
    pub reason: CullReason,
}

impl CullDecision {
    /// A visible result.
    #[must_use]
    pub fn visible() -> Self {
        Self {
            visible: true,
            reason: CullReason::Visible,
        }
    }

    /// A culled result with a reason.
    #[must_use]
    pub fn culled(reason: CullReason) -> Self {
        Self {
            visible: false,
            reason,
        }
    }
}

/// Frustum-and-distance cull test for a bounding sphere (design §13).
///
/// Distance is tested first (cheapest), then the frustum. A non-positive
/// `max_distance` disables the distance test. The `HZB` stage is applied
/// separately by [`apply_hzb`] because occlusion is resolved on the `GPU`.
#[must_use]
pub fn cull_sphere(
    frustum: Frustum,
    camera_position: Vec3,
    center: Vec3,
    radius: f32,
    max_distance: f32,
) -> CullDecision {
    if !valid_cull_sphere(center, radius)
        || !frustum.is_valid()
        || !camera_position.x.is_finite()
        || !camera_position.y.is_finite()
        || !camera_position.z.is_finite()
    {
        return CullDecision::culled(CullReason::InvalidBounds);
    }
    if max_distance > 0.0 {
        let cull_at = max_distance + radius;
        if camera_position.distance_squared(center) > cull_at * cull_at {
            return CullDecision::culled(CullReason::BeyondDistance);
        }
    }
    if !frustum.intersects_sphere(center, radius) {
        return CullDecision::culled(CullReason::OutsideFrustum);
    }
    CullDecision::visible()
}

/// Folds a `GPU` `HZB` occlusion result into a prior cull decision.
///
/// Only a currently-visible particle can be occlusion-culled; an already-culled
/// decision is returned unchanged so the earliest (cheapest) reason is kept.
/// The `HZB` depth-pyramid sample itself is pending the `GPU` backend, so the
/// `occluded` flag is supplied by the caller.
#[must_use]
pub fn apply_hzb(decision: CullDecision, occluded: bool) -> CullDecision {
    if decision.visible && occluded {
        CullDecision::culled(CullReason::HzbOccluded)
    } else {
        decision
    }
}

/// Significance of an emitter this frame, blending screen coverage and motion.
///
/// Both terms are normalized to `0..=1` and the larger wins: a still but large
/// emitter stays significant, and a small but fast one does too. `speed_ref` is
/// the speed treated as fully significant; a non-positive reference ignores the
/// motion term.
#[must_use]
#[expect(
    clippy::manual_clamp,
    reason = "The chained max/min deliberately flushes a NaN input to a defined bound (NaN.max(0.0) = 0.0), whereas f32::clamp would propagate NaN into the significance."
)]
pub fn significance(screen_coverage: f32, speed: f32, speed_ref: f32) -> f32 {
    let coverage = screen_coverage.max(0.0).min(1.0);
    let motion = if speed_ref > 0.0 {
        (speed / speed_ref).max(0.0).min(1.0)
    } else {
        0.0
    };
    coverage.max(motion)
}

/// Parameters controlling significance-based sleeping (design §13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepParams {
    /// At or below this significance an emitter accrues idle frames.
    pub sleep_below: f32,
    /// Consecutive idle frames required before the emitter sleeps.
    pub frames_to_sleep: u32,
}

/// Sleep bookkeeping for one emitter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SleepState {
    /// Consecutive frames spent at or below the sleep threshold.
    pub idle_frames: u32,
    /// Whether the emitter is currently asleep (simulation skipped).
    pub asleep: bool,
}

/// Advances an emitter's sleep state by one frame.
///
/// Significance above the threshold wakes the emitter immediately and resets
/// the idle counter; significance at or below it accumulates idle frames and
/// sleeps once `frames_to_sleep` is reached. Waking is instantaneous so a
/// suddenly-visible effect never misses a frame.
#[must_use]
pub fn update_sleep(state: SleepState, significance: f32, params: SleepParams) -> SleepState {
    if significance > params.sleep_below {
        return SleepState {
            idle_frames: 0,
            asleep: false,
        };
    }
    let idle_frames = state.idle_frames.saturating_add(1);
    SleepState {
        idle_frames,
        asleep: idle_frames >= params.frames_to_sleep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_sort_and_cull_abis_have_stable_sizes_and_checked_capacity() {
        assert_eq!(core::mem::size_of::<GpuSortItem>(), 8);
        assert_eq!(core::mem::size_of::<GpuCullSphere>(), 32);
        let layout = sort_cull_buffer_layout(1025, 256, 128, 1 << 20).expect("fits");
        assert_eq!(layout.item_bytes, 1025 * 8);
        assert_eq!(layout.visible_index_bytes, 1025 * 4);
        assert_eq!(layout.histogram_bytes, 9 * 256 * 4);
        assert_eq!(
            sort_cull_buffer_layout(0, 256, 128, 1 << 20),
            Err(SortCullLayoutError::ZeroCapacity)
        );
        assert_eq!(
            sort_cull_buffer_layout(1, 255, 128, 1 << 20),
            Err(SortCullLayoutError::InvalidRadixConfiguration)
        );
        assert_eq!(
            sort_cull_buffer_layout(4096, 256, 128, 1024),
            Err(SortCullLayoutError::ExceedsDeviceLimit)
        );
    }

    #[test]
    fn invalid_spheres_are_rejected_before_frustum_or_hzb() {
        let frustum = Frustum::new([Plane::new(Vec3::new(1.0, 0.0, 0.0), 1.0); 6]);
        for (center, radius) in [
            (Vec3::ZERO, -1.0),
            (Vec3::new(f32::NAN, 0.0, 0.0), 1.0),
            (Vec3::ZERO, f32::INFINITY),
        ] {
            let result = cull_sphere(frustum, Vec3::ZERO, center, radius, 100.0);
            assert_eq!(result, CullDecision::culled(CullReason::InvalidBounds));
            assert_eq!(apply_hzb(result, true), result);
        }
        assert!(!valid_cull_sphere(Vec3::ZERO, -0.01));
        assert!(valid_cull_sphere(Vec3::ZERO, 0.0));
    }

    #[test]
    fn depth_quantization_spans_the_range_and_clamps() {
        assert_eq!(quantize_depth(0.0, 0.0, 100.0), 0);
        assert_eq!(quantize_depth(100.0, 0.0, 100.0), u16::MAX);
        // Midpoint rounds to the middle of the range.
        assert_eq!(quantize_depth(50.0, 0.0, 100.0), 32768);
        // Out of range clamps to the ends.
        assert_eq!(quantize_depth(-10.0, 0.0, 100.0), 0);
        assert_eq!(quantize_depth(1000.0, 0.0, 100.0), u16::MAX);
        // Degenerate range is defined.
        assert_eq!(quantize_depth(5.0, 10.0, 10.0), 0);
    }

    #[test]
    fn back_to_front_key_inverts_ordering() {
        let near_key = sort_key(10.0, 0.0, 100.0, true);
        let far_key = sort_key(90.0, 0.0, 100.0, true);
        // Back-to-front: the far particle must sort first (smaller key).
        assert!(far_key < near_key);
        // Front-to-back keeps natural depth order.
        assert!(sort_key(10.0, 0.0, 100.0, false) < sort_key(90.0, 0.0, 100.0, false));
    }

    #[test]
    fn blend_modes_report_sort_need() {
        assert!(!BlendMode::Opaque.needs_sort());
        assert!(!BlendMode::Additive.needs_sort());
        assert!(!BlendMode::Premultiplied.needs_sort());
        assert!(BlendMode::AlphaBlend.needs_sort());
    }

    fn decision(blend: BlendMode, count: u32, shared: bool) -> SortDecision {
        SortDecision {
            blend,
            particle_count: count,
            radix_min_count: 1000,
            prefer_shared_oit: shared,
        }
    }

    #[test]
    fn order_independent_blends_never_sort() {
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::Additive, 100_000, false)),
            SortStrategy::None
        );
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::Opaque, 100_000, true)),
            SortStrategy::None
        );
    }

    #[test]
    fn alpha_blend_picks_radix_bitonic_or_shared() {
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::AlphaBlend, 5000, false)),
            SortStrategy::ViewDepthRadix
        );
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::AlphaBlend, 100, false)),
            SortStrategy::ViewDepthBitonic
        );
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::AlphaBlend, 5000, true)),
            SortStrategy::SharedOit
        );
        // A single (or empty) particle never needs a sort.
        assert_eq!(
            choose_sort_strategy(decision(BlendMode::AlphaBlend, 1, false)),
            SortStrategy::None
        );
    }

    #[test]
    fn empty_bounds_is_invalid_and_neutral() {
        let empty = reduce_bounds(&[]);
        assert!(!empty.is_valid());
        // Union with the empty box is the other box.
        let b = Aabb::from_point(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(empty.union(b), b);
        assert_eq!(b.union(empty), b);
    }

    #[test]
    fn bounds_reduction_covers_all_points() {
        let positions = [
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(3.0, -4.0, 0.0),
            Vec3::new(0.0, 5.0, -6.0),
        ];
        let bounds = reduce_bounds(&positions);
        assert!(bounds.is_valid());
        assert_eq!(bounds.min, Vec3::new(-1.0, -4.0, -6.0));
        assert_eq!(bounds.max, Vec3::new(3.0, 5.0, 2.0));
        assert_eq!(bounds.center(), Vec3::new(1.0, 0.5, -2.0));
    }

    fn unit_box_frustum() -> Frustum {
        // A box-shaped "frustum" spanning [-1, 1] on every axis, planes facing
        // inward.
        Frustum::new([
            Plane::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            Plane::new(Vec3::new(-1.0, 0.0, 0.0), 1.0),
            Plane::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            Plane::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
            Plane::new(Vec3::new(0.0, 0.0, 1.0), 1.0),
            Plane::new(Vec3::new(0.0, 0.0, -1.0), 1.0),
        ])
    }

    #[test]
    fn frustum_keeps_inside_and_straddling_spheres() {
        let frustum = unit_box_frustum();
        assert!(frustum.intersects_sphere(Vec3::ZERO, 0.1));
        // Center just outside but radius straddles the plane: kept
        // (conservative).
        assert!(frustum.intersects_sphere(Vec3::new(1.2, 0.0, 0.0), 0.5));
        // Fully outside: culled.
        assert!(!frustum.intersects_sphere(Vec3::new(3.0, 0.0, 0.0), 0.5));
    }

    #[test]
    fn cull_reports_the_first_failing_test() {
        let frustum = unit_box_frustum();
        let cam = Vec3::new(0.0, 0.0, -10.0);
        // Inside, close: visible.
        let ok = cull_sphere(frustum, cam, Vec3::ZERO, 0.2, 100.0);
        assert_eq!(ok, CullDecision::visible());
        // Beyond distance is reported before frustum.
        let far = cull_sphere(frustum, cam, Vec3::new(0.0, 0.0, 90.0), 0.2, 50.0);
        assert_eq!(far.reason, CullReason::BeyondDistance);
        // Outside frustum, within distance.
        let out = cull_sphere(frustum, cam, Vec3::new(5.0, 0.0, 0.0), 0.2, 100.0);
        assert_eq!(out.reason, CullReason::OutsideFrustum);
    }

    #[test]
    fn hzb_only_culls_visible_particles() {
        let visible = CullDecision::visible();
        assert_eq!(
            apply_hzb(visible, true),
            CullDecision::culled(CullReason::HzbOccluded)
        );
        assert_eq!(apply_hzb(visible, false), visible);
        // An already-culled decision keeps its earlier reason.
        let out = CullDecision::culled(CullReason::OutsideFrustum);
        assert_eq!(apply_hzb(out, true), out);
    }

    #[test]
    fn significance_takes_the_dominant_term() {
        // Large but still.
        assert!((significance(0.8, 0.0, 10.0) - 0.8).abs() < 1e-6);
        // Small but fast.
        assert!((significance(0.1, 10.0, 10.0) - 1.0).abs() < 1e-6);
        // Clamped into range.
        assert!((significance(2.0, -5.0, 10.0) - 1.0).abs() < 1e-6);
        // Non-positive speed reference ignores motion.
        assert!((significance(0.3, 100.0, 0.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn sleep_accumulates_then_wakes_instantly() {
        let params = SleepParams {
            sleep_below: 0.1,
            frames_to_sleep: 3,
        };
        let mut state = SleepState::default();
        state = update_sleep(state, 0.05, params);
        assert!(!state.asleep);
        assert_eq!(state.idle_frames, 1);
        state = update_sleep(state, 0.05, params);
        state = update_sleep(state, 0.05, params);
        assert!(state.asleep);
        assert_eq!(state.idle_frames, 3);
        // A significant frame wakes it immediately.
        state = update_sleep(state, 0.9, params);
        assert!(!state.asleep);
        assert_eq!(state.idle_frames, 0);
    }
}
