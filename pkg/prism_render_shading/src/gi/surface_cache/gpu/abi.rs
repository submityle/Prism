//! Host/device `ABI` for the surfel-allocation (atlas-addressing) producer.
//!
//! These `repr(C)` structs are the byte-exact layout the compute kernel
//! `shaders/surfel_alloc.wesl` reads and writes through its three `@group(0)`
//! bindings. The host fills a [`GpuSurfelAllocParams`] uniform plus a
//! `requests` storage array of [`GpuSurfelAllocRequest`] and the kernel writes
//! one [`GpuSurfelAllocSlot`] per request into a `read_write` storage array.
//!
//! # Layout discipline
//!
//! Every field is a 4-byte scalar (`u32` / `f32`) so the whole `ABI` is
//! naturally 4-byte aligned and needs no interior padding under `std430`: the
//! sample direction in [`GpuSurfelAllocRequest`] is carried as three scalars
//! rather than a `vec3` precisely to keep the storage stride at 16 bytes with a
//! 4-byte alignment (a `vec3<f32>` would force a 16-byte alignment and a padded
//! 16-byte stride, and worse, invite the classic `vec3` straddling bug). The
//! `const` size assertions below fail the build if the Rust layout ever drifts
//! from the stride constants the kernel and the host allocator assume.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; the layout is validated structurally through
//! `bytemuck`'s derived [`Pod`]/[`Zeroable`] impls (which reject padding) and
//! the compile-time size assertions, never through pointer casts.
//!
//! Provenance: standard octahedral surfel-atlas addressing `ABI`; no Unreal
//! Engine source or derived code.

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable};

use crate::gi::surface_cache::atlas::SurfelAtlas;
use crate::gi::surface_cache::integration::{SurfelCacheEntry, TemporalParams};
use crate::gi::surface_cache::surfel::{CoverageParams, Surfel};

/// Threads per workgroup for the surfel-allocation dispatch; mirrors the
/// `@workgroup_size(64)` in `shaders/surfel_alloc.wesl`.
pub const SURFEL_ALLOC_WORKGROUP_SIZE: u32 = 64;

/// `std430` byte size of [`GpuSurfelAllocParams`] (the uniform block).
pub const SURFEL_ALLOC_PARAMS_SIZE: usize = 16;

/// `std430` storage stride of one [`GpuSurfelAllocRequest`].
pub const SURFEL_ALLOC_REQUEST_STRIDE: usize = 16;

/// `std430` storage stride of one [`GpuSurfelAllocSlot`].
pub const SURFEL_ALLOC_SLOT_STRIDE: usize = 20;

/// Valid-slot bit inside [`GpuSurfelAllocSlot::flags`] (`bit 0`).
pub const SURFEL_ALLOC_FLAG_VALID: u32 = 1;

/// Uniform parameters for one surfel-allocation dispatch.
///
/// `repr(C)` `std430` uniform block mirrored by `struct Params` in
/// `shaders/surfel_alloc.wesl`. `count` is the number of live requests (threads
/// past it early-out); the remaining three fields are the atlas dimensions,
/// already clamped to at least `1` by [`SurfelAtlas::new`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfelAllocParams {
    /// Number of valid allocation requests in the `requests` buffer.
    pub count: u32,
    /// Tile columns across the atlas (`>= 1`).
    pub tiles_per_row: u32,
    /// Tile rows down the atlas (`>= 1`).
    pub tile_rows: u32,
    /// Side length in texels of each square tile (`>= 1`).
    pub tile_resolution: u32,
}

impl GpuSurfelAllocParams {
    /// Build the dispatch parameters from an atlas descriptor and a live
    /// request count, copying the atlas's already-clamped dimensions verbatim.
    #[must_use]
    pub fn from_atlas(atlas: &SurfelAtlas, count: u32) -> Self {
        Self {
            count,
            tiles_per_row: atlas.tiles_per_row,
            tile_rows: atlas.tile_rows,
            tile_resolution: atlas.tile_resolution,
        }
    }

    /// Number of workgroups needed to cover [`count`](Self::count) at
    /// [`SURFEL_ALLOC_WORKGROUP_SIZE`] threads each (ceil-divide).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.count.div_ceil(SURFEL_ALLOC_WORKGROUP_SIZE)
    }
}

/// One surfel-allocation request: the surfel id plus the sample direction.
///
/// `repr(C)` `std430` element mirrored by `struct Request` in
/// `shaders/surfel_alloc.wesl`. The direction is three scalars (not a `vec3`)
/// to keep the stride at [`SURFEL_ALLOC_REQUEST_STRIDE`] bytes with a 4-byte
/// alignment; it need not be normalised (the octahedral fold normalises via the
/// `L1` norm, and the zero vector maps to the `+z` pole).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelAllocRequest {
    /// Surfel id to allocate; ids at or past the atlas capacity yield an
    /// invalid slot.
    pub surfel_id: u32,
    /// Sample direction `x` component.
    pub dir_x: f32,
    /// Sample direction `y` component.
    pub dir_y: f32,
    /// Sample direction `z` component.
    pub dir_z: f32,
}

impl GpuSurfelAllocRequest {
    /// Build a request from a surfel id and a direction given as three scalars.
    #[must_use]
    pub fn new(surfel_id: u32, dir_x: f32, dir_y: f32, dir_z: f32) -> Self {
        Self {
            surfel_id,
            dir_x,
            dir_y,
            dir_z,
        }
    }
}

/// Resolved global atlas slot for one request.
///
/// `repr(C)` `std430` element mirrored by `struct Slot` in
/// `shaders/surfel_alloc.wesl`. `texel_x`/`texel_y` are the global (not
/// tile-local) atlas texel; `tile_col`/`tile_row` are the surfel's tile; and
/// `flags` bit [`SURFEL_ALLOC_FLAG_VALID`] is set when the id was in range. An
/// out-of-range id leaves every field zero (an all-zero slot is the canonical
/// "invalid" value, which is why [`Zeroable`] is sound here).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfelAllocSlot {
    /// Global atlas texel column.
    pub texel_x: u32,
    /// Global atlas texel row.
    pub texel_y: u32,
    /// Tile column of the surfel's tile.
    pub tile_col: u32,
    /// Tile row of the surfel's tile.
    pub tile_row: u32,
    /// Status flags; bit [`SURFEL_ALLOC_FLAG_VALID`] marks a resolved slot.
    pub flags: u32,
}

impl GpuSurfelAllocSlot {
    /// All-zero invalid slot (the value written for an out-of-range id).
    #[must_use]
    pub fn invalid() -> Self {
        Self::zeroed()
    }

    /// Whether the valid bit is set in [`flags`](Self::flags).
    #[must_use]
    pub fn valid(&self) -> bool {
        self.flags & SURFEL_ALLOC_FLAG_VALID != 0
    }
}

const _: () = assert!(size_of::<GpuSurfelAllocParams>() == SURFEL_ALLOC_PARAMS_SIZE);
const _: () = assert!(align_of::<GpuSurfelAllocParams>() == 4);
const _: () = assert!(size_of::<GpuSurfelAllocRequest>() == SURFEL_ALLOC_REQUEST_STRIDE);
const _: () = assert!(align_of::<GpuSurfelAllocRequest>() == 4);
const _: () = assert!(size_of::<GpuSurfelAllocSlot>() == SURFEL_ALLOC_SLOT_STRIDE);
const _: () = assert!(align_of::<GpuSurfelAllocSlot>() == 4);

// ---------------------------------------------------------------------------
// Surfel-update (temporal-integration) slice
// ---------------------------------------------------------------------------
//
// Host/device `ABI` for the surfel-update producer kernel
// `shaders/surfel_update.wesl`, the on-device twin of the `CPU` golden
// `integrate_radiance` in [`crate::gi::surface_cache::integration`]. The host
// fills a [`GpuSurfelUpdateParams`] uniform plus an `inputs` storage array of
// [`GpuSurfelUpdateInput`] and the kernel writes one [`GpuSurfelUpdateResult`]
// per input into a `read_write` storage array. As with the allocation slice,
// every field is a 4-byte scalar so the layout is naturally 4-byte aligned and
// `vec3` straddling cannot occur under `std430`.

/// Threads per workgroup for the surfel-update dispatch; mirrors the
/// `@workgroup_size(64)` in `shaders/surfel_update.wesl`.
pub const SURFEL_UPDATE_WORKGROUP_SIZE: u32 = 64;

/// `std430` byte size of [`GpuSurfelUpdateParams`] (the uniform block).
pub const SURFEL_UPDATE_PARAMS_SIZE: usize = 16;

/// `std430` storage stride of one [`GpuSurfelUpdateInput`].
pub const SURFEL_UPDATE_INPUT_STRIDE: usize = 88;

/// `std430` storage stride of one [`GpuSurfelUpdateResult`].
pub const SURFEL_UPDATE_RESULT_STRIDE: usize = 16;

/// Uniform parameters for one surfel-update dispatch.
///
/// `repr(C)` `std430` uniform block mirrored by `struct UpdateParams` in
/// `shaders/surfel_update.wesl`. `count` is the number of live inputs (threads
/// past it early-out); the remaining three fields are the temporal-integration
/// tunables copied verbatim from [`TemporalParams`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelUpdateParams {
    /// Number of valid update inputs in the `inputs` buffer.
    pub count: u32,
    /// Confidence cap; the exponential-moving-average weight floors at
    /// `1 / max(max_samples, 1)`.
    pub max_samples: u32,
    /// Anchor-displacement tolerance as a fraction of the surfel radius; a jump
    /// beyond `position_tolerance * radius` forces a disocclusion reset.
    pub position_tolerance: f32,
    /// Minimum `dot(prev_normal, curr_normal)` to keep the history; below this
    /// the entry resets.
    pub normal_tolerance: f32,
}

impl GpuSurfelUpdateParams {
    /// Build the dispatch parameters from the temporal tunables and a live
    /// input count, copying [`TemporalParams`] verbatim.
    #[must_use]
    pub fn from_temporal(params: &TemporalParams, count: u32) -> Self {
        Self {
            count,
            max_samples: params.max_samples,
            position_tolerance: params.position_tolerance,
            normal_tolerance: params.normal_tolerance,
        }
    }

    /// Number of workgroups needed to cover [`count`](Self::count) at
    /// [`SURFEL_UPDATE_WORKGROUP_SIZE`] threads each (ceil-divide).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.count.div_ceil(SURFEL_UPDATE_WORKGROUP_SIZE)
    }
}

/// One surfel-update input: the previous entry, the previous and current surfel
/// geometry, and the new radiance sample.
///
/// `repr(C)` `std430` element mirrored by `struct UpdateInput` in
/// `shaders/surfel_update.wesl`. Every vector is carried as separate scalar
/// components (not a `vec3`) to keep the stride at
/// [`SURFEL_UPDATE_INPUT_STRIDE`] bytes with a 4-byte alignment. When
/// `has_history` is zero the `prev_*` fields are ignored and the kernel reseeds
/// from the sample.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelUpdateInput {
    /// `1` when a previous entry/surfel is present, `0` on first sight.
    pub has_history: u32,
    /// Previous confidence (frame count) of the cached entry.
    pub prev_count: u32,
    /// Previous accumulated radiance `x` component.
    pub prev_radiance_x: f32,
    /// Previous accumulated radiance `y` component.
    pub prev_radiance_y: f32,
    /// Previous accumulated radiance `z` component.
    pub prev_radiance_z: f32,
    /// Previous surfel anchor `x` component.
    pub prev_pos_x: f32,
    /// Previous surfel anchor `y` component.
    pub prev_pos_y: f32,
    /// Previous surfel anchor `z` component.
    pub prev_pos_z: f32,
    /// Previous surfel normal `x` component.
    pub prev_normal_x: f32,
    /// Previous surfel normal `y` component.
    pub prev_normal_y: f32,
    /// Previous surfel normal `z` component.
    pub prev_normal_z: f32,
    /// Previous surfel radius.
    pub prev_radius: f32,
    /// Current surfel anchor `x` component.
    pub curr_pos_x: f32,
    /// Current surfel anchor `y` component.
    pub curr_pos_y: f32,
    /// Current surfel anchor `z` component.
    pub curr_pos_z: f32,
    /// Current surfel normal `x` component.
    pub curr_normal_x: f32,
    /// Current surfel normal `y` component.
    pub curr_normal_y: f32,
    /// Current surfel normal `z` component.
    pub curr_normal_z: f32,
    /// Current surfel radius.
    pub curr_radius: f32,
    /// New radiance sample `x` component.
    pub new_radiance_x: f32,
    /// New radiance sample `y` component.
    pub new_radiance_y: f32,
    /// New radiance sample `z` component.
    pub new_radiance_z: f32,
}

impl GpuSurfelUpdateInput {
    /// Build an update input from the previous `(entry, surfel)` state (or
    /// `None` on first sight), the current surfel, and the new radiance sample.
    ///
    /// The `prev_*` geometry/radiance is copied from the already-sanitised
    /// [`Surfel`]/[`SurfelCacheEntry`] fields so the device twin and the golden
    /// see identical inputs. When `prev` is `None` the history fields are left
    /// zero and `has_history` is `0`.
    #[must_use]
    pub fn from_states(
        prev: Option<(&SurfelCacheEntry, &Surfel)>,
        curr: &Surfel,
        new_radiance: Vec3,
    ) -> Self {
        let (has_history, prev_count, prev_radiance, prev_pos, prev_normal, prev_radius) =
            match prev {
                Some((entry, surfel)) => (
                    1,
                    entry.sample_count,
                    entry.radiance,
                    surfel.position,
                    surfel.normal,
                    surfel.radius,
                ),
                None => (0, 0, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0),
            };
        Self {
            has_history,
            prev_count,
            prev_radiance_x: prev_radiance.x,
            prev_radiance_y: prev_radiance.y,
            prev_radiance_z: prev_radiance.z,
            prev_pos_x: prev_pos.x,
            prev_pos_y: prev_pos.y,
            prev_pos_z: prev_pos.z,
            prev_normal_x: prev_normal.x,
            prev_normal_y: prev_normal.y,
            prev_normal_z: prev_normal.z,
            prev_radius,
            curr_pos_x: curr.position.x,
            curr_pos_y: curr.position.y,
            curr_pos_z: curr.position.z,
            curr_normal_x: curr.normal.x,
            curr_normal_y: curr.normal.y,
            curr_normal_z: curr.normal.z,
            curr_radius: curr.radius,
            new_radiance_x: new_radiance.x,
            new_radiance_y: new_radiance.y,
            new_radiance_z: new_radiance.z,
        }
    }
}

/// Advanced cache entry produced by one update invocation.
///
/// `repr(C)` `std430` element mirrored by `struct UpdateResult` in
/// `shaders/surfel_update.wesl`: the integrated radiance plus the advanced
/// confidence (frame count).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelUpdateResult {
    /// Integrated radiance `x` component (finite, non-negative).
    pub radiance_x: f32,
    /// Integrated radiance `y` component (finite, non-negative).
    pub radiance_y: f32,
    /// Integrated radiance `z` component (finite, non-negative).
    pub radiance_z: f32,
    /// Advanced confidence (frame count), capped at `max(max_samples, 1)`.
    pub sample_count: u32,
}

impl GpuSurfelUpdateResult {
    /// Pack an integrated radiance and confidence into a result.
    #[must_use]
    pub fn new(radiance: Vec3, sample_count: u32) -> Self {
        Self {
            radiance_x: radiance.x,
            radiance_y: radiance.y,
            radiance_z: radiance.z,
            sample_count,
        }
    }

    /// The integrated radiance as a [`Vec3`].
    #[must_use]
    pub fn radiance(&self) -> Vec3 {
        Vec3::new(self.radiance_x, self.radiance_y, self.radiance_z)
    }
}

const _: () = assert!(size_of::<GpuSurfelUpdateParams>() == SURFEL_UPDATE_PARAMS_SIZE);
const _: () = assert!(align_of::<GpuSurfelUpdateParams>() == 4);
const _: () = assert!(size_of::<GpuSurfelUpdateInput>() == SURFEL_UPDATE_INPUT_STRIDE);
const _: () = assert!(align_of::<GpuSurfelUpdateInput>() == 4);
const _: () = assert!(size_of::<GpuSurfelUpdateResult>() == SURFEL_UPDATE_RESULT_STRIDE);
const _: () = assert!(align_of::<GpuSurfelUpdateResult>() == 4);

// ---------------------------------------------------------------------------
// Surfel spatial-filter slice
// ---------------------------------------------------------------------------
//
// Host/device `ABI` for the surfel spatial-filter producer kernel
// `shaders/surfel_spatial_filter.wesl`, the on-device twin of the `CPU` golden
// `spatial_filter` in [`crate::gi::surface_cache::integration`]. The host fills
// a [`GpuSpatialFilterParams`] uniform, a `centers` storage array of
// [`GpuSpatialCenter`] (each carrying a slice into the shared neighbour buffer)
// and a `neighbors` storage array of [`GpuSpatialNeighbor`]; the kernel writes
// one [`GpuSpatialResult`] per centre. Every field is a 4-byte scalar so the
// layout is naturally 4-byte aligned and `vec3` straddling cannot occur under
// `std430`.

/// Threads per workgroup for the spatial-filter dispatch; mirrors the
/// `@workgroup_size(64)` in `shaders/surfel_spatial_filter.wesl`.
pub const SURFEL_SPATIAL_WORKGROUP_SIZE: u32 = 64;

/// `std430` byte size of [`GpuSpatialFilterParams`] (the uniform block).
pub const SURFEL_SPATIAL_PARAMS_SIZE: usize = 16;

/// `std430` storage stride of one [`GpuSpatialCenter`].
pub const SURFEL_SPATIAL_CENTER_STRIDE: usize = 48;

/// `std430` storage stride of one [`GpuSpatialNeighbor`].
pub const SURFEL_SPATIAL_NEIGHBOR_STRIDE: usize = 40;

/// `std430` storage stride of one [`GpuSpatialResult`].
pub const SURFEL_SPATIAL_RESULT_STRIDE: usize = 16;

/// Uniform parameters for one spatial-filter dispatch.
///
/// `repr(C)` `std430` uniform block mirrored by `struct FilterParams` in
/// `shaders/surfel_spatial_filter.wesl`. `count` is the number of centre
/// surfels; the two tunables are copied verbatim from [`CoverageParams`].
/// `reserved` pads the block to the 16-byte uniform granularity and must be
/// zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSpatialFilterParams {
    /// Number of centre surfels in the `centers` buffer.
    pub count: u32,
    /// Orientation exponent; larger values narrow the normal-agreement lobe.
    pub normal_sharpness: f32,
    /// Off-plane tolerance as a fraction of the surfel radius.
    pub axial_tolerance: f32,
    /// Reserved padding to the 16-byte uniform granularity; must be zero.
    pub reserved: u32,
}

impl GpuSpatialFilterParams {
    /// Build the dispatch parameters from the coverage tunables and a centre
    /// count, copying [`CoverageParams`] verbatim.
    #[must_use]
    pub fn from_coverage(params: &CoverageParams, count: u32) -> Self {
        Self {
            count,
            normal_sharpness: params.normal_sharpness,
            axial_tolerance: params.axial_tolerance,
            reserved: 0,
        }
    }

    /// Number of workgroups needed to cover [`count`](Self::count) at
    /// [`SURFEL_SPATIAL_WORKGROUP_SIZE`] threads each (ceil-divide).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.count.div_ceil(SURFEL_SPATIAL_WORKGROUP_SIZE)
    }
}

/// One centre surfel, its radiance, and its neighbour slice.
///
/// `repr(C)` `std430` element mirrored by `struct Center` in
/// `shaders/surfel_spatial_filter.wesl`. `neighbor_offset`/`neighbor_count`
/// bound a `[offset, offset + count)` range into the shared `neighbors` buffer
/// that the kernel reduces in order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSpatialCenter {
    /// Centre surfel anchor `x` component.
    pub pos_x: f32,
    /// Centre surfel anchor `y` component.
    pub pos_y: f32,
    /// Centre surfel anchor `z` component.
    pub pos_z: f32,
    /// Centre surfel normal `x` component.
    pub normal_x: f32,
    /// Centre surfel normal `y` component.
    pub normal_y: f32,
    /// Centre surfel normal `z` component.
    pub normal_z: f32,
    /// Centre surfel radius.
    pub radius: f32,
    /// Centre radiance `x` component.
    pub radiance_x: f32,
    /// Centre radiance `y` component.
    pub radiance_y: f32,
    /// Centre radiance `z` component.
    pub radiance_z: f32,
    /// Start index of this centre's neighbour slice in the `neighbors` buffer.
    pub neighbor_offset: u32,
    /// Number of neighbours in this centre's slice.
    pub neighbor_count: u32,
}

impl GpuSpatialCenter {
    /// Build a centre element from a surfel, its radiance, and its neighbour
    /// slice bounds in the shared neighbour buffer.
    #[must_use]
    pub fn new(surfel: &Surfel, radiance: Vec3, neighbor_offset: u32, neighbor_count: u32) -> Self {
        Self {
            pos_x: surfel.position.x,
            pos_y: surfel.position.y,
            pos_z: surfel.position.z,
            normal_x: surfel.normal.x,
            normal_y: surfel.normal.y,
            normal_z: surfel.normal.z,
            radius: surfel.radius,
            radiance_x: radiance.x,
            radiance_y: radiance.y,
            radiance_z: radiance.z,
            neighbor_offset,
            neighbor_count,
        }
    }
}

/// One neighbour surfel and its radiance.
///
/// `repr(C)` `std430` element mirrored by `struct Neighbor` in
/// `shaders/surfel_spatial_filter.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSpatialNeighbor {
    /// Neighbour surfel anchor `x` component.
    pub pos_x: f32,
    /// Neighbour surfel anchor `y` component.
    pub pos_y: f32,
    /// Neighbour surfel anchor `z` component.
    pub pos_z: f32,
    /// Neighbour surfel normal `x` component.
    pub normal_x: f32,
    /// Neighbour surfel normal `y` component.
    pub normal_y: f32,
    /// Neighbour surfel normal `z` component.
    pub normal_z: f32,
    /// Neighbour surfel radius.
    pub radius: f32,
    /// Neighbour radiance `x` component.
    pub radiance_x: f32,
    /// Neighbour radiance `y` component.
    pub radiance_y: f32,
    /// Neighbour radiance `z` component.
    pub radiance_z: f32,
}

impl GpuSpatialNeighbor {
    /// Build a neighbour element from a surfel and its radiance.
    #[must_use]
    pub fn new(surfel: &Surfel, radiance: Vec3) -> Self {
        Self {
            pos_x: surfel.position.x,
            pos_y: surfel.position.y,
            pos_z: surfel.position.z,
            normal_x: surfel.normal.x,
            normal_y: surfel.normal.y,
            normal_z: surfel.normal.z,
            radius: surfel.radius,
            radiance_x: radiance.x,
            radiance_y: radiance.y,
            radiance_z: radiance.z,
        }
    }
}

/// Filtered radiance for one centre, plus the accumulated total weight.
///
/// `repr(C)` `std430` element mirrored by `struct FilterResult` in
/// `shaders/surfel_spatial_filter.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSpatialResult {
    /// Filtered radiance `x` component (finite, non-negative).
    pub radiance_x: f32,
    /// Filtered radiance `y` component (finite, non-negative).
    pub radiance_y: f32,
    /// Filtered radiance `z` component (finite, non-negative).
    pub radiance_z: f32,
    /// Accumulated total weight (centre unit weight plus every kept neighbour).
    pub total_weight: f32,
}

impl GpuSpatialResult {
    /// The filtered radiance as a [`Vec3`].
    #[must_use]
    pub fn radiance(&self) -> Vec3 {
        Vec3::new(self.radiance_x, self.radiance_y, self.radiance_z)
    }
}

const _: () = assert!(size_of::<GpuSpatialFilterParams>() == SURFEL_SPATIAL_PARAMS_SIZE);
const _: () = assert!(align_of::<GpuSpatialFilterParams>() == 4);
const _: () = assert!(size_of::<GpuSpatialCenter>() == SURFEL_SPATIAL_CENTER_STRIDE);
const _: () = assert!(align_of::<GpuSpatialCenter>() == 4);
const _: () = assert!(size_of::<GpuSpatialNeighbor>() == SURFEL_SPATIAL_NEIGHBOR_STRIDE);
const _: () = assert!(align_of::<GpuSpatialNeighbor>() == 4);
const _: () = assert!(size_of::<GpuSpatialResult>() == SURFEL_SPATIAL_RESULT_STRIDE);
const _: () = assert!(align_of::<GpuSpatialResult>() == 4);

// --- Surfel-atlas decode (gather) producer ------------------------------------
//
// The sampling-side twin of the allocation slice: given a surfel id and a
// global atlas texel the kernel decodes the unit world direction stored at that
// texel centre (octahedral decode of the tile-local offset), the on-device twin
// of `SurfelAtlas::texel_to_dir`. The host fills a [`GpuSurfelDecodeParams`]
// uniform plus a `requests` storage array of [`GpuSurfelDecodeRequest`] and the
// kernel writes one [`GpuSurfelDecodeResult`] per request. Every field is a
// 4-byte scalar so the layout stays 4-byte aligned with no interior padding.

/// Threads per workgroup for the surfel-decode dispatch; mirrors the
/// `@workgroup_size(64)` in `shaders/surfel_decode.wesl`.
pub const SURFEL_DECODE_WORKGROUP_SIZE: u32 = 64;

/// `std430` byte size of [`GpuSurfelDecodeParams`] (the uniform block).
pub const SURFEL_DECODE_PARAMS_SIZE: usize = 16;

/// `std430` storage stride of one [`GpuSurfelDecodeRequest`].
pub const SURFEL_DECODE_REQUEST_STRIDE: usize = 12;

/// `std430` storage stride of one [`GpuSurfelDecodeResult`].
pub const SURFEL_DECODE_RESULT_STRIDE: usize = 16;

/// Valid-result bit inside [`GpuSurfelDecodeResult::flags`] (`bit 0`); clear
/// when the id is out of range or the texel falls outside the surfel's tile.
pub const SURFEL_DECODE_FLAG_VALID: u32 = 1;

/// Uniform parameters for one surfel-decode dispatch.
///
/// `repr(C)` `std430` uniform block mirrored by `struct DecodeParams` in
/// `shaders/surfel_decode.wesl`. `count` is the number of live requests; the
/// remaining three fields are the atlas dimensions, already clamped to at least
/// `1` by [`SurfelAtlas::new`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfelDecodeParams {
    /// Number of valid decode requests in the `requests` buffer.
    pub count: u32,
    /// Tile columns across the atlas (`>= 1`).
    pub tiles_per_row: u32,
    /// Tile rows down the atlas (`>= 1`).
    pub tile_rows: u32,
    /// Side length in texels of each square tile (`>= 1`).
    pub tile_resolution: u32,
}

impl GpuSurfelDecodeParams {
    /// Build the dispatch parameters from an atlas descriptor and a live
    /// request count, copying the atlas's already-clamped dimensions verbatim.
    #[must_use]
    pub fn from_atlas(atlas: &SurfelAtlas, count: u32) -> Self {
        Self {
            count,
            tiles_per_row: atlas.tiles_per_row,
            tile_rows: atlas.tile_rows,
            tile_resolution: atlas.tile_resolution,
        }
    }

    /// Number of workgroups needed to cover [`count`](Self::count) at
    /// [`SURFEL_DECODE_WORKGROUP_SIZE`] threads each (ceil-divide).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.count.div_ceil(SURFEL_DECODE_WORKGROUP_SIZE)
    }
}

/// One decode request: a surfel id and the global atlas texel to decode.
///
/// `repr(C)` `std430` element mirrored by `struct DecodeRequest` in
/// `shaders/surfel_decode.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfelDecodeRequest {
    /// Surfel id whose tile owns the texel.
    pub surfel_id: u32,
    /// Global atlas texel column to decode.
    pub texel_x: u32,
    /// Global atlas texel row to decode.
    pub texel_y: u32,
}

impl GpuSurfelDecodeRequest {
    /// Build a decode request from a surfel id and a global atlas texel.
    #[must_use]
    pub fn new(surfel_id: u32, texel_x: u32, texel_y: u32) -> Self {
        Self {
            surfel_id,
            texel_x,
            texel_y,
        }
    }
}

/// Decoded unit direction for one request, plus the valid flag.
///
/// `repr(C)` `std430` element mirrored by `struct DecodeResult` in
/// `shaders/surfel_decode.wesl`. When [`flags`](Self::flags) has
/// [`SURFEL_DECODE_FLAG_VALID`] clear the direction is all-zero and must be
/// ignored (out-of-range id or a texel outside the surfel's tile).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelDecodeResult {
    /// Decoded direction `x` component.
    pub dir_x: f32,
    /// Decoded direction `y` component.
    pub dir_y: f32,
    /// Decoded direction `z` component.
    pub dir_z: f32,
    /// Flag word; bit 0 ([`SURFEL_DECODE_FLAG_VALID`]) marks a valid decode.
    pub flags: u32,
}

impl GpuSurfelDecodeResult {
    /// The all-zero result returned for an out-of-range id or a texel that
    /// falls outside the surfel's own tile.
    #[must_use]
    pub fn invalid() -> Self {
        Self {
            dir_x: 0.0,
            dir_y: 0.0,
            dir_z: 0.0,
            flags: 0,
        }
    }

    /// Whether this result carries a valid decoded direction.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.flags & SURFEL_DECODE_FLAG_VALID != 0
    }

    /// The decoded direction as a [`Vec3`].
    #[must_use]
    pub fn direction(&self) -> Vec3 {
        Vec3::new(self.dir_x, self.dir_y, self.dir_z)
    }
}

const _: () = assert!(size_of::<GpuSurfelDecodeParams>() == SURFEL_DECODE_PARAMS_SIZE);
const _: () = assert!(align_of::<GpuSurfelDecodeParams>() == 4);
const _: () = assert!(size_of::<GpuSurfelDecodeRequest>() == SURFEL_DECODE_REQUEST_STRIDE);
const _: () = assert!(align_of::<GpuSurfelDecodeRequest>() == 4);
const _: () = assert!(size_of::<GpuSurfelDecodeResult>() == SURFEL_DECODE_RESULT_STRIDE);
const _: () = assert!(align_of::<GpuSurfelDecodeResult>() == 4);

// --- Coverage-gather producer ABI -------------------------------------------

/// Threads per workgroup for the coverage-gather dispatch; mirrors the
/// `@workgroup_size(64)` in `shaders/surfel_coverage.wesl`.
pub const SURFEL_COVERAGE_WORKGROUP_SIZE: u32 = 64;

/// `std430` byte size of [`GpuSurfelCoverageParams`] (the uniform block).
pub const SURFEL_COVERAGE_PARAMS_SIZE: usize = 16;

/// `std430` storage stride of one [`GpuCoveragePoint`].
pub const SURFEL_COVERAGE_POINT_STRIDE: usize = 32;

/// `std430` storage stride of one [`GpuCoverageSurfel`].
pub const SURFEL_COVERAGE_SURFEL_STRIDE: usize = 40;

/// `std430` storage stride of one [`GpuCoverageResult`].
pub const SURFEL_COVERAGE_RESULT_STRIDE: usize = 16;

/// Uniform parameters for one coverage-gather dispatch.
///
/// `repr(C)` `std430` uniform block mirrored by `struct CoverageParams` in
/// `shaders/surfel_coverage.wesl`. `count` is the number of shading points
/// (threads past it early-out); the two tunables are copied verbatim from
/// [`CoverageParams`]. `reserved` pads the block to the 16-byte uniform
/// granularity and must be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuSurfelCoverageParams {
    /// Number of shading points in the `points` buffer.
    pub count: u32,
    /// Orientation exponent; larger values narrow the normal-agreement lobe.
    pub normal_sharpness: f32,
    /// Off-plane tolerance as a fraction of the surfel radius.
    pub axial_tolerance: f32,
    /// Reserved padding to the 16-byte uniform granularity; must be zero.
    pub reserved: u32,
}

impl GpuSurfelCoverageParams {
    /// Build the dispatch parameters from the coverage tunables and a shading
    /// point count, copying [`CoverageParams`] verbatim.
    #[must_use]
    pub fn from_coverage(params: &CoverageParams, count: u32) -> Self {
        Self {
            count,
            normal_sharpness: params.normal_sharpness,
            axial_tolerance: params.axial_tolerance,
            reserved: 0,
        }
    }

    /// Number of workgroups needed to cover [`count`](Self::count) at
    /// [`SURFEL_COVERAGE_WORKGROUP_SIZE`] threads each (ceil-divide).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.count.div_ceil(SURFEL_COVERAGE_WORKGROUP_SIZE)
    }
}

/// One shading point and the surfel slice that may cover it.
///
/// `repr(C)` `std430` element mirrored by `struct Point` in
/// `shaders/surfel_coverage.wesl`. `surfel_offset`/`surfel_count` bound a
/// `[offset, offset + count)` range into the shared `surfels` buffer that the
/// kernel gathers in order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuCoveragePoint {
    /// Shading point position `x` component.
    pub pos_x: f32,
    /// Shading point position `y` component.
    pub pos_y: f32,
    /// Shading point position `z` component.
    pub pos_z: f32,
    /// Shading point normal `x` component.
    pub normal_x: f32,
    /// Shading point normal `y` component.
    pub normal_y: f32,
    /// Shading point normal `z` component.
    pub normal_z: f32,
    /// Start index of this point's surfel slice in the `surfels` buffer.
    pub surfel_offset: u32,
    /// Number of candidate surfels in this point's slice.
    pub surfel_count: u32,
}

impl GpuCoveragePoint {
    /// Build a shading-point element from a world position, a surface normal,
    /// and its candidate-surfel slice bounds in the shared surfel buffer.
    #[must_use]
    pub fn new(position: Vec3, normal: Vec3, surfel_offset: u32, surfel_count: u32) -> Self {
        Self {
            pos_x: position.x,
            pos_y: position.y,
            pos_z: position.z,
            normal_x: normal.x,
            normal_y: normal.y,
            normal_z: normal.z,
            surfel_offset,
            surfel_count,
        }
    }
}

/// One candidate surfel and its cached radiance.
///
/// `repr(C)` `std430` element mirrored by `struct CandidateSurfel` in
/// `shaders/surfel_coverage.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuCoverageSurfel {
    /// Candidate surfel anchor `x` component.
    pub pos_x: f32,
    /// Candidate surfel anchor `y` component.
    pub pos_y: f32,
    /// Candidate surfel anchor `z` component.
    pub pos_z: f32,
    /// Candidate surfel normal `x` component.
    pub normal_x: f32,
    /// Candidate surfel normal `y` component.
    pub normal_y: f32,
    /// Candidate surfel normal `z` component.
    pub normal_z: f32,
    /// Candidate surfel radius.
    pub radius: f32,
    /// Cached radiance `x` component.
    pub radiance_x: f32,
    /// Cached radiance `y` component.
    pub radiance_y: f32,
    /// Cached radiance `z` component.
    pub radiance_z: f32,
}

impl GpuCoverageSurfel {
    /// Build a candidate element from a surfel and its cached radiance.
    #[must_use]
    pub fn new(surfel: &Surfel, radiance: Vec3) -> Self {
        Self {
            pos_x: surfel.position.x,
            pos_y: surfel.position.y,
            pos_z: surfel.position.z,
            normal_x: surfel.normal.x,
            normal_y: surfel.normal.y,
            normal_z: surfel.normal.z,
            radius: surfel.radius,
            radiance_x: radiance.x,
            radiance_y: radiance.y,
            radiance_z: radiance.z,
        }
    }
}

/// Gathered radiance for one shading point, plus the accumulated coverage
/// weight.
///
/// `repr(C)` `std430` element mirrored by `struct CoverageResult` in
/// `shaders/surfel_coverage.wesl`. A zero [`total_weight`](Self::total_weight)
/// means no surfel covered the point and the radiance is all-zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuCoverageResult {
    /// Gathered radiance `x` component (finite, non-negative).
    pub radiance_x: f32,
    /// Gathered radiance `y` component (finite, non-negative).
    pub radiance_y: f32,
    /// Gathered radiance `z` component (finite, non-negative).
    pub radiance_z: f32,
    /// Accumulated coverage weight summed over every covering surfel.
    pub total_weight: f32,
}

impl GpuCoverageResult {
    /// The gathered radiance as a [`Vec3`].
    #[must_use]
    pub fn radiance(&self) -> Vec3 {
        Vec3::new(self.radiance_x, self.radiance_y, self.radiance_z)
    }
}

const _: () = assert!(size_of::<GpuSurfelCoverageParams>() == SURFEL_COVERAGE_PARAMS_SIZE);
const _: () = assert!(align_of::<GpuSurfelCoverageParams>() == 4);
const _: () = assert!(size_of::<GpuCoveragePoint>() == SURFEL_COVERAGE_POINT_STRIDE);
const _: () = assert!(align_of::<GpuCoveragePoint>() == 4);
const _: () = assert!(size_of::<GpuCoverageSurfel>() == SURFEL_COVERAGE_SURFEL_STRIDE);
const _: () = assert!(align_of::<GpuCoverageSurfel>() == 4);
const _: () = assert!(size_of::<GpuCoverageResult>() == SURFEL_COVERAGE_RESULT_STRIDE);
const _: () = assert!(align_of::<GpuCoverageResult>() == 4);
