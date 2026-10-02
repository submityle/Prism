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

use bytemuck::{Pod, Zeroable};

use crate::gi::surface_cache::atlas::SurfelAtlas;

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
