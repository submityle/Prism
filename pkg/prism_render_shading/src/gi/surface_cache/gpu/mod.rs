//! `GPU` surface-cache producers: on-device twins of the surface-cache `CPU`
//! golden, built to the Lumen-style surfel-atlas model.
//!
//! A Lumen-style surface cache keeps every surfel's directional outgoing
//! radiance in a shared octahedral atlas and refreshes it from the `GPU` every
//! frame. The host-side golden in [`super::atlas`] / [`super::integration`]
//! owns the addressing and temporal-integration maths; this module hosts the
//! compute-kernel twins that actually run on device plus the `repr(C)` `ABI`
//! and the scalar `CPU` mirrors that pin each kernel to its golden.
//!
//! # Slices
//!
//! * [`abi`] — the `repr(C)` `std430` host/device layout shared with the
//!   `WESL` kernels (parameters, requests, resolved slots) and the stride
//!   constants and compile-time size assertions that guard it.
//! * [`alloc`] — the scalar `CPU` mirror [`allocate_slot`] of the
//!   surfel-allocation (atlas-addressing) kernel
//!   `shaders/surfel_alloc.wesl`, resolving each request's global atlas texel
//!   and tile exactly as the kernel does.
//! * [`update`] — the scalar `CPU` mirror [`update_entry`] of the
//!   surfel-update (temporal-integration) kernel `shaders/surfel_update.wesl`,
//!   advancing a surfel's cached radiance by one frame (confidence-weighted
//!   `EMA` + disocclusion reset) exactly as the kernel does.
//! * [`filter`] — the scalar `CPU` mirror [`filter_center`] of the
//!   surfel spatial-filter (bilateral neighbour reuse) kernel
//!   `shaders/surfel_spatial_filter.wesl`, reducing a centre surfel's
//!   neighbour slice by the geometric reuse weight exactly as the kernel does.
//!
//! # Validation model
//!
//! The sandbox has no `GPU` (Metal is unavailable), so the kernels are not
//! dispatched here. Instead each slice is validated two ways, mirroring the
//! no-`GPU` paradigm the sibling render crates use: `naga` parses and
//! type-checks the `WESL` source (proving it compiles exactly as it will on
//! device), and a `CPU` mirror reproduces the kernel's arithmetic and is
//! cross-checked bit-for-bit against the host golden over a direction/id sweep.
//! Both live in the `#[cfg(test)]` [`mod@tests`] module.
//!
//! Provenance: standard octahedral surfel-atlas addressing; no Unreal Engine
//! source or derived code.

pub mod abi;
pub mod alloc;
pub mod filter;
pub mod update;

pub use abi::{
    GpuSpatialCenter, GpuSpatialFilterParams, GpuSpatialNeighbor, GpuSpatialResult,
    SURFEL_SPATIAL_CENTER_STRIDE, SURFEL_SPATIAL_NEIGHBOR_STRIDE, SURFEL_SPATIAL_PARAMS_SIZE,
    SURFEL_SPATIAL_RESULT_STRIDE, SURFEL_SPATIAL_WORKGROUP_SIZE,
};
pub use abi::{
    GpuSurfelAllocParams, GpuSurfelAllocRequest, GpuSurfelAllocSlot, SURFEL_ALLOC_FLAG_VALID,
    SURFEL_ALLOC_PARAMS_SIZE, SURFEL_ALLOC_REQUEST_STRIDE, SURFEL_ALLOC_SLOT_STRIDE,
    SURFEL_ALLOC_WORKGROUP_SIZE,
};
pub use abi::{
    GpuSurfelUpdateInput, GpuSurfelUpdateParams, GpuSurfelUpdateResult, SURFEL_UPDATE_INPUT_STRIDE,
    SURFEL_UPDATE_PARAMS_SIZE, SURFEL_UPDATE_RESULT_STRIDE, SURFEL_UPDATE_WORKGROUP_SIZE,
};
pub use alloc::allocate_slot;
pub use filter::filter_center;
pub use update::update_entry;

#[cfg(test)]
mod tests;
