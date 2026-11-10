//! The material ABI on the GPU: packed headers and the parameter word heap.
//!
//! A material's GPU form is *data*, not one object per material: every material
//! gets a row in a header buffer, and its surface parameters live in a shared
//! word heap addressed by the row's `parameter_offset`
//! ([`prism_render_material::GpuMaterialHeader`]). A shader reads the
//! header for the material it is shading and decodes `parameter_size / 4` words
//! from the heap with the canonical schema's lobe layout. Neither the ABI
//! version nor a fixed per-material word count is duplicated here.
//!
//! This module owns that data on the Prism side: [`parameter_heap`] is the
//! address-space allocator, and [`buffers`] is the pair of RHI buffers plus the
//! sync that packs a registry's records into them. The ABI itself (what a header
//! contains, how a surface packs) stays in `prism_render_material` — this layer
//! only places and uploads it.

mod binding;
mod buffers;
mod composition_oracle;
#[cfg(test)]
mod fail_closed;
mod parameter_heap;
mod stage;
mod texture_heap;
#[cfg(test)]
mod texture_heap_tests;
pub mod virtual_texture;
mod vt_source_stage;

#[cfg(test)]
mod schedule_tests;
#[cfg(test)]
mod tests;

pub use binding::{FallbackTextures, MaterialBindingLayout, MaterialTextureSlots};
pub(crate) use buffers::{header_row_bytes, texture_row_bytes};
pub use buffers::{
    ExecutionTableSyncError, MaterialBufferSnapshot, MaterialFrameSnapshotBundle,
    MaterialFrameSnapshotError, MaterialParameterBuffers, MaterialSyncReport,
    NativeExecutionTableBuffers, NativeExecutionTableSnapshot,
};
pub(crate) use composition_oracle::{
    NativeMaterialCompositionAdapter, NativeMaterialCompositionError,
    NativeMaterialCompositionMetadata,
};
pub use parameter_heap::ParameterHeap;
pub(crate) use stage::sync_material_buffers;
pub use stage::TextureResolutionError;
pub use stage::{MaterialUploadStats, RenderMaterials, TextureRequest};
pub use texture_heap::{
    BindlessSlot, TextureFallback, TextureHeapAcquireError, TextureHeapStats, TextureSlotHeap,
};
pub use virtual_texture::MaterialVtLeaseSnapshot;
pub(crate) use vt_source_stage::extract_native_vt_sources;
pub use vt_source_stage::{NativeVtSourceIssue, NativeVtSourceStatus};
