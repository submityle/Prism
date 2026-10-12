//! The material ABI on the GPU: packed headers and the parameter word heap.
//!
//! A material's GPU form is *data*, not one object per material: every material
//! gets a row in a header buffer, and its surface parameters live in a shared
//! word heap addressed by the row's `parameter_offset`
//! ([`prism_render_material::GpuMaterialHeader`], ABI v4). A shader reads the
//! header for the material it is shading and decodes `parameter_size / 4` words
//! from the heap with `lobe_mask` — which is why the heap is variable-length:
//! a plain dielectric costs 12 words, a fully-loaded surface up to 36.
//!
//! This module owns that data on the Prism side: [`parameter_heap`] is the
//! address-space allocator, and [`buffers`] is the pair of RHI buffers plus the
//! sync that packs a registry's records into them. The ABI itself (what a header
//! contains, how a surface packs) stays in `prism_render_material` — this layer
//! only places and uploads it.

mod binding;
mod buffers;
mod parameter_heap;
mod texture_heap;
#[cfg(test)]
mod texture_heap_tests;
mod stage;

#[cfg(test)]
mod tests;

pub use binding::{FallbackTextures, MaterialTextureSlots};
pub use buffers::{MaterialParameterBuffers, MaterialSyncReport};
#[cfg(test)]
pub(crate) use buffers::{header_row_bytes, texture_row_bytes};
pub(crate) use stage::sync_material_buffers;
pub use stage::{MaterialUploadStats, RenderMaterials, TextureRequest};
pub use texture_heap::{BindlessSlot, TextureFallback, TextureHeapStats, TextureSlotHeap};
pub use parameter_heap::ParameterHeap;

