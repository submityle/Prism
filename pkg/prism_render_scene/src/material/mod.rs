mod bindings;
mod buffers;
mod consumer;
mod parameter_heap;
mod plugin;
pub(crate) mod runtime;
pub(crate) mod systems;
pub mod texture_heap;
mod texture_upload;
pub mod virtual_texture;
pub(crate) mod vt_source;

pub use bindings::MaterialBindGroup;
pub use consumer::{MaterialBufferBindings, MaterialReader};
pub use plugin::PrismMaterialPlugin;
pub use runtime::PrismMaterialDiagnostics;
pub use texture_heap::{BindlessHeapStats, BindlessSlot, BindlessTextureHeap};
pub use texture_upload::{MaterialTextureArrays, MAX_BINDLESS_TEXTURES};
pub use virtual_texture::{
    MaterialVirtualTextureAtlasConfig, MaterialVirtualTextureAtlasError,
    MaterialVirtualTextureGpuBundle, MaterialVirtualTextureGpuConfig,
    MaterialVirtualTextureResourceSizes, MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE,
    MATERIAL_VIRTUAL_TEXTURE_ENTRY_WORDS,
};
