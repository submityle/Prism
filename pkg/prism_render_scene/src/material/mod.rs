mod bindings;
mod buffers;
mod parameter_heap;
mod consumer;
mod plugin;
pub(crate) mod runtime;
pub(crate) mod systems;
pub mod texture_heap;
mod texture_upload;

pub use bindings::MaterialBindGroup;
pub use consumer::{MaterialBufferBindings, MaterialReader};
pub use plugin::PrismMaterialPlugin;
pub use runtime::PrismMaterialDiagnostics;
pub use texture_heap::{BindlessHeapStats, BindlessSlot, BindlessTextureHeap};
pub use texture_upload::{MaterialTextureArrays, MAX_BINDLESS_TEXTURES};
