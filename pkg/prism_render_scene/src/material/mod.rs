mod bindings;
mod buffers;
mod consumer;
mod plugin;
pub(crate) mod runtime;
pub(crate) mod systems;

pub use bindings::MaterialBindGroup;
pub use consumer::{MaterialBufferBindings, MaterialReader};
pub use plugin::PrismMaterialPlugin;
pub use runtime::PrismMaterialDiagnostics;
