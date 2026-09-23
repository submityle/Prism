mod bindings;
mod buffers;
mod rows;
mod runtime;
mod systems;

pub use bindings::GeometryBindGroup;
pub use runtime::RenderGeometryRegistry;

pub(crate) use bindings::prepare_geometry_bind_group;
pub(crate) use buffers::RenderGeometryBuffers;
pub(crate) use systems::{sync_geometry_registry, upload_geometry_buffers};
