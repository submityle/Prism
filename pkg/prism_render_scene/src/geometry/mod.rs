mod bindings;
mod buffers;
pub(crate) mod rows;
mod runtime;
mod shading;
mod systems;

pub use bindings::GeometryBindGroup;
pub use runtime::RenderGeometryRegistry;
pub use shading::{
    build_shading_geometry, RenderShadingGeometry, RenderShadingGeometryHeader,
    RenderShadingPrimitive, RenderShadingVertex, ShadingGeometryBuildError,
    SHADING_GEOMETRY_FLAG_ACTIVE, SHADING_GEOMETRY_FLAG_INVALID,
    SHADING_GEOMETRY_FLAG_MISSING_NORMAL, SHADING_GEOMETRY_FLAG_MISSING_UV,
};

pub(crate) use bindings::prepare_geometry_bind_group;
pub(crate) use buffers::RenderGeometryBuffers;
pub(crate) use systems::{sync_geometry_registry, upload_geometry_buffers};
