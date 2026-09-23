mod bindings;
mod buffers;
pub(crate) mod rows;
mod runtime;
mod shading;
mod shading_buffers;
mod shading_registry;
mod shading_systems;
mod systems;

pub use bindings::GeometryBindGroup;
pub use runtime::RenderGeometryRegistry;
pub use shading::{
    build_shading_geometry, RenderShadingGeometry, RenderShadingGeometryHeader,
    RenderShadingPrimitive, RenderShadingVertex, ShadingGeometryBuildError,
    SHADING_GEOMETRY_FLAG_ACTIVE, SHADING_GEOMETRY_FLAG_INVALID,
    SHADING_GEOMETRY_FLAG_MISSING_NORMAL, SHADING_GEOMETRY_FLAG_MISSING_UV,
};

pub use shading_buffers::RenderShadingGeometryBuffers;
pub use shading_registry::{RenderShadingGeometryEntry, RenderShadingGeometryRegistry};

pub(crate) use bindings::prepare_geometry_bind_group;
pub(crate) use buffers::RenderGeometryBuffers;

pub(crate) use shading_systems::{
    extract_shading_geometry, sync_shading_geometry_registry, upload_shading_geometry_buffers,
    ShadingGeometryStaging,
};
pub(crate) use systems::{sync_geometry_registry, upload_geometry_buffers};
