//! Versioned material IR and runtime ABI shared by every Prism renderer.

#![expect(missing_docs, reason = "The ABI is documented as it freezes.")]

extern crate alloc;

mod authoring;
mod axis;
#[cfg(feature = "bevy")]
mod bevy_bridge;
mod handle;
mod ir;
mod record;
mod registry;
mod resources;
mod surface;
mod texture_addressing;
mod texture_lod;
mod texture_sample;
mod texture_filter;
mod texture_codec;
mod validation;

pub use authoring::FACE_SHADOW_SDF_SEMANTIC;
pub use axis::{Illumination, SpecializationId};
#[cfg(feature = "bevy")]
pub use bevy_bridge::{lower_standard_material, StandardMaterialTextureResolver, TextureSemantic};
pub use handle::{MaterialCapacityError, MaterialHandleAllocator, MaterialHandleError};
pub use ir::{
    ClosureKind, MaterialGraph, MaterialNode, MaterialNodeId, MaterialValue, NormalizedMaterial,
    MAX_CLOSURE_SLAB_DEPTH,
};
pub use record::{
    fallback_material_header, fallback_material_record, inactive_material_header,
    GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters, MaterialDomain,
    MaterialFeatureFlags, MaterialRecord, MaterialRenderClass, FALLBACK_MATERIAL_HANDLE,
    MATERIAL_ABI_VERSION, MAX_MATERIAL_TEXTURES,
};
pub use registry::{MaterialRegistry, MaterialRegistryError, MaterialSnapshot};
pub use resources::{MaterialResourceHandle, MaterialResourceKind, MaterialResourceTable};
pub use surface::{
    GpuAnisotropyLobe, GpuClearCoatLobe, GpuEmissionLobe, GpuSheenLobe, GpuSubsurfaceLobe,
    GpuSurfaceCore, GpuTransmissionLobe, LobeMask, SurfaceParameterBlock, SurfaceUnpackError,
    SURFACE_CORE_WORDS, SURFACE_LOBE_WORDS,
};
pub use texture_addressing::{address_uv, wrap_coord, AddressResult, WrapMode};
pub use texture_lod::{
    anisotropic_taps, cone_mip_level, mip_from_isotropic_footprint, trilinear_mip, AnisoTaps,
    AnisotropicMip, PageRequest, RayCone, RayDifferential, TriangleLodConstant, TrilinearMip,
    VirtualTexture, MAX_ANISO_TAPS, MIN_COS_INCIDENCE,
};
pub use texture_sample::{resolve_cone, resolve_differential, SampleRequest, SampleResolved};
pub use texture_filter::{
    bilinear, filter_resolved, trilinear, wrap_texel, TexelAddr, TexelSource,
};
pub use texture_codec::{
    decode_bc1, decode_bc3, decode_bc4, decode_bc5, rgb565_to_rgb888, BcFormat,
    BcSourceError, BcTexelSource,
};
pub use validation::{validate_graph, MaterialValidationError};

#[cfg(test)]
mod tests;
