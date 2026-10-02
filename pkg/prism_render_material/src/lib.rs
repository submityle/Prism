//! Versioned material IR and runtime ABI shared by every Prism renderer.

#![expect(missing_docs, reason = "The ABI is documented as it freezes.")]

extern crate alloc;

mod authoring;
mod axis;
#[cfg(feature = "bevy")]
mod bevy_bridge;
mod handle;
mod ir;
mod normal_map;
mod record;
mod registry;
mod resources;
mod surface;
mod texture_addressing;
mod texture_codec;
mod texture_filter;
mod texture_lod;
mod texture_mipgen;
mod texture_sample;
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
pub use normal_map::{
    average_unit_normals, blend_linear, blend_rnm, blend_udn, blend_whiteout, decode_ag, decode_rg,
    power_from_roughness, reconstruct_z, reduce_normal_roughness_2x, roughness_from_power,
    toksvig_factor, toksvig_roughness, unorm_to_snorm,
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
pub use texture_codec::{
    bc6h_mode_bits, bc7_mode, decode_bc1, decode_bc2, decode_bc3, decode_bc4, decode_bc4_signed,
    decode_bc5, decode_bc5_signed, decode_bc6h_mode11_signed, decode_bc6h_mode11_unsigned,
    decode_bc6h_signed, decode_bc6h_unsigned, decode_bc7, decode_bc7_mode4, decode_bc7_mode5,
    decode_bc7_mode6, decode_etc2_rgb8, encode_bc1, encode_bc2, encode_bc3, encode_bc4,
    encode_bc4_signed, encode_bc5, encode_bc5_signed, encode_bc6h_mode11_signed,
    encode_bc6h_mode11_unsigned, encode_bc7_mode4, encode_bc7_mode5, encode_bc7_mode6,
    encode_etc2_rgb8, etc2_rgb8_mode, half_bits_to_f32, rgb565_to_rgb888, Bc6hError, Bc7Error,
    BcFormat, BcSourceError, BcTexelSource, Etc2Error, Etc2Mode,
};
pub use texture_filter::{
    bicubic_catmull_rom, bilinear, bspline_cubic, bspline_cubic_fast, bspline_cubic_weights,
    catmull_rom_weights, cubic_mitchell, filter_resolved, mitchell_netravali_weights, trilinear,
    wrap_texel, TexelAddr, TexelSource, MITCHELL_B, MITCHELL_C,
};
pub use texture_lod::{
    anisotropic_taps, cone_mip_level, mip_from_isotropic_footprint, trilinear_mip, AnisoTaps,
    AnisotropicMip, PageRequest, RayCone, RayDifferential, TriangleLodConstant, TrilinearMip,
    VirtualTexture, MAX_ANISO_TAPS, MIN_COS_INCIDENCE,
};
pub use texture_mipgen::{
    alpha_test_coverage, apply_alpha_scale, box_downsample, generate_mip_chain,
    generate_mip_chain_kaiser, generate_mip_chain_premultiplied, generate_mip_chain_windowed,
    kaiser_downsample, linear_to_srgb, premultiplied_box_downsample, preserve_alpha_coverage,
    solve_alpha_scale, srgb_to_linear, windowed_downsample, ColorSpace, KaiserFilter, Rgba8Image,
    WindowedKernel,
};
pub use texture_sample::{resolve_cone, resolve_differential, SampleRequest, SampleResolved};
pub use validation::{validate_graph, MaterialValidationError};

#[cfg(test)]
mod tests;
