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
mod texture_blend;
mod texture_blur;
mod texture_codec;
mod texture_color;
mod texture_ewa;
mod texture_filter;
mod texture_lod;
mod texture_mipgen;
mod texture_morphology;
mod texture_resize;
mod texture_sample;
mod texture_upsample;
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
    anisotropic_ggx_from_covariance, average_unit_normals, blend_linear, blend_rnm,
    blend_surface_gradient, blend_surface_gradient_pair, blend_triplanar_whiteout, blend_udn,
    blend_whiteout, decode_ag, decode_rg, equirect_decode, equirect_encode, ggx_alpha_to_variance,
    height_to_normal, hemi_oct_decode, hemi_oct_decode_unorm, hemi_oct_encode,
    hemi_oct_encode_unorm, lean_average, lean_covariance, lean_effective_variance,
    lean_from_normal, lean_from_slope, lean_resolve_normal, normal_to_slope, object_to_tangent,
    orthonormalize_basis, power_from_roughness, reconstruct_z, reduce_normal_roughness_2x,
    resolve_surface_gradient, roughness_from_power, scale_strength, slope_covariance_eigen,
    slope_covariance_from_eigen, slope_to_normal, spheremap_decode, spheremap_decode_unorm,
    spheremap_encode, spheremap_encode_unorm, stereographic_decode, stereographic_encode,
    tangent_to_object, toksvig_factor, toksvig_roughness, triplanar_weights, unorm_to_snorm,
    variance_to_ggx_alpha, HeightGradient, LeanMoments, NormalLayer, SlopeEigen, TangentBasis,
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
pub use texture_blend::{
    blend_channel, blend_nonseparable, blend_nonseparable_rgba8, blend_rgba8, BlendMode,
    NonSeparableBlendMode,
};
pub use texture_blur::{
    bilateral_blur, bilateral_blur_plane, blur_plane, box_blur, box_blur_plane, gaussian_blur,
    gaussian_weights_1d, joint_bilateral_blur, joint_bilateral_blur_plane, unsharp_mask,
    unsharp_mask_plane,
};
pub use texture_codec::{
    bc6h_mode_bits, bc7_mode, decode_bc1, decode_bc2, decode_bc3, decode_bc4, decode_bc4_signed,
    decode_bc5, decode_bc5_signed, decode_bc6h_mode11_signed, decode_bc6h_mode11_unsigned,
    decode_bc6h_mode12_signed, decode_bc6h_mode12_unsigned, decode_bc6h_mode13_signed,
    decode_bc6h_mode13_unsigned, decode_bc6h_mode14_signed, decode_bc6h_mode14_unsigned,
    decode_bc6h_mode1_signed, decode_bc6h_mode1_unsigned, decode_bc6h_mode2_signed, decode_bc6h_mode2_unsigned, decode_bc6h_mode3_signed,
    decode_bc6h_mode3_unsigned, decode_bc6h_signed, decode_bc6h_unsigned,
    decode_bc7, decode_bc7_mode0, decode_bc7_mode1, decode_bc7_mode2, decode_bc7_mode3,
    decode_bc7_mode4, decode_bc7_mode5, decode_bc7_mode6, decode_bc7_mode7, decode_eac_r11_snorm,
    decode_eac_r11_unorm, decode_eac_rg11_snorm, decode_eac_rg11_unorm, decode_etc2_rgb8,
    encode_bc1, encode_bc2, encode_bc3, encode_bc4, encode_bc4_signed, encode_bc5,
    encode_bc5_signed, encode_bc6h_mode11_signed, encode_bc6h_mode11_unsigned, encode_bc7_mode4,
    encode_bc7_mode5, encode_bc7_mode6, encode_etc2_rgb8, etc2_rgb8_mode, half_bits_to_f32,
    rgb565_to_rgb888, Bc6hError, Bc7Error, BcFormat, BcSourceError, BcTexelSource, Etc2Error,
    Etc2Mode,
};
pub use texture_color::{
    downsample as chroma_downsample, hsl_to_rgb, hsv_to_rgb, rgb_to_hsl, rgb_to_hsv, rgb_to_ycbcr,
    rgb_to_ycocg, rgb_to_ycocg_r, upsample as chroma_upsample, ycbcr_to_rgb, ycocg_r_to_rgb,
    ycocg_to_rgb, ChromaPlane, ChromaSubsampling, YCbCrMatrix,
};
pub use texture_ewa::{ewa_sample_plane, ewa_sample_rgba8};
pub use texture_filter::{
    bicubic_catmull_rom, bilinear, bspline_cubic, bspline_cubic_fast, bspline_cubic_weights,
    catmull_rom_weights, cubic_mitchell, filter_resolved, filter_resolved_bicubic,
    filter_resolved_bspline, filter_resolved_mitchell, mitchell_netravali_weights, trilinear,
    trilinear_bicubic, trilinear_bspline, trilinear_mitchell, wrap_texel, TexelAddr, TexelSource,
    MITCHELL_B, MITCHELL_C,
};
pub use texture_lod::{
    anisotropic_taps, cone_mip_level, mip_from_isotropic_footprint, trilinear_mip, AnisoTaps,
    AnisotropicMip, PageRequest, RayCone, RayDifferential, TriangleLodConstant, TrilinearMip,
    VirtualTexture, MAX_ANISO_TAPS, MIN_COS_INCIDENCE,
};
pub use texture_mipgen::{
    alpha_test_coverage, apply_alpha_scale, box_downsample, gaussian_downsample,
    generate_mip_chain, generate_mip_chain_gaussian, generate_mip_chain_kaiser,
    generate_mip_chain_premultiplied, generate_mip_chain_tent, generate_mip_chain_windowed,
    kaiser_downsample, linear_to_srgb, premultiplied_box_downsample, preserve_alpha_coverage,
    solve_alpha_scale, srgb_to_linear, tent_downsample, windowed_downsample, ColorSpace,
    GaussianFilter, KaiserFilter, Rgba8Image, TentFilter, WindowedKernel,
};
pub use texture_morphology::{close_plane, dilate_plane, erode_plane, open_plane};
pub use texture_resize::{resize, ResizeFilter};
pub use texture_sample::{resolve_cone, resolve_differential, SampleRequest, SampleResolved};
pub use texture_upsample::{joint_bilateral_upsample_plane, joint_bilateral_upsample_rgba8};
pub use validation::{validate_graph, MaterialValidationError};

#[cfg(test)]
mod tests;
