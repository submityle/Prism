//! Tangent-space normal-map decoding and detail blending.
//!
//! This module turns the two-channel normal data that AAA content ships (see
//! [`reconstruct`]) into full unit normals, and layers detail normals onto a
//! base normal with the standard tangent-space blends (see [`blend`]). It sits
//! directly downstream of the texture sampler: a BC5 normal map sampled through
//! [`BcTexelSource`](crate::BcTexelSource) + the filter stack yields a
//! `unorm` `(R, G)`, which [`decode_rg`] turns into a unit normal, which the
//! blend operators then combine with mesostructure detail.
//!
//! Everything is pure analytic math -- no AI/ML -- so a CPU golden matches a
//! GPU twin to floating-point tolerance.
//!
//! # References
//! * Mittring, "Finding Next Gen -- `CryEngine` 2" (two-channel normals).
//! * Barre-Brisebois & Hill, "Blending in Detail" (reoriented normal mapping).

mod anisotropy;
mod blend;
mod height;
mod lean;
mod mipmap;
mod octahedral;
mod reconstruct;
mod strength;
mod surface_gradient;
mod triplanar;

pub use anisotropy::{
    anisotropic_ggx_from_covariance, ggx_alpha_to_variance, slope_covariance_eigen,
    slope_covariance_from_eigen, variance_to_ggx_alpha, SlopeEigen,
};
pub use blend::{blend_linear, blend_rnm, blend_udn, blend_whiteout};
pub use height::{height_to_normal, HeightGradient};
pub use lean::{
    lean_average, lean_covariance, lean_effective_variance, lean_from_normal, lean_from_slope,
    lean_resolve_normal, LeanMoments,
};
pub use mipmap::{
    average_unit_normals, power_from_roughness, reduce_normal_roughness_2x, roughness_from_power,
    toksvig_factor, toksvig_roughness,
};
pub use octahedral::{
    hemi_oct_decode, hemi_oct_decode_unorm, hemi_oct_encode, hemi_oct_encode_unorm,
};
pub use reconstruct::{decode_ag, decode_rg, reconstruct_z, unorm_to_snorm};
pub use strength::{normal_to_slope, scale_strength, slope_to_normal};
pub use surface_gradient::{
    blend_surface_gradient, blend_surface_gradient_pair, resolve_surface_gradient, NormalLayer,
};
pub use triplanar::{blend_triplanar_whiteout, triplanar_weights};
