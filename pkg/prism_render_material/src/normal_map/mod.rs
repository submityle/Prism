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

mod blend;
mod mipmap;
mod reconstruct;
mod strength;

pub use blend::{blend_linear, blend_rnm, blend_udn, blend_whiteout};
pub use mipmap::{
    average_unit_normals, power_from_roughness, reduce_normal_roughness_2x, roughness_from_power,
    toksvig_factor, toksvig_roughness,
};
pub use reconstruct::{decode_ag, decode_rg, reconstruct_z, unorm_to_snorm};
pub use strength::{normal_to_slope, scale_strength, slope_to_normal};
