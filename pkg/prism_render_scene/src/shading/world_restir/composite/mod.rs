//! World-space `ReSTIR` composite: the copy + fold pipelines, per-view bind
//! groups, scratch texture, and the `Core3d` dispatch that folds the resolved
//! direct illumination back over the shaded scene under energy conservation.
//!
//! The resolve ([`super::resolve`]) leaves `rgb` = the pre-BRDF, cosine-weighted
//! `ReSTIR` direct irradiance and `a` = a `[0, 1]` hit confidence in `gi_out`.
//! The shading resolve already folded the *clustered punctual* direct diffuse
//! into `scene_color` and, in the same pass, exported that pre-BRDF punctual
//! irradiance at `world_restir_direct` (identical units to `gi_out`: raw
//! `sum max(dot(n, l), 0) * illuminance * visibility`, no `1/π`, no Fresnel, no
//! albedo). This stage performs the energy-conserving *substitution* every peer
//! GI subsystem does — swapping the clustered punctual diffuse the shading
//! resolve folded in for the `ReSTIR` estimate under the hit confidence:
//!
//! ```text
//! scene = max(base + confidence * albedo * INV_PI * (gi_out - world_restir_direct), 0)
//! ```
//!
//! where `albedo = base_color * (1 - metallic)` (the `ssgi_albedo` export) and
//! `INV_PI` converts the pre-BRDF irradiance to Lambertian diffuse radiance. A
//! fully confident pixel (`a == 1`) replaces the clustered punctual diffuse
//! with the `ReSTIR` estimate; a miss (`a == 0`) is a no-op, preserving the
//! shading-resolve direct. Specular, directional and area-light contributions
//! are untouched. The substitution drops the Fresnel `(1 - F)` factor the
//! shading resolve applied to the diffuse term — the accepted GI-substitution
//! approximation shared with SSGI / world-space GI — so the seam is only
//! exactly energy-neutral up to that factor; GPU cross-validation covers the
//! residual.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the pass runs two passes in one encoder (wgpu inserts
//! the barrier between them): a **copy** pass lifts `scene_color` into the
//! scratch `gi_base`, and a **fold** pass reads that base plus the resolve's
//! `gi_out`, the albedo and the clustered punctual direct exports and writes
//! the substitution into `scene_color`. Reading the base from a distinct
//! texture and writing `scene_color` keeps the pass free of any storage
//! read/write aliasing hazard.
//!
//! Mirrors the structure of [`super::resolve`] and
//! [`super::super::world_space_gi`]'s composite: [`resources`] owns the
//! per-view scratch, [`pipeline`] owns the two compute pipelines and their
//! group-0 layouts, [`bind_groups`] builds the copy + fold groups, and
//! [`dispatch`] records the `Core3d` pass. The composite consumes the resolve
//! export, so it exists exactly when the subsystem is enabled and the view
//! carries a resident resolve export.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_world_restir_composite_bind_groups;
pub(crate) use dispatch::world_restir_composite_pass;
pub(crate) use pipeline::init_world_restir_composite_pipeline;
pub(crate) use resources::prepare_world_restir_composite;
