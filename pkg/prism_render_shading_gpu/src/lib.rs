//! # `prism_render_shading_gpu`
//!
//! The wgpu twin of Prism's CPU direct-lighting golden: a WGSL reimplementation
//! of [`prism_render_shading`]'s `evaluate_principled_direct` base lobe, driven
//! through the frozen `prism_render_driver` RHI, plus the reusable PBR surface
//! pass that carries it. This is the "material path on the RHI" proof: the
//! device evaluates the same BRDF as the CPU reference, on real hardware, and a
//! parity test asserts the two agree within a documented floating-point budget.
//!
//! ## Mirror scope
//!
//! [`MIRROR_SCOPE`] states, as data, exactly what the twin matches. In short:
//!
//! *Base lobe, matched term for term:*
//! `base_color`, `base_weight`, `metallic`, `diffuse_roughness` (Oren-Nayar),
//! `perceptual_roughness` (GGX width `r^2`), the `normal_variance` GGX-width
//! filter, `specular_weight` / `specular_color` / `specular_ior` (the
//! `modulated_ior` dielectric Fresnel and the metal `F82`-tint), anisotropic
//! GGX `D`/`V` with `anisotropy` / `anisotropy_rotation`, the Kulla-Conty +
//! Turquin multiple-scattering compensation, and `emissive` including the
//! `n.l <= 0` early-out.
//!
//! *Not mirrored:* the `OpenPBR` lobes outside the base layer — coat/clearcoat,
//! fuzz/sheen, transmission, subsurface, and thin-film iridescence. The
//! parameter struct exposes no field for them, so a call site cannot ask the
//! GPU for a lobe the shader does not evaluate; the CPU golden must be built
//! with those fields at their identity defaults.
//!
//! *Also not mirrored:* the CPU reference's defensive non-finite guards. WGSL as
//! compiled by naga 30 / wgpu 30 has no `isNan`/`isInf` builtin, so the twin is
//! defined for finite inputs — which is what every finite case is verified
//! against, and the only place those guards ever fire is outside that domain.
//!
//! ## Layout
//! - [`params`]: [`GpuSurfaceParams`], the `#[repr(C)]` ABI shared with the WGSL
//!   uniform, with its field-offset table and byte packing.
//! - [`bindings`]: the bind-group layout, the uniform batch upload, the result
//!   storage buffer, and the bind group.
//! - [`pipeline`]: [`SurfacePass`], the compiled shader/pipeline and its batch.
//! - [`shader`]: the single-sourced WGSL text and entry-point metadata.
//!
//! ## Backends
//!
//! The library depends only on `prism_render_driver`; `prism_render_driver_wgpu`
//! and the CPU golden `prism_render_shading` are dev-dependencies used by the
//! parity test. The crate names no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod bindings;
pub mod params;
pub mod pipeline;
pub mod shader;

#[cfg(test)]
mod tests;

pub use bindings::{
    create_result_buffer, surface_bind_group, surface_bind_group_layout, upload_surface_batch,
    RESULT_STRIDE,
};
pub use params::{GpuSurfaceParams, BATCH_BYTES, MAX_BATCH, SURFACE_PARAMS_SIZE};
pub use pipeline::{SurfaceEvaluation, SurfacePass};
pub use shader::{EVALUATE_ENTRY_POINT, SURFACE_SHADER_LABEL, SURFACE_SHADER_WGSL, WORKGROUP_SIZE};

/// A machine-checkable statement of what the WGSL twin mirrors.
///
/// The parity test asserts that every term listed under [`MATCHED_PREFIX`]
/// appears here and that each `OpenPBR` lobe named in [`UNMIRRORED_PREFIX`] is
/// documented as excluded, so the mirror scope cannot drift from the code
/// without failing a test.
///
/// [`MATCHED_PREFIX`]: Self::MATCHED_PREFIX
/// [`UNMIRRORED_PREFIX`]: Self::UNMIRRORED_PREFIX
pub const MIRROR_SCOPE: &str = "\
MATCHED (base lobe of evaluate_principled_direct):
  base_color, base_weight, metallic, diffuse_roughness (Oren-Nayar),
  perceptual_roughness (GGX width r^2), normal_variance (GGX-width filter),
  specular_weight, specular_color, specular_ior (modulated_ior dielectric
  Fresnel + metal F82-tint), anisotropy, anisotropy_rotation (anisotropic GGX
  D/V), multiple-scattering compensation (Kulla-Conty + Turquin), emissive
  (including the n.l <= 0 early-out).
UNMIRRORED (OpenPBR lobes outside the base layer):
  coat, clearcoat, fuzz, sheen, transmission, subsurface, thin-film iridescence.
OMITTED GUARDS (defensive, never fire on finite inputs):
  the CPU is_finite checks and their saturating fallbacks, because WGSL as
  compiled by naga 30 / wgpu 30 exposes no isNan/isInf builtin.";

impl MirrorScope {
    /// The heading that introduces the matched terms in [`MIRROR_SCOPE`].
    pub const MATCHED_PREFIX: &str = "MATCHED (base lobe of evaluate_principled_direct):";
    /// The heading that introduces the excluded lobes in [`MIRROR_SCOPE`].
    pub const UNMIRRORED_PREFIX: &str = "UNMIRRORED (OpenPBR lobes outside the base layer):";
    /// The heading that introduces the omitted defensive guards.
    pub const GUARD_PREFIX: &str = "OMITTED GUARDS (defensive, never fire on finite inputs):";
    /// The surface terms the twin matches.
    pub const MATCHED_TERMS: &[&str] = &[
        "base_color",
        "base_weight",
        "metallic",
        "diffuse_roughness (Oren-Nayar)",
        "perceptual_roughness (GGX width r^2)",
        "normal_variance (GGX-width filter)",
        "specular_weight",
        "specular_color",
        "specular_ior (modulated_ior dielectric",
        "Fresnel + metal F82-tint)",
        "anisotropy",
        "anisotropy_rotation (anisotropic GGX",
        "D/V)",
        "multiple-scattering compensation (Kulla-Conty + Turquin)",
        "emissive",
        "(including the n.l <= 0 early-out)",
    ];
    /// The `OpenPBR` lobes the twin does not mirror.
    pub const UNMIRRORED_LOBES: &[&str] = &[
        "coat",
        "clearcoat",
        "fuzz",
        "sheen",
        "transmission",
        "subsurface",
        "thin-film iridescence",
    ];
    /// The defensive behavioural difference the shader cannot express.
    pub const OMITTED_GUARDS: &[&str] = &["is_finite", "isNan/isInf"];
}

/// Namespace for the [`MIRROR_SCOPE`] accessors.
///
/// This is a marker type: every member is an associated constant, so the
/// mirror-scope vocabulary is reachable without instantiating anything.
#[derive(Clone, Copy, Debug)]
pub struct MirrorScope;
