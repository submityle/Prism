//! Screen-space lens-flare / glare post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::lens_flare`] and the shader
//! twin in `shaders/lens_flare.wesl`; this module is the render-world plumbing
//! that runs them. Lens flare reads the resolved, *pre-exposed* linear `HDR`
//! radiance (see the exposure subsystem), isolates the bright tail and
//! re-images it as the coloured `ghost` discs and radial `halo` a real lens
//! throws — the `ghost`s stepped through the optical centre, an optional
//! per-channel chromatic dispersion fringing them, then composited additively
//! over the scene by an artist intensity.
//!
//! Lens flare is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same `HDR` buffer this pass consumes, so one implementation
//! serves them all. The slices land the params immediate over the resolved
//! `HDR` target, the flare compute pass and its insertion into the post chain —
//! each with its first live consumer so no committed ABI is dead, matching the
//! SSR / SSGI / exposure / bloom / colour-grade precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_lens_flare_bind_groups;
pub(crate) use dispatch::lens_flare_pass;
pub(crate) use pipeline::init_lens_flare_pipeline;
pub(crate) use resources::prepare_lens_flare_textures;
pub(crate) use settings::PrismLensFlareSettings;

#[cfg(test)]
mod shader_tests;
