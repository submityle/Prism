//! Chromatic aberration (lens colour fringing) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::chromatic_aberration`] and
//! the shader twin in `shaders/chromatic_aberration.wesl`; this module is the
//! render-world plumbing that will run them. Lateral chromatic aberration reads
//! the resolved image and samples each colour channel from a *radially* offset
//! texture coordinate — the split growing with distance from the optical centre
//! — then recombines the three reads, reproducing the wavelength-dependent
//! magnification of a real lens (the Unity Post Processing Stack v2 / Unreal
//! Engine "Scene Fringe" convention).
//!
//! Chromatic aberration is a shared post-processing effect, not a peer of the
//! PBR/NPR shading fronts: every illumination model — physically based or
//! stylized — resolves into the same image this pass consumes, so one effect
//! serves them all. The remaining slices land the params uniform over the
//! resolved target, the aberration compute pass and its insertion into the post
//! chain — each with its first live consumer so no committed ABI is dead,
//! matching the SSR / SSGI / exposure / bloom / colour-grade precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_chromatic_aberration_bind_groups;
pub(crate) use dispatch::chromatic_aberration_pass;
pub(crate) use pipeline::init_chromatic_aberration_pipeline;
pub(crate) use resources::prepare_chromatic_aberration_textures;
pub(crate) use settings::PrismChromaticAberrationSettings;

#[cfg(test)]
mod shader_tests;
