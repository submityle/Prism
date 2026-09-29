//! NPR posterization (tone-separation) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::posterize`] and the shader
//! twin in `shaders/posterize.wesl`; this module is the render-world plumbing
//! that will run them. Posterize reads the resolved, *pre-exposed* linear HDR
//! radiance (see the exposure subsystem), *before* the display tone-map curve,
//! and collapses the continuous tonal range into a small set of flat bands — the
//! poster-print / cel-shaded look — either by quantizing each RGB channel
//! independently or by quantizing the Rec. 709 luminance and rescaling to keep
//! the hue, then blending the result over the input by a strength knob.
//!
//! Posterize is a shared stylized post-processing base, not a peer of the
//! PBR/NPR shading fronts: every illumination model — physically based or
//! stylized — writes into the same HDR buffer this pass consumes, so one
//! posterize serves them all. The remaining slices land the params uniform over
//! the resolved HDR target, the posterize compute pass and its insertion into
//! the post chain — each with its first live consumer so no committed ABI is
//! dead, matching the SSR / SSGI / exposure / bloom / colour-grade / gamut-map
//! precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_posterize_bind_groups;
pub(crate) use dispatch::posterize_pass;
pub(crate) use pipeline::init_posterize_pipeline;
pub(crate) use resources::prepare_posterize_textures;
pub(crate) use settings::PrismPosterizeSettings;

#[cfg(test)]
mod shader_tests;
