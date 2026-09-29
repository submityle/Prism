//! Kuwahara anisotropic painterly-filter NPR post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::kuwahara`] and the shader
//! twin in `shaders/kuwahara.wesl`; this module is the render-world plumbing
//! that will run them. The Kuwahara filter is a stylized (NPR) front: it splits
//! the neighbourhood around each pixel into four overlapping quadrants, takes
//! each quadrant's RGB mean and luminance variance, and writes the mean of the
//! lowest-variance quadrant. Selecting the flattest region preserves hard edges
//! while flooding smooth interiors with a single averaged colour, producing the
//! characteristic oil-painting look.
//!
//! Like the other post-processing stylizers, this pass reads the resolved HDR
//! radiance every illumination model writes into, so one filter serves the PBR
//! and NPR fronts alike. The remaining slices land the params uniform, the
//! quadrant-selection compute pass and its insertion into the post chain — each
//! with its first live consumer so no committed ABI is dead, matching the
//! color-grade / SSR / SSGI / exposure / bloom precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_kuwahara_bind_groups;
pub(crate) use dispatch::kuwahara_pass;
pub(crate) use pipeline::init_kuwahara_pipeline;
pub(crate) use resources::prepare_kuwahara_textures;
pub(crate) use settings::PrismKuwaharaSettings;

#[cfg(test)]
mod shader_tests;
