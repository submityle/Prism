//! NPR halftone (manga / screentone) style post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::halftone`] and the shader
//! twin in `shaders/halftone.wesl`; this module is the render-world plumbing
//! that will run them. Halftone reads the resolved, stylized HDR/tone buffer
//! and reprints it as a rotated screen of tone-driven dots — dark tones grow a
//! large ink dot, bright tones shrink it — so the frame reads as comic-book
//! screentone while its average coverage still tracks the original luminance.
//!
//! Halftone is one of the NPR shading fronts that writes into the shared HDR
//! buffer, so it composes with the same post chain (exposure, colour grade,
//! vignette, ...) as every other illumination model. The remaining slices land
//! the params uniform over the resolved target, the halftone compute pass and
//! its insertion into the post chain — each with its first live consumer so no
//! committed ABI is dead, matching the color-grade / vignette precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_halftone_bind_groups;
pub(crate) use dispatch::halftone_pass;
pub(crate) use pipeline::init_halftone_pipeline;
pub(crate) use resources::prepare_halftone_textures;
pub(crate) use settings::PrismHalftoneSettings;

#[cfg(test)]
mod shader_tests;
