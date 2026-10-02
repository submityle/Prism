//! Display tone-mapping subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::tonemap`] and the shader
//! twin in `shaders/tonemap.wesl`; this module is the render-world plumbing
//! that will run them. Tone mapping is the step after exposure: it takes the
//! pre-exposed linear HDR radiance in the resolved target and compresses its
//! open domain into the display-referred `[0, 1]` range with a chosen operator
//! (Reinhard, extended Reinhard, ACES Narkowicz, ACES fitted or `AgX`) before the
//! sRGB / PQ transfer encode.
//!
//! Tone mapping is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model writes pre-exposed radiance that
//! flows through the same operator, and a stylized front can pin a flatter
//! curve (or `AgX` with a custom look) for an illustrative grade. The remaining
//! slices land the operator/white-point uniform, the full-screen tone-map pass
//! over the resolved HDR target and the display-encode hand-off — each with its
//! first live consumer so no committed ABI is dead, matching the exposure / SSR
//! / SSGI precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_tonemap_bind_groups;
pub(crate) use dispatch::tonemap_pass;
pub(crate) use pipeline::init_tonemap_pipeline;
pub(crate) use resources::prepare_tonemap_textures;
pub(crate) use settings::PrismTonemapSettings;

#[cfg(test)]
mod shader_tests;
