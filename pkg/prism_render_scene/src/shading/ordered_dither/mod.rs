//! NPR ordered-dithering (Bayer-threshold posterisation) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::ordered_dither`] and the
//! shader twin in `shaders/ordered_dither.wesl`; this module is the render-world
//! plumbing that runs them. Ordered dithering reads the resolved, pre-exposed
//! `HDR` radiance (see the exposure subsystem) and biases a per-channel colour
//! quantisation with a fixed 4x4 `Bayer` matrix, snapping the scene to a small
//! palette / bit depth so flat gradients resolve into the stable retro dither of
//! a limited indexed display.
//!
//! This is *ordered dithering*, not the `halftone` pass: `halftone` renders tonal
//! value as the area of round ink dots, while this pass thresholds each channel
//! against the `Bayer` matrix and snaps the result to discrete `levels`. The
//! `Bayer` bias is what turns naive posterise banding into the familiar stable,
//! non-flickering cross-hatch.
//!
//! Ordered dithering is a shared post-processing base, not a peer of the
//! `PBR`/NPR shading fronts: every illumination model — physically based or
//! stylized — writes into the same `HDR` buffer this pass consumes, so one dither
//! serves them all. The subsystem is the standard five-file split — the
//! [`GpuOrderedDitherParams`](abi::GpuOrderedDitherParams) immediate block, the
//! [`PrismOrderedDitherSettings`] resource, the compute pipeline, the per-view
//! output texture and its bind group, and the `Core3d` dispatch — each committed
//! with its first live consumer so no ABI is dead, matching the
//! SSR / SSGI / exposure / bloom / colour-grade / gamut-map precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_ordered_dither_bind_groups;
pub(crate) use dispatch::ordered_dither_pass;
pub(crate) use pipeline::init_ordered_dither_pipeline;
pub(crate) use resources::prepare_ordered_dither_textures;
pub(crate) use settings::PrismOrderedDitherSettings;

#[cfg(test)]
mod shader_tests;
