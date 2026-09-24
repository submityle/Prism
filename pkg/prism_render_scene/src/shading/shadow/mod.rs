//! GPU shadow-visibility subsystem for the shading resolve pass.
//!
//! The shadow evaluation math lives in `shaders/shadow.wesl`, the WESL twin of
//! the CPU golden reference in `prism_render_shading::shadow`.  It computes the
//! analytic `visibility` term (directional cascaded shadow maps with PCF/PCSS,
//! and omnidirectional point-light cube distance maps) that the resolve pass
//! multiplies into each light's contribution.
//!
//! This module currently owns the WESL compilation coverage for that shader.
//! The GPU shadow-map render passes that fill the atlas (depth rasterization,
//! directional CSM texel-snap stabilization, cube-face rendering, atlas budget
//! allocation) and the resolve-side wiring that feeds these matrices and the
//! atlas texture are built in later slices and require on-device validation for
//! numerical parity against the CPU reference.

#[cfg(test)]
mod shader_tests;
