//! Motion-blur reconstruction subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::motion_blur`] and the shader
//! twin in `shaders/motion_blur.wesl`; this module is the render-world plumbing
//! that will run them. Motion blur is a shared post-processing base, not a peer
//! of the PBR/NPR shading fronts: every illumination model resolves the same
//! velocity buffer, and a stylized front can pin a crisp shutter while a
//! photographic front opens it for cinematic streaking. The remaining slices
//! land the TileMax/NeighborMax velocity dilation over the resolved velocity
//! target, the depth-aware reconstruction gather, and the composite back into
//! the HDR buffer — each with its first live consumer so no committed ABI is
//! dead, matching the exposure / bloom / tonemap precedent.

#[cfg(test)]
mod shader_tests;
