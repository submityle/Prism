//! Depth-of-field (DoF) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::dof`] and the shader twin in
//! `shaders/dof.wesl`; this module is the render-world plumbing that will run
//! them. DoF reads the resolved, *pre-exposed* HDR radiance (see the exposure
//! subsystem) plus the scene depth, computes each pixel's thin-lens circle of
//! confusion from the physical camera (aperture / focal length / focus
//! distance / sensor width), separates the near and far fields and blends a
//! soft-edged bokeh gather over the sharp image by the CoC.
//!
//! DoF is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: every illumination model — physically based or stylized — writes
//! into the same HDR buffer this pass consumes, so one implementation serves
//! them all. The remaining slices land the CoC prepass over the depth target,
//! the near/far gather compute passes and the final composite into the post
//! chain — each with its first live consumer so no committed ABI is dead,
//! matching the SSR / SSGI / exposure / bloom precedent.

#[cfg(test)]
mod shader_tests;
