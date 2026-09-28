//! Vignette (lens shading / darkened edges) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::vignette`] and the shader
//! twin in `shaders/vignette.wesl`; this module is the render-world plumbing
//! that will run them. Vignette reads the resolved, *pre-exposed* linear HDR
//! radiance (see the exposure subsystem) and darkens the frame toward its edges
//! either physically (the `cos^4` optical fall-off of a real lens) or as an
//! artistic framing device (a `Unity`-`PPv2`-style `smoothstep` fall-off that
//! blends a circular and square shape by a roundness knob).
//!
//! Vignette is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: every illumination model — physically based or stylized — writes
//! into the same HDR buffer this pass consumes, so one vignette serves them
//! all. The remaining slices land the params uniform over the resolved HDR
//! target, the vignette compute pass and its insertion into the post chain —
//! each with its first live consumer so no committed ABI is dead, matching the
//! SSR / SSGI / exposure / bloom / colour-grade precedent.

#[cfg(test)]
mod shader_tests;
