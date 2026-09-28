//! Cross-hatching (Hatching) NPR post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::hatching`] and the shader
//! twin in `shaders/hatching.wesl`; this module is the render-world plumbing
//! that will run them. Cross-hatching is a stylized (NPR) front: it reads the
//! resolved HDR radiance, maps its Rec. 709 luminance to a four-tier ramp and
//! overlays one to three sets of rotated periodic line strokes — densest in the
//! shadows — then composites the ink over paper, reproducing the pen-and-ink
//! shading technique.
//!
//! Like the other post-processing stylizers, this pass consumes the same HDR
//! buffer every illumination model writes into, so one hatch serves the PBR and
//! NPR fronts alike. The remaining slices land the params uniform, the coverage
//! compute pass and its insertion into the post chain — each with its first live
//! consumer so no committed ABI is dead, matching the color-grade / kuwahara /
//! SSR / SSGI / exposure / bloom precedent.

#[cfg(test)]
mod shader_tests;
