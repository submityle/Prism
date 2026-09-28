//! HDR colour grading post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::color_grade`] and the shader
//! twin in `shaders/color_grade.wesl`; this module is the render-world plumbing
//! that will run them. Colour grading reads the resolved, *pre-exposed* linear
//! HDR radiance (see the exposure subsystem) and applies the colourist's
//! primary controls — a von Kries white balance, ASC CDL lift/gamma/gain, a
//! contrast expansion about middle grey and a luma-preserving saturation — in a
//! fixed order, *before* the display tone-map curve so hue shifts stay stable
//! across the dynamic range (the DaVinci / ACES workflow).
//!
//! Colour grading is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same HDR buffer this pass consumes, so one grade serves them
//! all. The remaining slices land the params uniform over the resolved HDR
//! target, the grade compute pass and its insertion into the post chain — each
//! with its first live consumer so no committed ABI is dead, matching the
//! SSR / SSGI / exposure / bloom precedent.

#[cfg(test)]
mod shader_tests;
