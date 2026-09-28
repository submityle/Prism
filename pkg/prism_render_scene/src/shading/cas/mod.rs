//! Contrast-Adaptive Sharpening post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::cas`] and the shader twin in
//! `shaders/cas.wesl`; this module is the render-world plumbing that will run
//! them. `CAS` reads the display-referred (tone-mapped, roughly `[0, 1]`)
//! buffer and lifts perceived detail adaptively — strongly in low-contrast
//! regions, weakly across hard edges — so it sharpens without the ringing
//! haloes a fixed unsharp mask leaves.
//!
//! Sharpening is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same buffer this pass consumes, so one sharpen serves them
//! all. The remaining slices land the params uniform over the resolved target,
//! the gather compute pass and its insertion into the post chain — each with
//! its first live consumer so no committed ABI is dead, matching the
//! `SSR` / `SSGI` / exposure / bloom precedent.

#[cfg(test)]
mod shader_tests;
