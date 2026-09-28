//! Film-grain post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::film_grain`] and the shader
//! twin in `shaders/film_grain.wesl`; this module is the render-world plumbing
//! that will run them. Film grain reads the resolved, *pre-exposed* linear HDR
//! radiance (see the exposure subsystem) and adds a deterministic, per-pixel,
//! per-frame hash noise that emulates the stochastic silver-halide grains of
//! photographic film (or sensor noise), biased toward the shadows by a
//! luminance response and mixed by an artist `intensity`, reintroducing the
//! high-frequency texture a clean render lacks.
//!
//! Film grain is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same HDR buffer this pass consumes, so one implementation
//! serves them all. The remaining slices land the params uniform over the
//! resolved HDR target, the grain compute pass and its insertion into the post
//! chain — each with its first live consumer so no committed ABI is dead,
//! matching the SSR / SSGI / exposure / bloom / colour-grade precedent.

#[cfg(test)]
mod shader_tests;
