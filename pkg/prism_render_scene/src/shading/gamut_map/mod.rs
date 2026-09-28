//! HDR gamut-mapping post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::gamut_map`] and the shader
//! twin in `shaders/gamut_map.wesl`; this module is the render-world plumbing
//! that will run them. Gamut mapping reads the resolved, *pre-exposed* linear
//! HDR radiance (see the exposure subsystem), after colour grading and *before*
//! the display tone-map curve, and compresses colours that fall outside the
//! target display gamut (linear `sRGB` / Rec. 709) smoothly back toward the
//! achromatic axis — the `OpenColorIO` / `ACES` gamut-compress approach — rather
//! than hard-clamping them and shifting their hue.
//!
//! Gamut mapping is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same HDR buffer this pass consumes, so one gamut map serves
//! them all. The remaining slices land the params uniform over the resolved HDR
//! target, the gamut-map compute pass and its insertion into the post chain —
//! each with its first live consumer so no committed ABI is dead, matching the
//! SSR / SSGI / exposure / bloom / colour-grade precedent.

#[cfg(test)]
mod shader_tests;
