//! SDF face-shadow visibility subsystem for the stylized (NPR) axis.
//!
//! The CPU golden lives in [`prism_render_shading::face_shadow`] and the shader
//! twin in `shaders/face_shadow.wesl`; this module is the render-world plumbing
//! that will run them. The face shadow keys off the key light's azimuth in the
//! head's own frame: a signed distance field baked into the face texture encodes
//! at what light angle each pixel flips from lit to shadowed, so the art-directed
//! anime face shadow shape stays stable under animation instead of relying on
//! `N.L` or shadow maps (the Guilty Gear / Genshin / Honkai lineage).
//!
//! Face shadow is not a peer of the whole PBR/NPR shading front: it produces a
//! `[0, 1]` visibility term that feeds the `visibility` input of the stylized
//! direct lighting (via `stylized_shadow`), so it composes with the
//! stepped-shadow and cel-ramp stack rather than replacing it. The remaining
//! slices land the head-frame and params uniforms over the resolved face pass,
//! the SDF sample (with mirrored UV addressing) and the visibility evaluation —
//! each with its first live consumer so no committed ABI is dead, matching the
//! color-grade / vignette precedent.

#[cfg(test)]
mod shader_tests;
