//! Bloom (light glow) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::bloom`] and the shader twin
//! in `shaders/bloom.wesl`; this module is the render-world plumbing that will
//! run them. Bloom reads the resolved, *pre-exposed* HDR radiance (see the
//! exposure subsystem) and scatters energy from the brightest features into
//! their neighbourhood via a Karis-prefiltered, energy-preserving "dual filter"
//! pyramid (COD 13-tap downsample + 3x3 tent upsample), then blends the result
//! over the scene by an artist intensity.
//!
//! Bloom is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: every illumination model — physically based or stylized — writes
//! into the same HDR buffer this pass consumes, so one implementation serves
//! them all. The remaining slices land the mip-pyramid allocation over the HDR
//! target, the prefilter/downsample/upsample compute passes and the final
//! combine into the post chain — each with its first live consumer so no
//! committed ABI is dead, matching the SSR / SSGI / exposure precedent.

#[cfg(test)]
mod shader_tests;
