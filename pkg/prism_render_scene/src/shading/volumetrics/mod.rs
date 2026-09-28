//! Froxel volumetric fog subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::volumetrics`] and the shader
//! twin in `shaders/volumetrics.wesl`; this module is the render-world plumbing
//! that will run them. Volumetric fog integrates *participating media* — fog,
//! dust, god rays, self-lit magical haze — into a view-frustum-aligned 3D grid
//! of "froxels". Each froxel stores its medium's scattering/absorption
//! coefficients and the light already in-scattered toward the eye; a final
//! front-to-back column march folds each slice's energy-conserving
//! in-scattering under the transmittance accumulated by nearer slices, so
//! nearer media correctly occlude farther media. The per-view result is applied
//! to the lit scene as `final = background * transmittance + in_scattering`.
//!
//! Fog is a subsystem on the shared GPU-driven base rather than a peer of the
//! PBR/NPR shading fronts: it consumes the depth the opaque pass already
//! resolves and the shadow maps the lighting pass already produces, and its
//! single-scattering integration is illumination-model agnostic (a stylized
//! front can drive the same froxel grid with an authored phase/colour). The
//! remaining slices land the froxel grid resource, the light-injection and
//! column-integration compute passes over the depth/shadow reads, the `Core3d`
//! dispatch nodes, and the composite consumption — each with its first live
//! consumer so no committed ABI is dead, matching the SSR / SSGI / outline
//! precedent.

#[cfg(test)]
mod shader_tests;
