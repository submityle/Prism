//! GPU screen-space global illumination (SSGI) subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::screen_space::gi`] and the
//! shader twin in `shaders/ssgi.wesl`; this module is the render-world plumbing
//! that runs them. SSGI integrates one indirect *diffuse* bounce by casting
//! several cosine-weighted hemisphere rays per pixel and marching them across
//! the *same* reverse-Z Hi-Z pyramid and current-frame colour pyramid the
//! reflection subsystem already builds. Hits pick up on-screen radiance (colour
//! bleeding); misses take the IBL/SH ambient the resolve evaluates, so the
//! gather augments the ambient term without introducing energy discontinuities.
//!
//! Because SSGI reuses SSR's rebuilt inputs, the additional plumbing is a
//! single compute pass; it lands across cohesive slices matching the rest of
//! the shading pipeline:
//!
//! * *(following slices)* — the shared `SsgiConfig` immediate ABI, the per-view
//!   output target, the trace pipeline and bind group, the `Core3d` dispatch
//!   node, and the resolve consumption that blends the pre-albedo indirect
//!   radiance over the pure IBL/SH ambient under the pass's confidence. The ABI
//!   struct lands with that first live consumer so no committed ABI is dead,
//!   matching the SSR precedent.

#[cfg(test)]
mod shader_tests;
