//! Lighting-channel and NPR light-layer routing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::light_routing`] and the
//! shader twin in `shaders/light_routing.wesl`; this module is the
//! render-world plumbing that will run them. Routing answers two gates before
//! the resolve pass evaluates a BSDF for a light/surface pair: a *lighting
//! channel* visibility gate (per-light and per-primitive bitmasks that must
//! intersect, Unreal's Lighting Channels) shared by every illumination model,
//! and — for stylized fronts — an NPR *light layer* gate routing the
//! contribution into a key/fill/rim accumulation bin. PBR and hybrid read only
//! the channel gate; NPR reads both, which is why the same per-light routing
//! record serves all four engine modes with no divergent storage.
//!
//! The gate is decided on the light/primitive pair, not the shading maths, so
//! it slots into the shared GPU-driven base rather than a per-front branch: the
//! clustered-forward cull ([`prism_render_shading::cluster`]) refines a
//! cluster's per-light bitset by channel, and the resolve pass consumes the
//! layer gate when accumulating stylized bins. The remaining slices land the
//! per-view routing buffer, the channel byte in the vis-buffer/G-buffer, the
//! cull refinement in the resolve dispatch, and the stylized composite's
//! per-layer consumption — each with its first live consumer so no committed
//! ABI is dead, matching the SSR / SSGI / outline / volumetrics precedent.

#[cfg(test)]
mod shader_tests;
