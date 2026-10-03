//! Specular / glossy GI (GPU) — the on-device twin of the CPU golden
//! `prism_render_shading::gi::spec_gi`.
//!
//! This subsystem is being brought up block by block. The first block lands the
//! GGX visible-normal lobe primitives in `shaders/spec_gi_lobe.wesl` (the
//! microfacet maths the later glossy-ReSTIR trace / resolve / denoise passes
//! import) together with the parity harness in [`shader_tests`], which proves
//! the WESL transcription matches the golden `ggx_lobe` numerically rather than
//! merely compiling. Pipeline wiring (trace reservoirs, resolve, MIS) arrives in
//! the following blocks.

// Host ABI for the glossy-specular ReSTIR *reuse* pass (storage reservoir +
// dispatch uniform). Promoted to a non-test module now that [`resources`] sizes
// the resident ping-pong reservoir buffers against `GpuSpecrReservoir`; the
// `GpuSpecGiReuseConfig` uniform it also exports is consumed by the bind-group /
// dispatch slices that follow.
mod abi;

// Per-view resident resources for the reuse pass: the ping-pong pair of glossy
// reservoir storage buffers and the resolved specular+confidence target.
mod resources;

// Compute pipeline, group-0 layout and the `RenderStartup` initializer for the
// reuse dispatch. The per-view bind group and the `Core3d` dispatch node that
// consume them arrive in the following slices.
mod pipeline;

// Per-view group-0 bind group + per-frame config-uniform upload for the reuse
// dispatch. Consumes `pipeline`'s layout and `resources`' ping-pong accessors;
// the `Core3d` dispatch node that records against it arrives in the next slice.
mod bind_groups;

#[cfg(test)]
mod shader_tests;

#[cfg(test)]
mod reservoir_tests;

#[cfg(test)]
mod brdf_mis_tests;

#[cfg(test)]
mod reuse_tests;
