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
// dispatch uniform). Gated to test builds until the pipeline/bind-group/
// dispatch slices land a non-test consumer; its live consumers today are the
// layout + round-trip parity tests in `reuse_tests` and `abi`'s own `tests`.
#[cfg(test)]
mod abi;

#[cfg(test)]
mod shader_tests;

#[cfg(test)]
mod reservoir_tests;

#[cfg(test)]
mod brdf_mis_tests;

#[cfg(test)]
mod reuse_tests;
