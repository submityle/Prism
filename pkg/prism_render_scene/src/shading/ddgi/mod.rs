//! DDGI (dynamic diffuse global illumination) irradiance-volume GPU subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::gi::irradiance_volume`] (the
//! Majercik et al. 2019 / RTXGI irradiance field) and the per-pixel sample
//! shader twin in `shaders/ddgi_sample.wesl`; this module is the render-world
//! plumbing that will run it. DDGI resolves diffuse indirect light by storing a
//! sparse lattice of octahedral probes, each holding a small irradiance map and
//! a two-moment depth map, and reconstructing a shading point's irradiance from
//! the eight probes of its enclosing cell:
//!
//! * `sample_main` (this block) reconstructs the shading point's world position
//!   and normal, locates the enclosing probe cell, and blends the eight corner
//!   probes with combined trilinear x back-face x Chebyshev-visibility weights
//!   (golden `relocation`), reading each probe's octahedral irradiance / depth
//!   tiles from the two atlas textures (golden `IrradianceOct` /
//!   `DdgiDepthOct`). The clamped-cosine diffuse GI irradiance is written into
//!   an export buffer (`gi_out`, `rgba16float`; `rgb` = irradiance,
//!   `a` = confidence).
//! * The probe-update / relocation / classification compute passes and the
//!   energy-conserving `scene_color` composite land in later DDGI blocks, built
//!   on the same ABI ([`abi`]) and the opt-in [`settings::PrismDdgiSettings`]
//!   resource.
//!
//! Unlike the Lumen-style world-space path ([`super::world_space_gi`]), DDGI
//! probes persist across frames and resolve a volumetric field rather than a
//! screen-space one, so it stays stable off-screen and under fast camera
//! motion. The subsystem is opt-in; when disabled the passes allocate nothing
//! and dispatch nothing.
//!
//! Numerical parity with the golden is enforced by the CPU-mirror tests in
//! [`shader_tests`], which transcribe the WESL sample maths op-for-op and assert
//! agreement to `1e-6`.

// The device passes that consume these ABIs and settings land across the later
// DDGI blocks (probe update, relocation, composite, pipeline/bind-group/dispatch
// wiring). Until that plumbing is in place the scaffolding is exercised only by
// the shader-parity tests, so suppress the interim dead-code / unused-re-export
// noise here rather than leaking per-item attributes that would churn as the
// passes land.
#![allow(dead_code, unused_imports)]

mod abi;
mod bind_groups;
mod composite;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use abi::{
    GpuDdgiSampleParams, GpuDdgiUpdateParams, GpuDdgiVolume, DDGI_PROBE_UPDATE_WORKGROUP_SIZE,
    DDGI_WORKGROUP_SIZE,
};
pub(crate) use bind_groups::prepare_ddgi_bind_groups;
pub(crate) use composite::{
    ddgi_composite_pass, init_ddgi_composite_pipeline, prepare_ddgi_composite_bind_groups,
};
pub(crate) use dispatch::{ddgi_probe_update_pass, ddgi_sample_pass};
pub(crate) use pipeline::init_ddgi_pipeline;
pub(crate) use resources::{prepare_ddgi_textures, ViewDdgi};
pub(crate) use settings::PrismDdgiSettings;

#[cfg(test)]
mod shader_tests;
