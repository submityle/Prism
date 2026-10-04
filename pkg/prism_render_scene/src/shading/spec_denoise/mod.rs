//! Specular GI denoiser (GPU) — the on-device twin of the CPU golden
//! `prism_render_shading::gi::spec_denoise`.
//!
//! This subsystem is brought up block by block alongside `spec_gi`. The first
//! block lands the virtual-reflection history reprojection primitives in
//! `shaders/spec_denoise_reproject.wesl` (the ReBLUR/ReLAX-style specular
//! reprojection the later spatial-filter and history-clamp passes feed from)
//! together with the parity harness in [`reproject_tests`], which proves the
//! WESL transcription matches the golden `reproject` numerically rather than
//! merely compiling. The spatial filter and history-clamp blocks arrive next.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

pub(crate) use bind_groups::prepare_spec_denoise_bind_groups;
pub(crate) use dispatch::spec_denoise_spatial_pass;
pub(crate) use pipeline::init_spec_denoise_spatial_pipeline;
pub(crate) use resources::{prepare_spec_denoise_resources, ViewSpecDenoise};

#[cfg(test)]
mod reproject_tests;

#[cfg(test)]
mod spatial_tests;

#[cfg(test)]
mod history_clamp_tests;
