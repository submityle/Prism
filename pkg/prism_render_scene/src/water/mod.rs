//! GPU water / fluid compute subsystem.
//!
//! The `CPU`-golden solvers (`Tessendorf` spectrum `IFFT`, `Gerstner`
//! superposition, `SWE`, `PBF`, `FLIP`/`APIC`) and their deterministic `GPU`
//! schedules live in `prism_render_architecture::water`: that crate is
//! dependency-free and float-audited, so it owns the algorithms, the per-frame
//! kernel order
//! ([`prism_render_architecture::water::gpu::kernels::WaterKernel`]) and the
//! resident buffer sizing
//! ([`prism_render_architecture::water::gpu::buffers`]). This module is the
//! `bevy`-render half that turns that schedule into real `wgpu` compute
//! dispatches against the sibling `WESL` shaders (`shaders/water.wesl`,
//! `shaders/water_ocean.wesl`, `shaders/water_flip.wesl`,
//! `shaders/water_pbf.wesl`, `shaders/water_surface.wesl`,
//! `shaders/water_render_fx.wesl`), which are already `naga`-validated.
//!
//! Mirroring the cloth compute subsystem, the module is split into cohesive
//! files rather than one large module:
//!
//! * [`abi`] - `#[repr(C)]` host records shared with the water shaders, plus
//!   the `size_of` contract tests that pin each record to the golden buffer
//!   strides so a layout drift fails the build.
//!
//! * [`pipeline`] - the pipeline table (keyed by `WaterKernel`) and the twelve
//!   bind-group layouts, built once at `RenderStartup`.
//! * [`bind_groups`] - one body's resident device buffers, storage / sampled
//!   textures and twelve bind groups.
//! * [`resources`] and [`dispatch`] - the resident render-world body set and
//!   the `Core3d` graph node that records the dispatches in solver order.
//! * [`body`], [`extract`] and [`prepare`] - the main-world author chain: the
//!   [`WaterBody`](body::WaterBody) component, its per-frame render-world
//!   snapshot, and the systems that mirror bodies into the render world and
//!   turn each into a resident `GPU` body with its ordered dispatch schedule.
//! * [`plugin`] - the plugin that embeds the shaders, installs the pipelines,
//!   the author chain and the graph node.

mod abi;
mod authoring;
mod bind_groups;
mod body;
mod dispatch;
mod extract;
mod fft_upload;
#[cfg(test)]
mod gpu_bench;
#[cfg(test)]
mod gpu_tests;
mod pipeline;
pub(crate) mod plugin;
mod prepare;
mod resources;
#[cfg(test)]
mod shader_tests;
mod surface_draw;
mod surface_mesh;
mod surface_node;
mod surface_pipeline;
mod surface_shading;
mod surface_ssr;
mod surface_vsm;

/// The high-level water authoring presets and the water body component they
/// build, re-exported so a game can spawn an ocean, `FLIP`/`PBF` pool, or
/// shallow-water pond with one call.
pub use authoring::{
    CoastlinePreset, FlipPoolPreset, LakeInflow, LakePreset, OceanPreset, PbfPoolPreset,
    RiverControlPoint, RiverPreset, ShallowWaterPreset,
};
pub use body::WaterBody;
