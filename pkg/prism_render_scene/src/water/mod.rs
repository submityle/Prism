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
//! The remaining slices add the pipeline table (keyed by `WaterKernel`), the
//! per-body resident bind groups, the `Core3d` graph node that records the
//! dispatches in solver order, and the plugin that embeds the shaders and
//! installs the pipelines and graph node.

mod abi;
mod pipeline;
