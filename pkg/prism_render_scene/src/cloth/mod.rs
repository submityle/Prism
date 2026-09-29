//! GPU cloth compute subsystem.
//!
//! The CPU-golden XPBD solver and its GPU schedule live in
//! `prism_render_architecture::cloth`: that crate is deterministic and
//! dependency-free, so it owns the algorithms, the per-substep kernel order
//! ([`prism_render_architecture::cloth::gpu::kernels::ClothKernel`]) and the
//! resident buffer sizing
//! ([`prism_render_architecture::cloth::gpu::buffers`]). This module is the
//! `bevy`-render half that turns that schedule into real `wgpu` compute
//! dispatches against the sibling `WESL` shaders
//! (`shaders/cloth_sim.wesl`, `shaders/cloth_collision.wesl`,
//! `shaders/cloth_embed.wesl`, `shaders/cloth_aerodynamics_snapshot.wesl`,
//! `shaders/cloth_aerodynamics.wesl`), which are already naga-validated.
//!
//! The subsystem is split into cohesive files rather than one large module,
//! mirroring the layout of the other compute passes under `shading/`:
//!
//! * [`abi`] - `#[repr(C)]` host records shared with the cloth shaders, plus
//!   the `size_of` contract tests that pin each record to the golden buffer
//!   strides so a layout drift fails the build.
//! * [`pipeline`] - the thirteen compute pipelines and the seven group-0
//!   bind-group layouts, keyed by `ClothKernel` so the dispatch slice looks each
//!   pass up directly from the golden schedule.
//! * [`bind_groups`] - per-piece resident buffer allocation (sized by the golden
//!   buffer contract) and the seven bind groups those pipelines dispatch against.
//!
//! The remaining slice adds the `Core3d` graph node that records the dispatches
//! in the
//! [`ClothKernel`](prism_render_architecture::cloth::gpu::kernels::ClothKernel)
//! order, plus the plugin that embeds the shaders and installs the pipelines and
//! graph node.

mod abi;
mod bind_groups;
mod dispatch;
mod extract;
mod garment;
mod pack;
mod pipeline;
pub(crate) mod plugin;
mod prepare;
mod resources;
#[cfg(test)]
mod aero_gpu_tests;
#[cfg(test)]
mod aero_parity;
#[cfg(test)]
mod shader_tests;
#[cfg(test)]
mod sim_gpu_tests;
mod solve_plan;
