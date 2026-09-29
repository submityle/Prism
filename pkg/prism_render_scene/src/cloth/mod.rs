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
//! `shaders/cloth_embed.wesl`), which are already naga-validated.
//!
//! The subsystem is split into cohesive files rather than one large module,
//! mirroring the layout of the other compute passes under `shading/`:
//!
//! * [`abi`] - `#[repr(C)]` host records shared with the cloth shaders, plus
//!   the `size_of` contract tests that pin each record to the golden buffer
//!   strides so a layout drift fails the build.
//!
//! Later slices add the render-resource pipeline, per-piece bind-group
//! preparation and the `Core3d` graph node that records the dispatches in the
//! [`ClothKernel`](prism_render_architecture::cloth::gpu::kernels::ClothKernel)
//! order. Keeping the ABI as its own verified slice means those slices bind
//! against a layout that is already proven byte-for-byte against both the
//! shader `struct`s and the golden buffer sizing.

mod abi;
