//! World-space `ReSTIR` direct-illumination for the shading stack.
//!
//! Keeps a `SHARC`-style spatial-hash cache of streaming `RIS` reservoirs in a
//! resident GPU table — one open-addressed slot per hashed world cell — so a
//! many-light scene can be shaded from a handful of surviving, spatially reused
//! light samples instead of iterating every light per pixel. The CPU golden
//! (hash keys + `GRIS` cross-cell merge + the streaming reservoir estimator)
//! lives in [`prism_render_shading::gi::world_restir`] and
//! [`prism_render_shading::gi::screen_probe`]'s `restir`; this subsystem is the
//! render-world GPU producer, mirroring the structure of
//! [`super::world_space_gi`] (the resident world-space GI producer the
//! water-surface pass already consumes at `@group(8)`).
//!
//! This foundation slice freezes the device contract — the per-cell reservoir
//! byte layout and the fill immediate block ([`abi`]) plus the opt-in
//! render-world resource ([`settings`]) — so the fill compute pass, its WESL
//! twin, and the water-surface `@group(9)` consumer can be built against a
//! stable ABI. This slice adds the resident ping-pong reservoir tables
//! ([`resources`]), the fill compute pipeline ([`pipeline`]) and its bind
//! group ([`bind_groups`]), and the `Core3d` dispatch ([`dispatch`]) that puts
//! the fill pass on the render graph.
//!
//! This slice also freezes the seed pass's device contract: the per-frame
//! light record [`abi::GpuWorldRestirLight`], the seed immediate block
//! [`abi::GpuWorldRestirSeedParams`] (plus the [`settings`] `candidate_count`
//! budget and the seed workgroup / light-stride constants), and the WESL seed
//! kernel `shaders/world_restir_seed.wesl` — the per-cell light `RIS` producer
//! that streams `candidate_count` candidates through each occupied slot's
//! reservoir, caps the confidence and finalises `W`. The kernel is compiled
//! through the `ShaderCache` and asserted bit-equal to the golden
//! [`prism_render_shading::gi::world_restir::world_reservoir`] estimator under
//! a finite-input sweep (see [`shader_tests`]); the sandbox has no GPU, so this
//! CPU parity is its device-equivalence proof.
//!
//! This slice adds the per-frame candidate light list ([`lights`]): the
//! render world's extracted punctual lights (prism [`crate::lighting`]) are
//! repacked into the frozen [`abi::GpuWorldRestirLight`] layout and kept in a
//! resident storage buffer ([`WorldRestirLights`]) ready for the seed group's
//! `@binding(2)`, uploaded every frame by [`prepare_world_restir_lights`] while
//! the subsystem is enabled.
//!
//! Wiring the seed pass onto the render graph (its pipeline, the `@binding(2)`
//! bind of [`WorldRestirLights`], and the seed -> fill dispatch ordering), and
//! the water-surface `@group(9)` consumer, land in follow-up slices.

mod abi;
mod bind_groups;
mod dispatch;
mod lights;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_world_restir_bind_groups;
pub(crate) use dispatch::world_restir_fill_pass;
pub(crate) use lights::{prepare_world_restir_lights, WorldRestirLights};
pub(crate) use pipeline::init_world_restir_pipeline;
pub(crate) use resources::prepare_world_restir_reservoirs;
pub(crate) use settings::PrismWorldRestirSettings;

#[cfg(test)]
mod shader_tests;
