//! World-space global-illumination GPU subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::gi::world_space`] and the
//! shader twins in `shaders/world_space_gi_probe_update.wesl` /
//! `shaders/world_space_gi_resolve.wesl`; this module is the render-world
//! plumbing that runs them. World-space GI is a Lumen-style diffuse indirect
//! path built from two same-frame compute passes over a shared screen-probe
//! buffer:
//!
//! * `probe_update_main` seeds one L1 spherical-harmonic radiance probe per
//!   screen tile from the pre-exposed scene colour, snapping gather directions
//!   through the octahedral map (golden `octahedral` / `radiance_cache`).
//! * `resolve_main` gathers the four screen probes around each pixel and blends
//!   their SH irradiance with bilinear plus geometry-aware weights (golden
//!   `probe_interpolation`), writing the diffuse GI into an export buffer.
//!
//! The result (`gi_out`, `rgba16float`) is a diffuse GI irradiance *export*
//! buffer (`rgb` = irradiance, `a` = confidence), which a third `Core3d` pass
//! (`composite`) then folds back over `scene_color` under an energy-conserving
//! substitution (mirroring SSGI's composite), swapping the flat IBL ambient the
//! shading resolve already applied for the world-space gather under confidence.
//!
//! The subsystem is opt-in (see [`PrismWorldSpaceGiSettings`]); when
//! disabled the passes allocate nothing and dispatch nothing.

mod abi;
mod bind_groups;
mod composite;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_world_space_gi_bind_groups;
pub(crate) use composite::{
    init_world_space_gi_composite_pipeline, prepare_world_space_gi_composite_bind_groups,
    world_space_gi_composite_pass,
};
pub(crate) use dispatch::world_space_gi_pass;
pub(crate) use pipeline::init_world_space_gi_pipeline;
pub(crate) use resources::prepare_world_space_gi_textures;
pub(crate) use settings::PrismWorldSpaceGiSettings;

#[cfg(test)]
mod shader_tests;
