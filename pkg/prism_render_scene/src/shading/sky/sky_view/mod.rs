//! GPU Hillaire-style physical-sky *sky-view* LUT bake — the consumer of the
//! multiple-scattering and transmittance LUTs.
mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
#[cfg(test)]
mod shader_tests;
pub(crate) use bind_groups::prepare_sky_view_bind_group;
pub(crate) use dispatch::sky_view_lut_pass;
pub(crate) use pipeline::init_sky_view_pipeline;
pub(crate) use resources::init_sky_view_lut;
