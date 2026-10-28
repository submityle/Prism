//! Production Decal SceneFabric vertical slice.
//!
//! The recorder consumes authenticated domain resources, builds the real
//! six-storage-buffer bind group, and submits the frozen `DECAL_RESOLVE_PASS`
//! through the shared SceneFabric runtime. It intentionally does not own a
//! readback path or a second allocator.

mod bind_groups;
pub(crate) mod fabric;
mod pipeline;
mod recorder;

#[cfg(test)]
mod cpu_tests;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod shader_tests;

pub(crate) use pipeline::init_decal_resolve_pipeline;
#[allow(unused_imports)]
pub(crate) use recorder::{
    install_decal_resolve, prepare_decal_resources_from_owner, publish_decal_resolve_from_owner,
    retire_decal_resolve_resources, DecalPreparedResources,
};
