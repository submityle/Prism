//! Clustered-forward (Forward+) light culling for the render world.
//!
//! This submodule mirrors the CPU golden
//! [`assign_lights_to_clusters`](prism_render_shading::assign_lights_to_clusters)
//! onto the GPU.  [`extract`] captures the active camera's froxel-grid view;
//! [`build`] turns the extracted punctual lights and that view into the three
//! flat tables described by [`abi`]; [`buffers`] streams them into storage
//! buffers; [`bindings`] exposes the read-only bind group; and [`systems`]
//! drives the per-frame build/upload/bind lifecycle.  The resolve pass consumes
//! the bind group to iterate only the lights touching each pixel's cluster
//! instead of the whole scene.

mod abi;
mod bindings;
mod buffers;
mod build;
mod extract;
mod systems;

pub use abi::GpuClusterGrid;
pub use bindings::ClusterBindGroup;
pub use buffers::ClusterGpuBuffers;
pub use build::{build_cluster_data, ClusterConfig, ClusterCpuData};
pub use extract::{ClusterViewFit, ExtractedClusterView};

pub(crate) use extract::extract_cluster_view;
pub(crate) use systems::{
    prepare_cluster_bind_group, rebuild_cluster_buffers, write_cluster_buffers,
};
