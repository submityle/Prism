//! `@group(0)` bind-group construction for the ray-traversal service.
//!
//! Each helper reads the auto-derived layout straight off its compute pipeline
//! (`pipeline.get_bind_group_layout(0)`) and binds the uploaded buffers in the
//! exact binding order the sibling `WESL` kernel declares - the same order the
//! `#[cfg(test)]` parity harness proved on device. The bindings are pinned to
//! the frozen [`super::abi`] record layout, so a drift fails the shared
//! contract tests rather than mis-binding a dispatch.

use bevy_render::render_resource::{BindGroup, BindGroupEntry, BindGroupLayout, ComputePipeline};
use bevy_render::renderer::RenderDevice;

use super::resources::{BvhTraversalResources, FootprintResources, TlasTraversalResources};

/// Binds the single-`BLAS` `ray_traverse` kernel's five `@group(0)` resources.
///
/// Order: `nodes` (0), `triangles` (1), `rays` (2), `hits` (3), `params` (4).
pub(crate) fn bvh_bind_group(
    device: &RenderDevice,
    pipeline: &ComputePipeline,
    resources: &BvhTraversalResources,
) -> BindGroup {
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    device.create_bind_group(
        Some("prism_rt_bvh_group0"),
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: resources.nodes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: resources.triangles.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: resources.rays.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: resources.hits.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: resources.params.as_entire_binding(),
            },
        ],
    )
}

/// Binds the two-level `tlas_traverse` kernel's eight `@group(0)` resources.
///
/// Order: `tlas_nodes` (0), `instances` (1), `pool_nodes` (2),
/// `pool_triangles` (3), `pool_offsets` (4), `rays` (5), `hits` (6),
/// `params` (7).
pub(crate) fn tlas_bind_group(
    device: &RenderDevice,
    pipeline: &ComputePipeline,
    resources: &TlasTraversalResources,
) -> BindGroup {
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    device.create_bind_group(
        Some("prism_rt_tlas_group0"),
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: resources.tlas_nodes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: resources.instances.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: resources.pool_nodes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: resources.pool_triangles.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: resources.pool_offsets.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: resources.rays.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: resources.hits.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: resources.params.as_entire_binding(),
            },
        ],
    )
}

/// Binds the `ray_footprint` kernel's three `@group(0)` resources.
///
/// Order: `footprints` (0), `results` (1), `params` (2).
pub(crate) fn footprint_bind_group(
    device: &RenderDevice,
    pipeline: &ComputePipeline,
    resources: &FootprintResources,
) -> BindGroup {
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    device.create_bind_group(
        Some("prism_rt_footprint_group0"),
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: resources.footprints.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: resources.results.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: resources.params.as_entire_binding(),
            },
        ],
    )
}
