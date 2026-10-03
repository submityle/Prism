//! `PrepareBindGroups` system building the four surface-cache group-0 bind
//! groups per view.
//!
//! Mirrors [`super::super::world_space_gi::bind_groups`]: the reverse-Z scene
//! depth and packed `normal_roughness` come from the `SSR` prepass
//! ([`ViewSsrTextures`]), the pre-exposed scene colour from the visibility
//! buffer ([`ViewVisibilityBuffer`]), and the three surfel storage buffers plus
//! the `GI` export target from this subsystem's [`ViewSurfaceCache`]. All four
//! groups are present only when every backing resource is resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::SurfaceCachePipeline;
use super::resources::ViewSurfaceCache;

/// The four group-0 bind groups a single view's surface-cache passes record
/// against. Present only when the backing prepass, visibility and surfel
/// resources are all resident.
#[derive(Component)]
pub(crate) struct ViewSurfaceCacheBindGroups {
    /// group 0 for `surface_cache_alloc_main`: depth + normal + colour reads
    /// and the fresh scratch surfel buffer (read-write).
    alloc: BindGroup,
    /// group 0 for `surface_cache_update_main`: the fresh scratch buffer
    /// (read-only) and the persistent surfel buffer (read-write).
    update: BindGroup,
    /// group 0 for `surface_cache_spatial_filter_main`: the persistent surfel
    /// buffer (read-only) and the filtered scratch buffer (read-write).
    filter: BindGroup,
    /// group 0 for `surface_cache_coverage_main`: depth + normal reads, the
    /// filtered surfel buffer (read-only) and the `GI` export storage write.
    coverage: BindGroup,
}

impl ViewSurfaceCacheBindGroups {
    /// group-0 bind group for the `surface_cache_alloc_main` dispatch.
    pub(crate) fn alloc_group(&self) -> &BindGroup {
        &self.alloc
    }

    /// group-0 bind group for the `surface_cache_update_main` dispatch.
    pub(crate) fn update_group(&self) -> &BindGroup {
        &self.update
    }

    /// group-0 bind group for the `surface_cache_spatial_filter_main` dispatch.
    pub(crate) fn filter_group(&self) -> &BindGroup {
        &self.filter
    }

    /// group-0 bind group for the `surface_cache_coverage_main` dispatch.
    pub(crate) fn coverage_group(&self) -> &BindGroup {
        &self.coverage
    }
}

/// `PrepareBindGroups` system building [`ViewSurfaceCacheBindGroups`] for every
/// view whose `SSR` prepass, visibility buffer and surface-cache resources are
/// all resident.
pub(crate) fn prepare_surface_cache_bind_groups(
    mut commands: Commands,
    pipeline: Res<SurfaceCachePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewVisibilityBuffer,
        &ViewSurfaceCache,
    )>,
) {
    for (entity, ssr, visibility, sc) in &views {
        // alloc group: depth (0), normal_roughness (1), pre-exposed scene
        // colour (2), fresh scratch surfel buffer read-write (3).
        let alloc = device.create_bind_group(
            "prism surface cache alloc",
            pipeline.alloc_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                visibility.scene_color_view(),
                sc.scratch_current().as_entire_binding(),
            )),
        );

        // update group: fresh scratch buffer read-only (0), persistent surfel
        // buffer read-write (1).
        let update = device.create_bind_group(
            "prism surface cache update",
            pipeline.update_layout(),
            &BindGroupEntries::sequential((
                sc.scratch_current().as_entire_binding(),
                sc.surfels().as_entire_binding(),
            )),
        );

        // filter group: persistent surfel buffer read-only (0), filtered
        // scratch buffer read-write (1).
        let filter = device.create_bind_group(
            "prism surface cache spatial filter",
            pipeline.filter_layout(),
            &BindGroupEntries::sequential((
                sc.surfels().as_entire_binding(),
                sc.scratch_filtered().as_entire_binding(),
            )),
        );

        // coverage group: depth (0), normal_roughness (1), filtered surfel
        // buffer read-only (2), GI export storage write (3).
        let coverage = device.create_bind_group(
            "prism surface cache coverage",
            pipeline.coverage_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                sc.scratch_filtered().as_entire_binding(),
                sc.gi_out_view(),
            )),
        );

        commands.entity(entity).insert(ViewSurfaceCacheBindGroups {
            alloc,
            update,
            filter,
            coverage,
        });
    }
}
