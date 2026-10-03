//! `Core3d` scheduling system recording the four surface-cache dispatches for
//! every view.
//!
//! Mirrors [`super::super::world_space_gi::dispatch`]: gate on the enable,
//! resolve the view, recover the inverse projection + near plane from the
//! [`ExtractedView`], then run the four same-frame passes in order:
//!
//! 1. `surface_cache_alloc_main` — one workgroup per 64 *surfels*; seed a fresh
//!    surfel per screen tile into the scratch-current buffer.
//! 2. `surface_cache_update_main` — one workgroup per 64 *surfels*; blend the
//!    fresh surfel into the persistent surfel buffer via the golden `EMA`.
//! 3. `surface_cache_spatial_filter_main` — one workgroup per 64 *surfels*;
//!    bilaterally filter the persistent surfels into the filtered scratch.
//! 4. `surface_cache_coverage_main` — one workgroup per 8x8 *pixel* block;
//!    gather the filtered surfels per pixel into the `GI` export.
//!
//! Every entry point bounds-checks its invocation, so the `div_ceil` rounding
//! that over-dispatches the final workgroup is safe. The persistent surfel
//! buffers are read and written within this same frame's passes, so there is no
//! `+1`-frame latency; the composite stage folds `gi_out` over `scene_color`.

use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{SURFACE_CACHE_WORKGROUP_SIZE_1D, SURFACE_CACHE_WORKGROUP_SIZE_2D};
use super::bind_groups::ViewSurfaceCacheBindGroups;
use super::pipeline::SurfaceCachePipeline;
use super::resources::ViewSurfaceCache;
use super::settings::PrismSurfaceCacheSettings;

/// `Core3d` scheduling system recording the alloc -> update -> spatial-filter
/// -> coverage dispatches for every view whose surface-cache resources and bind
/// groups are resident.
pub(crate) fn surface_cache_pass(
    settings: Res<PrismSurfaceCacheSettings>,
    view: ViewQuery<(
        &ViewSurfaceCache,
        &ViewSurfaceCacheBindGroups,
        &ExtractedView,
    )>,
    pipeline: Res<SurfaceCachePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (sc, groups, extracted) = view.into_inner();

    // All four pipelines must be resident before the pass runs.
    let Some(alloc_pipeline) = cache.get_compute_pipeline(pipeline.alloc()) else {
        return;
    };
    let Some(update_pipeline) = cache.get_compute_pipeline(pipeline.update()) else {
        return;
    };
    let Some(filter_pipeline) = cache.get_compute_pipeline(pipeline.filter()) else {
        return;
    };
    let Some(coverage_pipeline) = cache.get_compute_pipeline(pipeline.coverage()) else {
        return;
    };

    let size = sc.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection (clip -> view) used to reconstruct view-space
    // positions in the alloc and coverage passes; near plane recovered the same
    // way as world-space GI / SSGI.
    let view_from_clip = extracted.clip_from_view.inverse();
    // Reverse-Z: device depth 1.0 is the near plane, so inverse-projecting
    // clip (0, 0, 1, 1) yields a view-space point at `-near` along `-Z`.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };

    let grid = sc.surfel_grid;
    let surfel_count = grid.x * grid.y;
    if surfel_count == 0 {
        return;
    }
    let surfel_groups = surfel_count.div_ceil(SURFACE_CACHE_WORKGROUP_SIZE_1D);
    let pixel_groups_x = size.x.div_ceil(SURFACE_CACHE_WORKGROUP_SIZE_2D);
    let pixel_groups_y = size.y.div_ceil(SURFACE_CACHE_WORKGROUP_SIZE_2D);

    let alloc_params = settings.alloc_params(size, grid, view_from_clip, near);
    let update_params = settings.update_params(surfel_count);
    let filter_params = settings.filter_params(grid, surfel_count);
    let coverage_params = settings.coverage_params(size, grid, view_from_clip, near);

    let encoder = ctx.command_encoder();

    // Pass 1: seed one fresh surfel per screen tile into scratch_current.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache alloc"),
            timestamp_writes: None,
        });
        pass.set_pipeline(alloc_pipeline);
        pass.set_bind_group(0, groups.alloc_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&alloc_params));
        pass.dispatch_workgroups(surfel_groups, 1, 1);
    }

    // Pass 2: blend scratch_current into the persistent surfel buffer (EMA).
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(update_pipeline);
        pass.set_bind_group(0, groups.update_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&update_params));
        pass.dispatch_workgroups(surfel_groups, 1, 1);
    }

    // Pass 3: bilaterally filter the persistent surfels into scratch_filtered.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache spatial filter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(filter_pipeline);
        pass.set_bind_group(0, groups.filter_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&filter_params));
        pass.dispatch_workgroups(surfel_groups, 1, 1);
    }

    // Pass 4: gather the filtered surfels per pixel into the GI export.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache coverage"),
            timestamp_writes: None,
        });
        pass.set_pipeline(coverage_pipeline);
        pass.set_bind_group(0, groups.coverage_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&coverage_params));
        pass.dispatch_workgroups(pixel_groups_x, pixel_groups_y, 1);
    }
}
