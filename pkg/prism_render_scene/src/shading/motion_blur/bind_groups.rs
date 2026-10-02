//! Per-view group-0 bind groups for the three motion-blur passes.
//!
//! Mirrors [`super::super::ssr::reconstruct::prepare_ssr_reconstruct_bind_groups`]:
//! one `PrepareBindGroups` system builds the three bind groups a view needs from
//! its resident textures, present only when the visibility buffer (motion
//! vectors + scene colour), the SSR geometry prepass (device depth) and the
//! motion-blur tile/output textures are all live. Each group binds at the
//! shader's explicit indices, so `tile_max` is `sequential` `{0,1}` while
//! `neighbor_max` (`{2,3}`) and `reconstruct` (`{0,4,5,6,7}`) use
//! `with_indices`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::MotionBlurPipeline;
use super::resources::ViewMotionBlur;

/// The three group-0 bind groups a single view's motion-blur chain records
/// against. Present only when every backing texture is resident.
#[derive(Component)]
pub(crate) struct ViewMotionBlurBindGroups {
    /// group 0 for `tile_max`: motion-vector read + tile-max storage write.
    tile_max: BindGroup,
    /// group 0 for `neighbor_max`: tile-field read + dilated storage write.
    neighbor_max: BindGroup,
    /// group 0 for `reconstruct`: motion + scene colour + dilated tiles + depth
    /// reads, blurred storage write.
    reconstruct: BindGroup,
}

impl ViewMotionBlurBindGroups {
    /// group-0 bind group for the `tile_max` dispatch.
    pub(crate) fn tile_max(&self) -> &BindGroup {
        &self.tile_max
    }

    /// group-0 bind group for the `neighbor_max` dispatch.
    pub(crate) fn neighbor_max(&self) -> &BindGroup {
        &self.neighbor_max
    }

    /// group-0 bind group for the `reconstruct` dispatch.
    pub(crate) fn reconstruct(&self) -> &BindGroup {
        &self.reconstruct
    }
}

/// `PrepareBindGroups` system building [`ViewMotionBlurBindGroups`] for every
/// view whose visibility buffer, SSR textures and motion-blur textures are all
/// resident.
pub(crate) fn prepare_motion_blur_bind_groups(
    mut commands: Commands,
    pipeline: Res<MotionBlurPipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewSsrTextures,
        &ViewMotionBlur,
    )>,
) {
    for (entity, visibility, ssr, motion_blur) in &views {
        // TileMax: read the resolved motion-vector G-buffer, write the per-tile
        // longest velocity.
        let tile_max = device.create_bind_group(
            "prism motion blur tile max",
            pipeline.tile_max_layout(),
            &BindGroupEntries::sequential((
                visibility.motion_vectors_view(),
                motion_blur.tile_max_view(),
            )),
        );

        // NeighborMax: read the tile field (binding 2), write the dilated field
        // (binding 3).
        let neighbor_max = device.create_bind_group(
            "prism motion blur neighbor max",
            pipeline.neighbor_max_layout(),
            &BindGroupEntries::with_indices((
                (2, motion_blur.tile_max_view()),
                (3, motion_blur.neighbor_view()),
            )),
        );

        // Reconstruct: motion vectors (0), scene colour (4), dilated tiles (5),
        // SSR device depth (6), blurred output (7).
        let reconstruct = device.create_bind_group(
            "prism motion blur reconstruct",
            pipeline.reconstruct_layout(),
            &BindGroupEntries::with_indices((
                (0, visibility.motion_vectors_view()),
                (4, visibility.scene_color_view()),
                (5, motion_blur.neighbor_view()),
                (6, ssr.scene_depth_sampled()),
                (7, motion_blur.blur_out_view()),
            )),
        );

        commands.entity(entity).insert(ViewMotionBlurBindGroups {
            tile_max,
            neighbor_max,
            reconstruct,
        });
    }
}
