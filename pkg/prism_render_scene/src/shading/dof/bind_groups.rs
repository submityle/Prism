//! Per-view group-0 bind groups for the three depth-of-field passes.
//!
//! Mirrors [`super::super::motion_blur::bind_groups`]: one `PrepareBindGroups`
//! system builds the three bind groups a view needs from its resident textures,
//! present only when the visibility buffer (pre-exposed scene colour), the SSR
//! geometry prepass (device depth) and the `DoF` CoC/blurred/output textures are
//! all live. Each group binds at the shader's explicit indices, so `dof_coc` is
//! `sequential` `{0,1}` while `dof_gather` (`{2,3,4}`) and `dof_composite`
//! (`{2,3,5,6}`) use `with_indices`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::DofPipeline;
use super::resources::ViewDof;

/// The three group-0 bind groups a single view's `DoF` chain records against.
/// Present only when every backing texture is resident.
#[derive(Component)]
pub(crate) struct ViewDofBindGroups {
    /// group 0 for `dof_coc`: depth read + `CoC` storage write.
    coc: BindGroup,
    /// group 0 for `dof_gather`: scene-colour + `CoC` reads, blurred storage write.
    gather: BindGroup,
    /// group 0 for `dof_composite`: scene-colour + `CoC` + blurred reads, output
    /// storage write.
    composite: BindGroup,
}

impl ViewDofBindGroups {
    /// group-0 bind group for the `dof_coc` dispatch.
    pub(crate) fn coc(&self) -> &BindGroup {
        &self.coc
    }

    /// group-0 bind group for the `dof_gather` dispatch.
    pub(crate) fn gather(&self) -> &BindGroup {
        &self.gather
    }

    /// group-0 bind group for the `dof_composite` dispatch.
    pub(crate) fn composite(&self) -> &BindGroup {
        &self.composite
    }
}

/// `PrepareBindGroups` system building [`ViewDofBindGroups`] for every view
/// whose visibility buffer, SSR textures and `DoF` textures are all resident.
pub(crate) fn prepare_dof_bind_groups(
    mut commands: Commands,
    pipeline: Res<DofPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures, &ViewDof)>,
) {
    for (entity, visibility, ssr, dof) in &views {
        // CoC prepass: read the SSR reverse-Z device depth (0), write the CoC
        // field (1).
        let coc = device.create_bind_group(
            "prism dof coc",
            pipeline.coc_layout(),
            &BindGroupEntries::sequential((ssr.scene_depth_sampled(), dof.coc_view())),
        );

        // Gather: read the pre-exposed scene colour (2) and the CoC field (3),
        // write the blurred field (4).
        let gather = device.create_bind_group(
            "prism dof gather",
            pipeline.gather_layout(),
            &BindGroupEntries::with_indices((
                (2, visibility.scene_color_view()),
                (3, dof.coc_view()),
                (4, dof.blurred_view()),
            )),
        );

        // Composite: read the sharp scene colour (2), the CoC field (3) and the
        // blurred field (5), write the composited output (6).
        let composite = device.create_bind_group(
            "prism dof composite",
            pipeline.composite_layout(),
            &BindGroupEntries::with_indices((
                (2, visibility.scene_color_view()),
                (3, dof.coc_view()),
                (5, dof.blurred_view()),
                (6, dof.dof_out_view()),
            )),
        );

        commands.entity(entity).insert(ViewDofBindGroups {
            coc,
            gather,
            composite,
        });
    }
}
