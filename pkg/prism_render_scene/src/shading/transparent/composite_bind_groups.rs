//! Per-view preparation of the OIT composite bind group.
//!
//! The composite pass owns a single bind group (**group 0**) holding the two
//! WBOIT targets the transparent forward pass wrote, both living on the view
//! [`ViewOitTargets`]:
//!
//! * binding 0 - `oit_accum`, weighted premultiplied colour (`rgb`) + summed
//!   weighted alpha (`a`),
//! * binding 1 - `oit_revealage`, the running product of `1 - alpha`.
//!
//! Both are sampled as non-filterable float textures by integer `textureLoad`.
//! The entry order here is byte-identical to the `@group(0) @binding(N)`
//! declarations in `shaders/oit.wesl`, and the layout is fetched from the
//! [`PipelineCache`] using the *same* descriptor the pipeline specializes with,
//! so the bind group can never drift from the pipeline it feeds.
//!
//! A view whose [`ViewOitTargets`] were dropped this frame (the Prism path is
//! disabled or MSAA is active) has nothing to composite; its stale
//! [`ViewOitCompositeBindGroup`] is removed. [`super::composite_node`] treats a
//! present component as "safe to record".

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, PipelineCache},
    renderer::RenderDevice,
    view::ExtractedView,
};

use super::composite_pipeline::OitCompositePipeline;
use super::targets::ViewOitTargets;

/// The OIT composite pass group-0 bind group for one view.
///
/// Present only when the view WBOIT targets are resident; its absence is the
/// composite node signal to skip the view this frame.
#[derive(Component)]
pub(crate) struct ViewOitCompositeBindGroup(pub(crate) BindGroup);

/// `PrepareBindGroups` system building [`ViewOitCompositeBindGroup`] for every
/// view whose [`ViewOitTargets`] have been allocated this frame, and dropping
/// the component for views whose targets are gone so no bind group can outlive
/// the textures it references.
pub(crate) fn prepare_oit_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<OitCompositePipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, Option<&ViewOitTargets>), With<ExtractedView>>,
) {
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    for (entity, targets) in &views {
        let Some(targets) = targets else {
            commands
                .entity(entity)
                .remove::<ViewOitCompositeBindGroup>();
            continue;
        };
        let bind_group = device.create_bind_group(
            "prism oit composite",
            &layout,
            // Order mirrors oit.wesl: oit_accum(0), oit_revealage(1).
            &BindGroupEntries::sequential((targets.accum_view(), targets.revealage_view())),
        );
        commands
            .entity(entity)
            .insert(ViewOitCompositeBindGroup(bind_group));
    }
}
