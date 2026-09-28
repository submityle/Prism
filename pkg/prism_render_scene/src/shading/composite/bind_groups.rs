//! Per-view preparation of the composite bind group.
//!
//! The composite pass owns a single bind group (**group 0**), holding the
//! three resolve outputs that live on the view's [`ViewVisibilityBuffer`]:
//!
//! * binding 0 — `scene_color`, the linear HDR radiance the resolve compute
//!   pass wrote (sampled as a non-filterable float texture),
//! * binding 1 — `visibility_ids`, and
//! * binding 2 — `visibility_metadata`, both integer targets the fragment
//!   shader reads to distinguish covered pixels from the background sentinel,
//!   and
//! * binding 3 — the persistent per-view exposure state (read-only storage),
//!   whose `adapted_exposure` multiplier the fragment applies to the radiance
//!   before writing it out (a stationary `1.0` when auto-exposure is disabled).
//!
//! The entry order here is byte-identical to the `@group(0) @binding(N)`
//! declarations in `shaders/composite.wesl`, and the layout is fetched from the
//! [`PipelineCache`] using the *same* descriptor the pipeline specializes with,
//! so the bind group can never drift from the pipeline it feeds.
//!
//! A view whose [`ViewVisibilityBuffer`] was dropped this frame (the Prism path
//! is disabled or MSAA is active) has nothing to composite and no live textures
//! to bind, so its stale [`ViewCompositeBindGroup`] is removed. [`super::node`]
//! treats a present component as "safe to record".

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, PipelineCache},
    renderer::RenderDevice,
    view::ExtractedView,
};

use super::super::exposure::ViewExposureBuffers;
use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::ShadingCompositePipeline;

/// The composite pass' group-0 bind group for one view.
///
/// Present only when the view's resolve outputs are resident; its absence is
/// the composite node's signal to skip the view this frame.
#[derive(Component)]
pub(crate) struct ViewCompositeBindGroup(pub(crate) BindGroup);

/// `PrepareBindGroups` system building [`ViewCompositeBindGroup`] for every
/// view whose [`ViewVisibilityBuffer`] (and thus the resolve outputs it holds)
/// has been allocated this frame, and dropping the component for views whose
/// buffer is gone so no bind group can outlive the textures it references.
pub(crate) fn prepare_shading_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<ShadingCompositePipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    views: Query<
        (
            Entity,
            Option<&ViewVisibilityBuffer>,
            Option<&super::super::taa::ViewTaa>,
            Option<&ViewExposureBuffers>,
        ),
        With<ExtractedView>,
    >,
) {
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    for (entity, visibility, taa, exposure) in &views {
        let Some(visibility) = visibility else {
            commands.entity(entity).remove::<ViewCompositeBindGroup>();
            continue;
        };
        // The exposure state buffer is created for every view with a resident
        // visibility buffer and a known viewport; if it has not landed yet (a
        // view still without an extent on its first frame) there is nothing to
        // composite this frame, so drop any stale bind group and wait.
        let Some(exposure) = exposure else {
            commands.entity(entity).remove::<ViewCompositeBindGroup>();
            continue;
        };
        let (ids, metadata) = visibility.attachments();
        // When TAA resolved this frame it wrote the anti-aliased result into its
        // ping-pong write slot; composite reads that in place of the raw
        // `scene_color` so the presented image is the temporally resolved one.
        let scene_color = match taa {
            Some(taa) => taa.write_view(),
            None => visibility.scene_color_view(),
        };
        let bind_group = device.create_bind_group(
            "prism composite",
            &layout,
            // Order mirrors `composite.wesl`: scene_color(0), ids(1),
            // metadata(2), exposure_state(3).
            &BindGroupEntries::sequential((
                scene_color,
                ids,
                metadata,
                exposure.state_buffer().as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewCompositeBindGroup(bind_group));
    }
}
