//! Per-view preparation of the shading-resolve bind groups.
//!
//! The material (group 1) and light (group 3) bind groups already live on the
//! shared [`MaterialBindGroup`]/[`LightBindGroup`] resources, so this stage only
//! builds the two pass-owned groups:
//!
//! * **group 0** — the two visibility textures plus the HDR storage-texture
//!   output, all sourced from the view's [`ViewVisibilityBuffer`].
//! * **group 2** — the per-view compacted worklist ([`ViewShadingBuffers`])
//!   spliced together with the render-world scene-instance and
//!   shading-geometry tables.
//!
//! Both groups are cleared to `None` (by dropping the component) unless every
//! upstream buffer is resident, so [`super::dispatch`] can treat a present
//! [`ViewResolveBindGroups`] as "safe to record".

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use crate::{GpuSceneBuffers, RenderShadingGeometryBuffers};

use super::super::resources::{ViewShadingBuffers, ViewVisibilityBuffer};
use super::pipeline::ShadingResolvePipeline;

/// The two pass-owned bind groups (group 0 + group 2) for one view.
///
/// Present only when every backing buffer/texture is resident; its absence is
/// the dispatch node's signal to skip the view this frame.
#[derive(Component)]
pub(crate) struct ViewResolveBindGroups {
    /// group 0: visibility ids/metadata + HDR storage-texture output.
    pub(crate) view: BindGroup,
    /// group 2: worklist + scene/geometry tables.
    pub(crate) scene: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewResolveBindGroups`] for every view
/// that has both a visibility buffer and a shading worklist, provided the
/// render-world scene-instance and shading-geometry tables have uploaded.
pub(crate) fn prepare_shading_resolve_bind_groups(
    mut commands: Commands,
    pipeline: Res<ShadingResolvePipeline>,
    device: Res<RenderDevice>,
    scene: Res<GpuSceneBuffers>,
    geometry: Res<RenderShadingGeometryBuffers>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewShadingBuffers)>,
) {
    // Scene/geometry tables are shared across all views; if either has not
    // uploaded yet there is nothing to resolve, so clear any stale groups.
    let (Some(instances), Some((geo_headers, geo_vertices, geo_primitives))) =
        (scene.instances(), geometry.buffers())
    else {
        for (entity, _, _) in &views {
            commands
                .entity(entity)
                .remove::<ViewResolveBindGroups>();
        }
        return;
    };

    for (entity, visibility, buffers) in &views {
        let (ids, metadata) = visibility.attachments();
        let view = device.create_bind_group(
            "prism resolve view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((ids, metadata, visibility.scene_color_view())),
        );
        let scene_group = device.create_bind_group(
            "prism resolve scene",
            &pipeline.scene_layout,
            &BindGroupEntries::sequential((
                buffers.work_items.as_entire_binding(),
                buffers.class_offsets.as_entire_binding(),
                buffers.class_counts.as_entire_binding(),
                instances.as_entire_binding(),
                geo_headers.as_entire_binding(),
                geo_vertices.as_entire_binding(),
                geo_primitives.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ViewResolveBindGroups {
            view,
            scene: scene_group,
        });
    }
}
