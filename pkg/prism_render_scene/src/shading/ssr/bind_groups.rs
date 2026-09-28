//! Per-view preparation of the SSR geometry-prepass bind groups.
//!
//! Builds the two pass-owned groups for every view that has both a resident
//! visibility buffer and the [`ViewSsrTextures`] the prepare stage allocates:
//!
//! * **group 0** — the two visibility textures read plus the device-depth /
//!   view-normal targets written, sourced from the view's
//!   [`ViewVisibilityBuffer`] and [`ViewSsrTextures`].
//! * **group 1** — the shared render-world scene-instance and shading-geometry
//!   tables the surface reconstruction walks.
//!
//! The component is removed unless every upstream buffer/texture is resident,
//! so [`super::dispatch`] can treat a present [`ViewSsrPrepassBindGroups`] as
//! "safe to record". Mirrors [`super::super::ao::prepare_gtao_prepass_bind_groups`].

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use crate::{GpuSceneBuffers, RenderShadingGeometryBuffers};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::SsrPrepassPipeline;
use super::resources::ViewSsrTextures;

/// The two pass-owned bind groups (group 0 + group 1) for one view's SSR
/// prepass. Present only when every backing buffer/texture is resident.
#[derive(Component)]
pub(crate) struct ViewSsrPrepassBindGroups {
    /// group 0: visibility ids/metadata + device-depth/view-normal outputs.
    pub(crate) view: BindGroup,
    /// group 1: scene instances + geometry tables.
    pub(crate) scene: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrPrepassBindGroups`] for every
/// view that has both a visibility buffer and SSR textures, provided the shared
/// scene-instance and shading-geometry tables have uploaded.
pub(crate) fn prepare_ssr_prepass_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrPrepassPipeline>,
    device: Res<RenderDevice>,
    scene: Res<GpuSceneBuffers>,
    geometry: Res<RenderShadingGeometryBuffers>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures)>,
) {
    // Scene/geometry tables are shared across all views; without them there is
    // nothing to reconstruct, so drop any stale groups.
    let (Some(instances), Some((geo_headers, geo_vertices, geo_primitives))) =
        (scene.instances(), geometry.buffers())
    else {
        for (entity, _, _) in &views {
            commands
                .entity(entity)
                .remove::<ViewSsrPrepassBindGroups>();
        }
        return;
    };

    for (entity, visibility, textures) in &views {
        let (ids, metadata) = visibility.attachments();
        let view = device.create_bind_group(
            "prism SSR prepass view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                ids,
                metadata,
                textures.scene_depth_view(),
                textures.view_normal_view(),
            )),
        );
        let scene_group = device.create_bind_group(
            "prism SSR prepass scene",
            &pipeline.scene_layout,
            &BindGroupEntries::sequential((
                instances.as_entire_binding(),
                geo_headers.as_entire_binding(),
                geo_vertices.as_entire_binding(),
                geo_primitives.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ViewSsrPrepassBindGroups {
            view,
            scene: scene_group,
        });
    }
}
