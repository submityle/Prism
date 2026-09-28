//! Per-view preparation of the TAA resolve's group-0 bind group.
//!
//! Mirrors [`super::super::ssr::temporal`]'s bind-group preparation: for every
//! view carrying both a resident [`ViewVisibilityBuffer`] (holding the
//! composited `scene_color` and the motion-vector G-buffer) and a resolved
//! [`ViewTaa`] (the ping-pong history slots), it assembles the five-entry
//! group-0 bind group in the byte-identical order the resolve pipeline layout
//! and `taa_resolve.wesl` declare:
//!
//! * binding 0 — this frame's composited `scene_color` (`textureLoad`ed),
//! * binding 1 — the previous-frame history (sampled at the reprojected UV),
//! * binding 2 — the bilinear clamp history sampler,
//! * binding 3 — the write-only `rgba16float` resolved output, and
//! * binding 4 — the resolve's `rg16float` motion-vector G-buffer.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::TaaResolvePipeline;
use super::resources::ViewTaa;

/// The TAA resolve's group-0 bind group for a single view. Present only when
/// both the visibility buffer (`scene_color` + motion) and the ping-pong
/// history are resident.
#[derive(Component)]
pub(crate) struct ViewTaaBindGroup {
    group: BindGroup,
}

impl ViewTaaBindGroup {
    /// The prepared group-0 bind group bound by the resolve dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewTaaBindGroup`] for every view with
/// a resident [`ViewVisibilityBuffer`] and a resolved [`ViewTaa`].
pub(crate) fn prepare_taa_bind_groups(
    mut commands: Commands,
    pipeline: Res<TaaResolvePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewTaa)>,
) {
    for (entity, visibility, taa) in &views {
        let group = device.create_bind_group(
            "prism TAA resolve",
            pipeline.layout(),
            // Order mirrors `taa_resolve.wesl`: scene_color(0), history(1),
            // sampler(2), taa_out(3), motion(4).
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                taa.read_view(),
                pipeline.sampler(),
                taa.write_view(),
                visibility.motion_vectors_view(),
            )),
        );
        commands.entity(entity).insert(ViewTaaBindGroup { group });
    }
}
