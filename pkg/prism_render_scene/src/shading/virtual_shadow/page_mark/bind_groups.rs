//! `PrepareBindGroups` assembly of the page-request (`page-mark`) pass's
//! group-0 bind group.
//!
//! Mirrors [`super::super::bind_groups`] (the receiver-generation pass): for
//! every view carrying both the prepared [`super::super::resources::ViewVsmReceivers`]
//! receiver buffer and the prepared [`super::resources::ViewVsmPageRequests`]
//! request bitmap, it assembles the two-entry group-0 bind group in the
//! byte-identical order the pipeline layout and `vsm_page_mark.wesl` declare:
//!
//! * binding 0 -- the read-only per-pixel receiver `storage` array produced by
//!   the receiver-generation pass, and
//! * binding 1 -- the read-write per-window-slot request bitmap
//!   (`array<atomic<u32>>`) this pass marks with `atomicOr`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::super::runtime::PrismShadingSettings;
use super::pipeline::VsmPageMarkPipeline;
use super::resources::ViewVsmPageRequests;
use super::super::resources::ViewVsmReceivers;

/// The page-mark pass's group-0 bind group for a single view. Present only when
/// both the receiver buffer and the request bitmap are resident this frame.
#[derive(Component)]
pub(crate) struct ViewVsmPageMarkBindGroup {
    group: BindGroup,
}

impl ViewVsmPageMarkBindGroup {
    /// The prepared group-0 bind group bound by the page-mark dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewVsmPageMarkBindGroup`] for every
/// view with prepared [`ViewVsmReceivers`] and [`ViewVsmPageRequests`] buffers.
///
/// Gated on [`PrismShadingSettings::enable_virtual_shadow`]; the per-view
/// [`ViewVsmPageRequests`] only exists on views the prepare step ran while VSM
/// was enabled, so a disabled frame matches no views regardless, but the gate
/// keeps this pass's activation contract identical to the dispatch's.
pub(crate) fn prepare_vsm_page_mark_bind_groups(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    pipeline: Res<VsmPageMarkPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVsmReceivers, &ViewVsmPageRequests)>,
) {
    if !settings.enable_virtual_shadow {
        return;
    }

    for (entity, receivers, requests) in &views {
        let group = device.create_bind_group(
            "prism VSM page-mark",
            pipeline.layout(),
            // Order mirrors `vsm_page_mark.wesl`: receivers(0), page_requests(1).
            &BindGroupEntries::sequential((
                receivers.receivers_buffer().as_entire_binding(),
                requests.requests_buffer().as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewVsmPageMarkBindGroup { group });
    }
}
