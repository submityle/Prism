//! Per-view preparation of the temporal-upscale passes' group-0 bind groups.
//!
//! Mirrors [`super::super::taa::bind_groups`]: for every view carrying a
//! resident [`ViewVisibilityBuffer`] (this frame's low-resolution
//! `render_color` + the `motion_vectors` G-buffer), a resident
//! [`ViewSsrTextures`] (the reverse-Z device `depth`) and a resolved
//! [`ViewUpscale`] (the three persistent ping-pong pairs plus `upscale_out`),
//! it assembles both passes' group-0 bind groups in the byte-identical order
//! their layouts and shaders declare.
//!
//! * The reconstruction group (ten entries) mirrors `upscale_reconstruct.wesl`:
//!   `render_color`(0), the sampled `history_color`(1) + its filtering
//!   sampler(2), `motion_vectors`(3), the render-res `depth`(4), the display-res
//!   `history_depth`(5) and `history_meta`(6), and the three write-only outputs
//!   `color_out`(7), `meta_out`(8) and `depth_out`(9).
//! * The RCAS group (two entries) mirrors `upscale_rcas.wesl`: the reconstructed
//!   `input_color`(0) — the reconstruction's `color_out` slot — and the
//!   write-only `upscale_out`(1).

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::UpscalePipeline;
use super::resources::ViewUpscale;

/// Both temporal-upscale passes' group-0 bind groups for a single view. Present
/// only when the visibility buffer (`render_color` + motion), the SSR device
/// depth and the ping-pong history are all resident.
#[derive(Component)]
pub(crate) struct ViewUpscaleBindGroups {
    /// The reconstruction pass's ten-entry group-0 bind group.
    reconstruct: BindGroup,
    /// The RCAS pass's two-entry group-0 bind group.
    rcas: BindGroup,
}

impl ViewUpscaleBindGroups {
    /// The reconstruction pass's group-0 bind group.
    pub(crate) fn reconstruct(&self) -> &BindGroup {
        &self.reconstruct
    }

    /// The RCAS pass's group-0 bind group.
    pub(crate) fn rcas(&self) -> &BindGroup {
        &self.rcas
    }
}

/// `PrepareBindGroups` system building [`ViewUpscaleBindGroups`] for every view
/// with a resident [`ViewVisibilityBuffer`], [`ViewSsrTextures`] and resolved
/// [`ViewUpscale`].
pub(crate) fn prepare_upscale_bind_groups(
    mut commands: Commands,
    pipeline: Res<UpscalePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures, &ViewUpscale)>,
) {
    for (entity, visibility, ssr, upscale) in &views {
        // Order mirrors `upscale_reconstruct.wesl`: render_color(0),
        // history_color(1), sampler(2), motion_vectors(3), depth(4),
        // history_depth(5), history_meta(6), color_out(7), meta_out(8),
        // depth_out(9).
        let reconstruct = device.create_bind_group(
            "prism upscale reconstruct",
            pipeline.reconstruct_layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                upscale.history_color_read(),
                pipeline.sampler(),
                visibility.motion_vectors_view(),
                ssr.scene_depth_sampled(),
                upscale.history_depth_read(),
                upscale.history_meta_read(),
                upscale.history_color_write(),
                upscale.history_meta_write(),
                upscale.history_depth_write(),
            )),
        );

        // Order mirrors `upscale_rcas.wesl`: input_color(0), color_out(1). The
        // reconstruction's `color_out` slot is RCAS's `input_color`, and RCAS
        // writes the finished `upscale_out`.
        let rcas = device.create_bind_group(
            "prism upscale rcas",
            pipeline.rcas_layout(),
            &BindGroupEntries::sequential((
                upscale.history_color_write(),
                upscale.upscale_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewUpscaleBindGroups { reconstruct, rcas });
    }
}
