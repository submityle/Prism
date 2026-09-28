//! `Core3d` node running the transparent forward (WBOIT) draw pass.
//!
//! Opens the two WBOIT MRT targets plus the shared opaque depth buffer, clears
//! the targets to the blend identity and renders the [`TransparentOit3d`] phase
//! into them. This replaces the interim clear-only node: the `LoadOp::Clear`
//! here now owns the per-frame reset *and* the draws write into it in one pass.
//!
//! Clears:
//! * accum -> `(0, 0, 0, 0)` (additive identity);
//! * revealage -> `(1, 1, 1, 1)` (multiplicative identity; nothing occluded).
//!
//! Depth: `get_attachment(StoreOp::Store)` returns `LoadOp::Load` here because
//! the visibility raster already consumed the depth texture's first-use flag
//! earlier this frame. The pipeline tests (reverse-Z `GreaterEqual`) against the
//! opaque depth but never writes it (`depth_write_enabled = false`), so opaque
//! geometry occludes transparency without transparent fragments perturbing the
//! depth used by anything downstream.
//!
//! Gated by the same `enable_visibility_buffer` / single-sample check as the
//! rest of the visibility chain; a view without [`ViewOitTargets`] is skipped.

use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::ViewBinnedRenderPhases,
    render_resource::{
        LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
    },
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, Msaa, ViewDepthStencilTexture},
};

use super::super::runtime::PrismShadingSettings;
use super::phase::TransparentOit3d;
use super::targets::ViewOitTargets;

/// Additive-blend identity for the accumulation target.
const OIT_ACCUM_CLEAR: wgpu_types::Color = wgpu_types::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

/// Multiplicative-blend identity for the revealage target: 1.0 means no
/// transparent coverage has occluded the background yet.
const OIT_REVEALAGE_CLEAR: wgpu_types::Color = wgpu_types::Color {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

pub(crate) fn transparent_forward_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthStencilTexture,
        &ViewOitTargets,
        Option<&Msaa>,
    )>,
    phases: Res<ViewBinnedRenderPhases<TransparentOit3d>>,
    settings: Res<PrismShadingSettings>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let view_entity = view.entity();
    let (camera, extracted, depth, targets, msaa) = view.into_inner();
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }
    let Some(phase) = phases.get(&extracted.retained_view_entity) else {
        return;
    };
    let (accum, revealage) = targets.attachments();
    let colors = [
        Some(RenderPassColorAttachment {
            view: accum,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(OIT_ACCUM_CLEAR),
                store: StoreOp::Store,
            },
        }),
        Some(RenderPassColorAttachment {
            view: revealage,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(OIT_REVEALAGE_CLEAR),
                store: StoreOp::Store,
            },
        }),
    ];
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism oit forward"),
        color_attachments: &colors,
        // Load the opaque depth: test against it, never write it.
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }
    let _ = phase.render(&mut pass, world, view_entity);
}
