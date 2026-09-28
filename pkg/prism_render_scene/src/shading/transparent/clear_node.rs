//! `Core3d` graph node clearing the WBOIT targets to their identity each frame.
//!
//! Weighted-blended OIT accumulates with additive (`accum`) and multiplicative
//! (`revealage`) blending, so both targets must start each frame at the blend
//! identity:
//!
//! * `accum` -> `(0, 0, 0, 0)` (additive identity),
//! * `revealage` -> `(1, 1, 1, 1)` (multiplicative identity; nothing occluded).
//!
//! This interim node opens the MRT render pass, clears both attachments and
//! ends. Until the transparent forward *draw* pass exists (next slice) nothing
//! writes the targets, so after this clear the composite reads accum.a = 0 and
//! revealage = 1, emits coverage 0 and leaves the view target untouched - a
//! correct no-op. Once the forward draw lands it will own the `LoadOp::Clear`
//! and this node is removed.
//!
//! It is gated by the same `enable_visibility_buffer` / single-sample check as
//! the rest of the chain and skips any view without [`ViewOitTargets`].

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
    },
    renderer::{RenderContext, ViewQuery},
    view::Msaa,
};

use super::super::runtime::PrismShadingSettings;
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

pub(crate) fn clear_oit_targets(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewOitTargets, Option<&Msaa>)>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let (targets, msaa) = view.into_inner();
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }
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
    // Opening and ending the pass performs the clear; no draws yet.
    let _pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism oit clear"),
        color_attachments: &colors,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}
