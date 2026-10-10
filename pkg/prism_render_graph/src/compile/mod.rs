//! The compiler: lowering a declared frame graph into an [`ExecutionPlan`].
//!
//! Compilation is a pure, allocation-light pipeline over the graph's borrowed
//! pass and resource tables. It never touches the move-only execute closures
//! and never allocates GPU memory, so it is fully deterministic and testable
//! against a mock device. The stages run in a fixed order, each consuming the
//! previous stage's output:
//!
//! 1. [`cull`] — drop passes whose results nobody observes.
//! 2. [`schedule`] — linearize the survivors into a deterministic order.
//! 3. [`lifetime`] — compute each resource's live interval over that order.
//! 4. [`alias`] — pack non-overlapping transients into a shared heap.
//! 5. [`barrier`] — replay the schedule through the driver state tracker to
//!    derive the minimal per-pass synchronization.
//!
//! A final pass derives each raster pass's [`RasterAttachments`] — resolving
//! load/store from SSA versioning and resource lifetime — so execution can
//! build a driver render pass without re-deriving anything.
//!
//! [`ExecutionPlan`]: crate::ExecutionPlan

pub mod alias;
pub mod barrier;
pub mod cull;
pub mod lifetime;
pub mod schedule;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{Color, DepthLoadOp, Extent3d, LoadOp, StoreOp};

use crate::access::TextureUse;
use crate::pass::{PassKind, PassNode};
use crate::plan::{
    ColorAttachmentPlan, CompileError, DepthAttachmentPlan, ExecutionPlan, RasterAttachments,
};
use crate::resource::{BufferResource, Lifetime, TextureResource};

/// Compiles a declared frame into an immutable [`ExecutionPlan`].
///
/// Borrows the graph's pass and resource tables and the frame's `swapchain`
/// extent (which resolves screen-relative sizes). Returns [`CompileError`] only
/// when the derived dependency graph contains a cycle; a well-formed graph
/// always compiles.
pub(crate) fn compile(
    passes: &[PassNode],
    textures: &[TextureResource],
    buffers: &[BufferResource],
    swapchain: Extent3d,
) -> Result<ExecutionPlan, CompileError> {
    let alive = cull::cull(passes, textures, buffers);
    let order = schedule::schedule(passes, &alive)?;

    // A resource is "used" if any scheduled pass touches it; only used
    // resources are realized, state-tracked, and aliased.
    let mut used_textures = vec![false; textures.len()];
    let mut used_buffers = vec![false; buffers.len()];
    for &pidx in &order {
        for acc in &passes[pidx].texture_accesses {
            used_textures[acc.resource.get() as usize] = true;
        }
        for acc in &passes[pidx].buffer_accesses {
            used_buffers[acc.resource.get() as usize] = true;
        }
    }

    let lifetimes = lifetime::analyze(passes, &order, textures.len(), buffers.len());
    let alias = alias::plan_alias(textures, buffers, &lifetimes, order.len(), swapchain);
    let barrier_plan = barrier::plan_barriers(
        passes,
        textures,
        buffers,
        &order,
        &used_textures,
        &used_buffers,
        swapchain,
    );

    // A version is "consumed" if any scheduled pass observes it on entry; its
    // producing attachment must therefore store rather than discard.
    let mut consumed: BTreeSet<(u32, u32)> = BTreeSet::new();
    for &pidx in &order {
        for acc in &passes[pidx].texture_accesses {
            consumed.insert((acc.resource.get(), acc.input_version));
        }
    }

    let attachments = derive_attachments(passes, textures, &order, &consumed);

    Ok(ExecutionPlan {
        order,
        alive,
        barriers: barrier_plan.per_pass,
        final_barriers: barrier_plan.final_barriers,
        attachments,
        alias,
        used_textures,
        used_buffers,
    })
}

/// Derives the resolved color/depth attachments of every scheduled raster pass.
///
/// Non-raster passes map to `None`. For each raster pass, the attachments are
/// taken from its texture accesses in declaration order; load/store are derived
/// from SSA versioning (clear a freshly-written target, otherwise load) and
/// consumption (store only what a later pass reads or what outlives the frame).
fn derive_attachments(
    passes: &[PassNode],
    textures: &[TextureResource],
    order: &[usize],
    consumed: &BTreeSet<(u32, u32)>,
) -> Vec<Option<RasterAttachments>> {
    let mut out = Vec::with_capacity(order.len());
    for &pidx in order {
        let pass = &passes[pidx];
        if pass.kind != PassKind::Raster {
            out.push(None);
            continue;
        }

        let mut colors: Vec<ColorAttachmentPlan> = Vec::new();
        let mut depth: Option<DepthAttachmentPlan> = None;
        for acc in &pass.texture_accesses {
            if !acc.usage.is_attachment() {
                continue;
            }
            let res = acc.resource;
            let lifetime = textures[res.get() as usize].lifetime;
            match acc.usage {
                TextureUse::ColorAttachment | TextureUse::ColorAttachmentRead => {
                    let writes = acc.usage == TextureUse::ColorAttachment;
                    let version = if writes {
                        acc.output_version
                    } else {
                        acc.input_version
                    };
                    let load = if writes && acc.input_version == 0 {
                        LoadOp::Clear(Color::BLACK)
                    } else {
                        LoadOp::Load
                    };
                    colors.push(ColorAttachmentPlan {
                        resource: res,
                        output_version: version,
                        load,
                        store: store_op(lifetime, res.get(), version, consumed),
                    });
                }
                TextureUse::DepthAttachment | TextureUse::DepthAttachmentRead => {
                    let writes = acc.usage == TextureUse::DepthAttachment;
                    let version = if writes {
                        acc.output_version
                    } else {
                        acc.input_version
                    };
                    let depth_load = if writes && acc.input_version == 0 {
                        DepthLoadOp::Clear(1.0)
                    } else {
                        DepthLoadOp::Load
                    };
                    depth = Some(DepthAttachmentPlan {
                        resource: res,
                        output_version: version,
                        writes,
                        depth_load,
                        depth_store: store_op(lifetime, res.get(), version, consumed),
                    });
                }
                // Input attachments are tile-local reads; they shape subpass
                // merging but add no color/depth slot to the render pass.
                _ => {}
            }
        }

        out.push(Some(RasterAttachments { colors, depth }));
    }
    out
}

/// Resolves an attachment's store op: keep the result only when a later pass
/// consumes the written version or the resource's contents outlive the frame.
fn store_op(
    lifetime: Lifetime,
    resource: u32,
    version: u32,
    consumed: &BTreeSet<(u32, u32)>,
) -> StoreOp {
    if matches!(lifetime, Lifetime::Imported | Lifetime::Persistent)
        || consumed.contains(&(resource, version))
    {
        StoreOp::Store
    } else {
        StoreOp::Discard
    }
}
