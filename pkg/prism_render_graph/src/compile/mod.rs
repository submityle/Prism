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
pub(crate) mod kernel;
pub mod lifetime;
pub mod schedule;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{Color, DepthLoadOp, Extent3d, LoadOp, StoreOp};

use crate::access::TextureUse;
use crate::capacity::GraphLimits;
use crate::pass::{PassKind, PassNode};
use crate::plan::{
    ColorAttachmentPlan, CompileError, CompiledAccess, DepthAttachmentPlan, ExecutionPlan,
    LogicalBarrier, LogicalResource, RasterAttachments,
};
use crate::resource::{BufferResource, Lifetime, TextureResource};

/// Compiles a declared frame into an immutable [`ExecutionPlan`].
///
/// Borrows the graph's pass and resource tables and the frame's `swapchain`
/// extent (which resolves screen-relative sizes). Validation happens before
/// culling; footprint arithmetic and dependency cycles are reported as errors.
pub(crate) fn compile(
    passes: &[PassNode],
    textures: &[TextureResource],
    buffers: &[BufferResource],
    swapchain: Extent3d,
) -> Result<ExecutionPlan, CompileError> {
    GraphLimits::check_count("passes", passes.len(), u32::MAX as usize)?;
    GraphLimits::check_count("textures", textures.len(), u32::MAX as usize)?;
    GraphLimits::check_count("buffers", buffers.len(), u32::MAX as usize)?;
    for pass in passes {
        GraphLimits::check_count(
            "accesses",
            pass.texture_accesses
                .len()
                .checked_add(pass.buffer_accesses.len())
                .ok_or(CompileError::Capacity {
                    kind: "accesses",
                    count: usize::MAX,
                    limit: u32::MAX as usize,
                })?,
            u32::MAX as usize,
        )?;
        // Validation precedes culling: a malformed index must not reach an
        // unchecked table lookup even when an access ultimately gets culled.
        for acc in &pass.texture_accesses {
            if acc.resource.get() as usize >= textures.len() {
                return Err(CompileError::InvalidHandle {
                    kind: "texture",
                    index: acc.resource.get(),
                    version: acc.input_version,
                });
            }
        }
        for acc in &pass.buffer_accesses {
            if acc.resource.get() as usize >= buffers.len() {
                return Err(CompileError::InvalidHandle {
                    kind: "buffer",
                    index: acc.resource.get(),
                    version: acc.input_version,
                });
            }
        }
    }
    for (pass_index, pass) in passes.iter().enumerate() {
        for &dependency in &pass.dependencies {
            if dependency >= passes.len() || dependency == pass_index {
                return Err(CompileError::InvalidDependency {
                    pass: crate::frame_graph::PassId(pass_index as u32),
                    dependency: crate::frame_graph::PassId(dependency as u32),
                });
            }
        }
    }
    let alive = cull::cull(passes, textures, buffers);
    let accesses = schedule::normalize(passes, textures.len());
    let solved = schedule::schedule(passes, &alive, &accesses)?;
    let queues: Vec<_> = passes.iter().map(schedule::queue_class).collect();
    let queue_batches = schedule::queue_batches(&queues, &solved.order, &solved.dependencies);
    let logical_barriers = solved
        .hazards
        .iter()
        .map(|h| {
            let resource = accesses[h.destination][h.after].resource;
            LogicalBarrier {
                resource: if resource < textures.len() {
                    LogicalResource::Texture(crate::ResourceIndex(resource as u32))
                } else {
                    LogicalResource::Buffer(crate::ResourceIndex(
                        (resource - textures.len()) as u32,
                    ))
                },
                source: h.source,
                destination: h.destination,
                before: h.before,
                after: h.after,
                queue_transfer: queues[h.source] != queues[h.destination],
            }
        })
        .collect();
    let order = solved.order;
    let mut position_by_pass = vec![None; passes.len()];
    for (position, &pass) in order.iter().enumerate() {
        position_by_pass[pass] = Some(position);
    }

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

    let lifetimes = lifetime::analyze(&accesses, &order, textures.len(), buffers.len());
    let alias = alias::plan_alias(textures, buffers, &lifetimes, swapchain)?;
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

    // Reuse the normalized authority records verbatim: consumers must not
    // reconstruct versions by counting writes in the final scheduled order.
    let compiled_accesses = accesses
        .iter()
        .map(|list| {
            list.iter()
                .map(|access| CompiledAccess {
                    resource: if access.resource < textures.len() {
                        LogicalResource::Texture(crate::ResourceIndex(access.resource as u32))
                    } else {
                        LogicalResource::Buffer(crate::ResourceIndex(
                            (access.resource - textures.len()) as u32,
                        ))
                    },
                    input_version: access.input,
                    output_version: access.output,
                })
                .collect()
        })
        .collect();

    Ok(ExecutionPlan {
        declared_pass_count: passes.len(),
        order,
        alive,
        position_by_pass,
        accesses: compiled_accesses,
        logical_barriers,
        queue_batches,
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
                    let load = if writes && acc.color_load.is_some() {
                        acc.color_load.expect("explicit load checked above")
                    } else if let Some(clear) = acc.color_clear.filter(|_| writes) {
                        LoadOp::Clear(clear)
                    } else if writes && acc.input_version == 0 {
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
                    // A depth attachment that has never been written is cleared.
                    //
                    // The value is the *far* plane of the engine's depth
                    // convention — reversed-Z with the near plane at 1 and the far
                    // plane at 0, tested with `GreaterEqual` (design: "所有透视投影
                    // reversed-Z；深度约定全仓一致（近 1 远 0，测试 GREATER，清 0）").
                    // Clearing to 1.0 (the GL/D3D far plane) would clear to *near*
                    // under reversed-Z, so nothing would pass the test.
                    let depth_load = if writes && acc.depth_load.is_some() {
                        acc.depth_load.expect("explicit depth load checked above")
                    } else if writes && acc.input_version == 0 {
                        DepthLoadOp::Clear(0.0)
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
