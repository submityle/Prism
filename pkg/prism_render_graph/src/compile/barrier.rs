//! Automatic barrier planning via the driver state tracker.
//!
//! Authors declare *uses*, never barriers. This module replays the scheduled
//! frame through the driver's [`StateTracker`], which knows the correct
//! synchronization for every state transition, and collects the minimal set of
//! barriers each pass needs. Imported resources start in the state their
//! previous owner left them (seeding the tracker) and are transitioned back to
//! their declared exit state at frame end, so a swapchain image ends in
//! `Present` and a history buffer ends ready for next frame.
//!
//! Barriers are keyed by [`ResourceIndex`], not realized ids, because the
//! solver runs before realization. Explicit-barrier backends map each index to
//! its realized id; auto-synced backends treat the result as a correctness
//! oracle to validate against.
//!
//! [`StateTracker`]: prism_render_driver::StateTracker

use alloc::vec::Vec;

use prism_render_driver::{
    BufferState, PipelineStages, StateTracker, SubresourceRange, TextureDimension, TextureState,
};

use crate::handle::ResourceIndex;
use crate::pass::{PassKind, PassNode};
use crate::plan::GraphBarrier;
use crate::resource::{BufferResource, TextureResource};

/// The barrier schedule for a frame.
pub(crate) struct BarrierPlan {
    /// Barriers to run before each scheduled pass, parallel to the schedule.
    pub per_pass: Vec<Vec<GraphBarrier>>,
    /// Barriers to run after the last pass to satisfy imported exit states.
    pub final_barriers: Vec<GraphBarrier>,
}

/// Replays the schedule to derive per-pass and frame-exit barriers.
pub(crate) fn plan_barriers(
    passes: &[PassNode],
    textures: &[TextureResource],
    buffers: &[BufferResource],
    order: &[usize],
    used_textures: &[bool],
    used_buffers: &[bool],
    swapchain: prism_render_driver::Extent3d,
) -> BarrierPlan {
    let mut tracker: StateTracker<ResourceIndex, ResourceIndex> = StateTracker::new();

    for (i, tr) in textures.iter().enumerate() {
        if !used_textures[i] {
            continue;
        }
        let (mips, layers, initial) = if let Some(imported) = tr.imported {
            (
                imported.mip_levels.max(1),
                imported.array_layers.max(1),
                imported.entry_state,
            )
        } else {
            let extent = tr.desc.size.resolve(swapchain);
            let layers = if matches!(tr.desc.dimension, TextureDimension::D3) {
                1
            } else {
                extent.depth_or_array_layers.max(1)
            };
            (
                tr.desc.mip_level_count.max(1),
                layers,
                TextureState::undefined(),
            )
        };
        tracker.register_texture(ResourceIndex(i as u32), mips, layers, initial);
    }
    for (i, br) in buffers.iter().enumerate() {
        if !used_buffers[i] {
            continue;
        }
        let initial = br
            .imported
            .map_or_else(BufferState::initial, |imported| imported.entry_state);
        tracker.register_buffer(ResourceIndex(i as u32), initial);
    }

    let mut per_pass: Vec<Vec<GraphBarrier>> = Vec::with_capacity(order.len());
    for &pidx in order {
        let pass = &passes[pidx];
        let stages = pass_stages(pass.kind);
        let mut batch: Vec<GraphBarrier> = Vec::new();
        for acc in &pass.texture_accesses {
            let next = acc.usage.state(stages);
            batch.append(&mut tracker.use_texture(acc.resource, acc.range, next));
        }
        for acc in &pass.buffer_accesses {
            let next = acc.usage.state(stages);
            if let Some(barrier) = tracker.use_buffer(acc.resource, next) {
                batch.push(barrier);
            }
        }
        per_pass.push(batch);
    }

    let mut final_barriers: Vec<GraphBarrier> = Vec::new();
    for (i, tr) in textures.iter().enumerate() {
        if !used_textures[i] {
            continue;
        }
        if let Some(imported) = tr.imported {
            final_barriers.append(&mut tracker.use_texture(
                ResourceIndex(i as u32),
                SubresourceRange::all(),
                imported.exit_state,
            ));
        }
    }

    BarrierPlan {
        per_pass,
        final_barriers,
    }
}

/// The pipeline stage shader accesses adopt for a given pass kind. Attachment,
/// transfer, and present uses override this with their own fixed stage inside
/// [`TextureUse::state`](crate::TextureUse::state).
fn pass_stages(kind: PassKind) -> PipelineStages {
    match kind {
        PassKind::Raster => PipelineStages::VERTEX_SHADER.union(PipelineStages::FRAGMENT_SHADER),
        PassKind::Compute => PipelineStages::COMPUTE_SHADER,
        PassKind::Transfer => PipelineStages::TRANSFER,
        PassKind::Present => PipelineStages::PRESENT,
    }
}
