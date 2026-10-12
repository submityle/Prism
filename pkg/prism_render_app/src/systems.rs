//! The systems that bridge the main world and the render sub-app.

use prism_ecs::prelude::{Res, ResMut};
use prism_ecs::world::World;
use prism_render_driver::{FrameSync, Instance};

use crate::camera::RenderCamera;
use crate::frame;
use crate::label::Render;
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::stats::RenderFrameStats;
use crate::target::RenderTarget;

/// The one-way extract step: publish the main world's render intent.
///
/// Runs once per frame, before the render sub-app updates, with
/// `(main_world, render_world)`. It reads the main world's [`RenderCamera`] and
/// writes a [`RenderRecord`] into the render world, advancing the record's
/// `frame_index` so the feed is observable.
///
/// The contract is one-way (design §21): it reads the main world and writes only
/// the render world, and the main world's authoritative state is never mutated
/// here. Only the small [`RenderRecord`] is touched — the render world's GPU
/// resources (the runtime and the target) are left alone, so the device and its
/// textures survive every extract.
pub(crate) fn extract_render_record(main: &mut World, render: &mut World) {
    let viewport = main
        .get_resource::<RenderCamera>()
        .copied()
        .unwrap_or_default()
        .viewport;
    let frame_index = render
        .get_resource::<RenderRecord>()
        .map_or(1, |record| record.frame_index.saturating_add(1));
    render.insert_resource(RenderRecord::new(viewport, frame_index));
}

/// Runs the render sub-app's [`Render`] schedule as part of its frame.
///
/// The sub-app's per-frame loop drives the fixed core phases; a dedicated render
/// schedule would otherwise never run, so this exclusive system — registered in
/// the render sub-app's `Last` phase — runs it explicitly. It is the render
/// world's analogue of the main loop calling into a render schedule, and it is
/// what makes a single `App::update` produce exactly one rendered frame.
pub(crate) fn drive_render_schedule(world: &mut World) {
    world.run_schedule(Render);
}

/// The render system: record and submit one frame through the RHI runtime.
///
/// Registered in the [`Render`] schedule, it reads the extracted
/// [`RenderRecord`], resizes the frame target if the main world asked for a new
/// size, records a frame graph, and submits it under the runtime's
/// frames-in-flight bound. Every outcome is published to [`RenderFrameStats`] so
/// the work is observable from outside the render world.
///
/// An empty viewport (a minimized or zero-sized frame) is a successful no-op:
/// nothing is recorded and the frame is counted as skipped.
pub(crate) fn execute_frame<I>(
    record: Res<RenderRecord>,
    runtime: Res<RenderRuntimeResource<I>>,
    mut target: ResMut<RenderTarget>,
    mut stats: ResMut<RenderFrameStats>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let viewport = record.viewport;
    stats.last_viewport = viewport;
    if viewport.is_empty() {
        stats.frames_skipped += 1;
        return;
    }

    let runtime = runtime.runtime();
    // The main world may have resized the frame; reallocate the target to match.
    // Drain in-flight frames first, since an outstanding frame may still read the
    // texture being replaced.
    if target.viewport() != viewport {
        let _ = runtime.wait_idle();
        target.resize(runtime.device(), viewport);
    }

    let graph = frame::clear_target_graph(&target);
    let passes = graph.pass_count() as u64;
    match runtime.execute(graph) {
        Ok(()) => {
            stats.frames_submitted += 1;
            stats.passes_declared += passes;
        }
        Err(_) => {
            // A compile failure is the only error `execute` can return; the
            // runtime issues no GPU work in that case, so we count the frame and
            // leave it at that. The minimal clear graph cannot cycle today, so
            // this is defensive rather than expected.
            stats.frames_failed += 1;
        }
    }
}
