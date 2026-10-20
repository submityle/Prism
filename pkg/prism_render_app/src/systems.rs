//! The systems that bridge the main world and the render sub-app.

use prism_ecs::prelude::{Res, ResMut, SystemParam};
use prism_ecs::world::World;
use prism_render_architecture::lighting::contract::JournalSnapshot;
use prism_render_driver::{FrameSync, Instance};

use crate::frame;
use crate::intent::RenderIntent;
use crate::label::Render;
use crate::lighting::LightingPlanState;
use crate::lighting_gpu::LightingGpuState;
use crate::record::RenderRecord;
use crate::runtime_resource::RenderRuntimeResource;
use crate::startup::ensure_render_startup;
use crate::stats::RenderFrameStats;
use crate::target::RenderTarget;

/// The host-side receipt for the frame pipeline.
///
/// This is diagnostic state, not a second execution path. It binds the
/// extracted scene record and lighting journal to the same frame that the
/// graph or presentation driver consumes, making a stale cross-frame input
/// observable without adding a control switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameExecutionTrace {
    pub frame_index: u64,
    pub journal_snapshot: JournalSnapshot,
    pub visibility_work_items: u64,
    pub gi_candidates: u64,
    pub graph: Option<frame::FrameGraphStages>,
    pub outcome: FrameExecutionOutcome,
}

impl Default for FrameExecutionTrace {
    fn default() -> Self {
        Self {
            frame_index: 0,
            journal_snapshot: JournalSnapshot::default(),
            visibility_work_items: 0,
            gi_candidates: 0,
            graph: None,
            outcome: FrameExecutionOutcome::NotRun,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameExecutionOutcome {
    NotRun,
    Skipped,
    Submitted,
    Presented,
    Failed,
    Mixed,
}

#[derive(SystemParam)]
pub(crate) struct OffscreenFrameTargets<'w> {
    target: ResMut<'w, RenderTarget>,
    display: ResMut<'w, crate::display_transform_gpu::OffscreenDisplayTransform>,
    output: ResMut<'w, crate::display_transform_gpu::OffscreenFrameOutput>,
}

#[derive(SystemParam)]
pub(crate) struct FrameExecutionResources<'w> {
    trace_gpu: ResMut<'w, crate::lighting_trace_gpu::LightingTraceState>,
    stats: ResMut<'w, RenderFrameStats>,
}

impl FrameExecutionTrace {
    fn begin(
        &mut self,
        record: &RenderRecord,
        visibility_work_items: u64,
        inputs: &crate::ExtractedLightingInputs,
    ) {
        self.frame_index = record.frame_index;
        self.journal_snapshot = inputs.journal_snapshot;
        self.visibility_work_items = visibility_work_items;
        self.gi_candidates = inputs
            .gi_candidates
            .iter()
            .map(|batch| batch.len() as u64)
            .sum();
        self.graph = None;
        self.outcome = FrameExecutionOutcome::NotRun;
    }

    fn assert_input_parity(&self, record: &RenderRecord, inputs: &crate::ExtractedLightingInputs) {
        assert_eq!(
            self.frame_index, record.frame_index,
            "scene and lighting inputs must belong to the extracted frame"
        );
        assert_eq!(
            self.journal_snapshot, inputs.journal_snapshot,
            "lighting journal must be the snapshot consumed by this frame"
        );
    }
}

fn classify_window_outcome(submitted: u64, skipped: u64, failed: u64) -> FrameExecutionOutcome {
    match (submitted != 0, skipped != 0, failed != 0) {
        (true, false, false) => FrameExecutionOutcome::Presented,
        (false, true, false) => FrameExecutionOutcome::Skipped,
        (false, false, true) => FrameExecutionOutcome::Failed,
        (false, false, false) => FrameExecutionOutcome::NotRun,
        _ => FrameExecutionOutcome::Mixed,
    }
}

/// The one-way extract step: publish the main world's render intent.
///
/// Runs once per frame, before the render sub-app updates, with
/// `(main_world, render_world)`. It reads the main world's [`RenderIntent`] and
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
        .get_resource::<RenderIntent>()
        .copied()
        .unwrap_or_default()
        .viewport;
    let frame_index = render
        .get_resource::<RenderRecord>()
        .map_or(1, |record| record.frame_index.saturating_add(1));
    render.insert_resource(RenderRecord::new(viewport, frame_index));
}

/// Runs the render sub-app's GPU startup, then its [`Render`] schedule, as part
/// of its frame.
///
/// The sub-app's per-frame loop drives the fixed core phases; a dedicated render
/// schedule would otherwise never run, so this exclusive system — registered in
/// the render sub-app's `Last` phase — runs it explicitly. It is the render
/// world's analogue of the main loop calling into a render schedule, and it is
/// what makes a single `App::update` produce exactly one rendered frame.
///
/// [`RenderStartup`] runs first (once per device): a plugin registered its GPU
/// resources from any plugin, in any order, and they must exist before the first
/// frame reads them.
pub(crate) fn drive_render_schedule<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if crate::recovery::recover_before_render::<I>(world) {
        return;
    }
    ensure_render_startup(world);
    world.run_schedule(Render);
}

/// Drops the frame's queued draws once they have been recorded.
///
/// Runs in [`RenderSystems::Cleanup`], after the record stage: the queue is
/// per-frame state (the queue stage fills it from this frame's visible set), so
/// leaving it populated would make the next frame draw the previous frame's
/// items again — and forever.
pub(crate) fn clear_queues(mut queues: ResMut<crate::RenderQueues>) {
    queues.clear();
}

/// Drains completed asynchronous pipeline compiles before the queue stage
/// specializes this frame's draws.
///
/// The worker owns a shareable device handle; the cache owns the request state.
/// Polling here keeps the render schedule single-threaded from the ECS point of
/// view while moving the expensive `create_render_pipeline` call off it.
pub(crate) fn pump_pipeline_compiles<I>(
    mut specializer: ResMut<crate::MeshPipelineSpecializer>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    specializer.pump(runtime.runtime().device());
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
    queues: Res<crate::RenderQueues>,
    functions: Res<crate::DrawFunctions>,
    runtime: Res<RenderRuntimeResource<I>>,
    plan: Res<LightingPlanState>,
    inputs: Res<crate::ExtractedLightingInputs>,
    lighting_gpu: Res<LightingGpuState<I>>,
    mut offscreen: OffscreenFrameTargets,
    execution: FrameExecutionResources,
    requests: Res<crate::WindowRenderRequests<I>>,
    overrides: Res<crate::WindowRenderQueues>,
    mut windows: ResMut<crate::WindowRenderTargets<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let mut trace_gpu = execution.trace_gpu;
    let mut stats = execution.stats;
    stats
        .execution_trace
        .begin(&record, queues.len() as u64, &inputs);
    stats.execution_trace.assert_input_parity(&record, &inputs);
    let runtime = runtime.runtime();
    lighting_gpu.prepare(&plan.plan, &inputs, record.frame_index, runtime.queue());
    let lighting_pass = lighting_gpu.pass(&plan.plan);
    let trace_device: std::sync::Arc<dyn prism_render_driver::RenderDevice + Send + Sync> =
        runtime.device_handle();
    let trace = trace_gpu.prepare(
        plan.plan
            .recipes
            .contains(&prism_render_architecture::lighting::contract::RecipeId::WorldRestir)
            .then_some(inputs.trace_scene.as_ref())
            .flatten(),
        &inputs.surfaces,
        record.frame_index,
        runtime.device_generation(),
        trace_device,
        runtime.queue(),
        lighting_pass
            .filter(|pass| pass.run_gi)
            .map(crate::lighting_gpu::LightingGpuPass::gi_candidate_target),
    );
    let submitted_before = stats.frames_submitted;
    let skipped_before = stats.frames_skipped;
    let failed_before = stats.frames_failed;
    if windows.drive(
        runtime,
        &requests,
        &overrides,
        &queues,
        &functions,
        lighting_pass,
        trace,
        &mut stats,
    ) {
        stats.execution_trace.outcome = classify_window_outcome(
            stats.frames_submitted.saturating_sub(submitted_before),
            stats.frames_skipped.saturating_sub(skipped_before),
            stats.frames_failed.saturating_sub(failed_before),
        );
        return;
    }
    let viewport = record.viewport;
    stats.last_viewport = viewport;
    if viewport.is_empty() {
        stats.frames_skipped += 1;
        stats.execution_trace.outcome = FrameExecutionOutcome::Skipped;
        return;
    }

    // The main world may have resized the frame; reallocate the target to match.
    // Drain in-flight frames first, since an outstanding frame may still read the
    // texture being replaced.
    if offscreen.target.viewport() != viewport {
        if !runtime.wait_idle() {
            stats.frames_failed += 1;
            stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
            return;
        }
        offscreen.output.invalidate();
        offscreen.target.resize(runtime.device(), viewport);
    }

    if offscreen
        .output
        .latest()
        .is_some_and(|frame| frame.device_generation() != runtime.device_generation())
    {
        offscreen.output.invalidate();
    }

    let display_device: std::sync::Arc<dyn prism_render_driver::RenderDevice + Send + Sync> =
        runtime.device_handle();
    if offscreen
        .display
        .ensure(
            display_device,
            offscreen.target.view(),
            offscreen.target.viewport().extent(),
            runtime.device_generation(),
        )
        .is_err()
    {
        stats.frames_failed += 1;
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    let graph = match frame::target_graph_with_display_transform_and_trace(
        &offscreen.target,
        &queues,
        &functions,
        lighting_pass,
        trace,
        &offscreen.display,
        runtime.queue(),
    ) {
        Ok(graph) => graph,
        Err(_) => {
            stats.frames_failed += 1;
            stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
            return;
        }
    };
    let stages = *graph
        .blackboard()
        .get::<frame::FrameGraphStages>()
        .expect("the host frame graph must publish its stage receipt");
    stages.assert_contract();
    assert_eq!(
        stages.gi_recipes,
        u32::from(lighting_pass.is_some_and(|pass| pass.run_gi)),
        "GI graph admission must match the selected lighting pass"
    );
    stats.execution_trace.graph = Some(stages);
    let passes = graph.pass_count() as u64;
    let draws = queues.len() as u64;
    match runtime.execute_tracked(graph) {
        Ok(submission) => {
            offscreen
                .output
                .publish(crate::display_transform_gpu::OffscreenOutputFrame::new(
                    offscreen
                        .display
                        .output_texture()
                        .expect("the display transform was prepared before graph construction"),
                    viewport.extent(),
                    record.frame_index,
                    submission,
                    runtime.device_generation(),
                ));
            stats.execution_trace.outcome = FrameExecutionOutcome::Submitted;
            stats.frames_submitted += 1;
            stats.passes_declared += passes;
            stats.draws_recorded += draws;
            stats.lighting_gpu_passes_recorded += lighting_pass.map_or(0, |pass| pass.pass_count());
            stats.trace_gpu_dispatches_recorded += u64::from(trace.is_some());
        }
        Err(_) => {
            // A compile failure is the only error `execute` can return; the
            // runtime issues no GPU work in that case, so we count the frame and
            // leave it at that. The minimal clear graph cannot cycle today, so
            // this is defensive rather than expected.
            stats.frames_failed += 1;
            stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        }
    }
}
