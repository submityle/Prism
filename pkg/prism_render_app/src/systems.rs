//! The systems that bridge the main world and the render sub-app.

use std::time::Instant;

use prism_ecs::prelude::{Res, ResMut, SystemParam};
use prism_ecs::world::World;
use prism_render_architecture::lighting::contract::{GiCandidateSource, JournalSnapshot};
use prism_render_driver::{FrameSync, Instance};

use crate::frame;
use crate::intent::RenderIntent;
use crate::label::Render;
use crate::lighting::LightingPlanState;
use crate::lighting_gpu::{
    lighting_submission_domain, LightingCompositeMaterialBuffers, LightingGpuState,
};
use crate::material::{MaterialParameterBuffers, NativeExecutionTableBuffers};
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FrameExecutionTrace {
    pub frame_index: u64,
    pub journal_snapshot: JournalSnapshot,
    pub visibility_work_items: u64,
    pub gi_candidates: u64,
    pub graph: Option<frame::FrameGraphStages>,
    pub surface_graph: Option<frame::SurfaceViewGraphStages>,
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
            surface_graph: None,
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
    fabric_frame: Option<Res<'w, crate::ExtractedRenderFabricFrame>>,
    particle: Option<Res<'w, crate::particle::ParticleGpuState>>,
    particle_status: Res<'w, crate::particle::ParticleUploadStatus>,
    ember: Res<'w, crate::particle::EmberGpuAssetRegistry>,
    ember_external: Res<'w, crate::particle::ExtractedEmberExternalGpuResources>,
    ember_status: Res<'w, crate::particle::EmberPrepareStatus>,
    frame_texture_imports: Res<'w, crate::frame::FrameGraphTextureImports>,
    material_buffers: Res<'w, MaterialParameterBuffers>,
    execution_buffers: Res<'w, NativeExecutionTableBuffers>,
}

fn composite_material_buffers(
    materials: &MaterialParameterBuffers,
    executions: &NativeExecutionTableBuffers,
) -> Option<LightingCompositeMaterialBuffers> {
    let snapshot = executions.snapshot()?;
    Some(LightingCompositeMaterialBuffers {
        headers: materials.headers(),
        headers_size_bytes: u64::from(materials.header_capacity())
            * core::mem::size_of::<prism_render_material::GpuMaterialHeader>() as u64,
        parameters: materials.parameters(),
        parameters_size_bytes: u64::from(materials.parameter_capacity_words()) * 4,
        execution_plans: Some(snapshot.plan_buffer()),
        execution_plans_size_bytes: snapshot.plan_rows().len() as u64
            * (prism_render_material::EXECUTION_PLAN_ROW_WORDS * 4) as u64,
        execution_programs: Some(snapshot.program_buffer()),
        execution_programs_size_bytes: snapshot.program_rows().len() as u64 * 20,
    })
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
        self.surface_graph = None;
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
    let frame_start = Instant::now();
    world.run_schedule(Render);
    let frame_index = world
        .get_resource::<RenderRecord>()
        .map_or(0, |record| record.frame_index);
    if let Some(stats) = world.get_resource_mut::<RenderFrameStats>() {
        stats.record_cpu_frame(frame_index, frame_start.elapsed());
    }
}

/// Drops the frame's queued draws once they have been recorded.
///
/// Runs in [`RenderSystems::Cleanup`], after the record stage: the queue is
/// per-frame state (the queue stage fills it from this frame's visible set), so
/// leaving it populated would make the next frame draw the previous frame's
/// items again — and forever.
pub(crate) fn clear_queues(
    mut queues: ResMut<crate::RenderQueues>,
    mut cameras: ResMut<crate::SurfaceViewQueues>,
) {
    queues.clear();
    cameras.clear();
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
    (queues, cameras, mut camera_targets): (
        Res<crate::RenderQueues>,
        Res<crate::SurfaceViewQueues>,
        ResMut<crate::surface_view_targets::SurfaceViewTargets<I::Device>>,
    ),
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
    let fabric_frame = execution.fabric_frame;
    let particle = execution.particle;
    let particle_status = execution.particle_status;
    let ember = execution.ember;
    let ember_external = execution.ember_external;
    let ember_status = execution.ember_status;
    let frame_texture_imports = execution.frame_texture_imports;
    stats
        .execution_trace
        .begin(&record, queues.len() as u64, &inputs);
    stats.execution_trace.assert_input_parity(&record, &inputs);
    stats.surface_view_error = None;
    stats.surface_view_output_rejections.clear();
    stats.surface_view_target_rejections.clear();
    let camera_error = if let Some(error) = cameras.error() {
        Some(crate::stats::SurfaceViewExecutionError::QueuePreparation(
            error,
        ))
    } else if !cameras.is_ready()
        || cameras.frame_index != record.frame_index
        || cameras.device_generation != runtime.runtime().device_generation()
    {
        Some(crate::stats::SurfaceViewExecutionError::StaleQueues)
    } else {
        None
    };
    if let Some(error) = camera_error {
        stats.surface_view_error = Some(error);
        stats.frames_failed += 1;
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    stats.fabric_frame_error =
        crate::fabric_frame::admit_render_fabric_frame(fabric_frame.as_deref(), record.frame_index)
            .err();
    stats.lighting_input_error = None;
    stats.lighting_upload_error = None;
    stats.ember_graphs_updated = stats
        .ember_graphs_updated
        .saturating_add(u64::from(ember_status.updated_graphs));
    stats.ember_graphs_preserved = stats
        .ember_graphs_preserved
        .saturating_add(u64::from(ember_status.preserved_graphs));
    stats.ember_initial_failures = stats
        .ember_initial_failures
        .saturating_add(u64::from(ember_status.initial_failures));
    stats.ember_prepare_error = ember_status.error_kind;
    stats.ember_prepare_outcome = ember_status.outcome;
    if stats.fabric_frame_error.is_some() {
        stats.frames_failed = stats.frames_failed.saturating_add(1);
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    if particle_status.error.is_some() {
        stats.frames_failed = stats.frames_failed.saturating_add(1);
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    if ember_status.fatal {
        stats.frames_failed = stats.frames_failed.saturating_add(1);
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    // Emitter-local adaptation rejections remain observable on `inputs` but
    // do not abort the frame; only a snapshot-wide extraction error is fatal.
    stats.lighting_input_error = inputs.light_scene_error;
    if inputs.light_scene_error.is_some() {
        stats.frames_failed = stats.frames_failed.saturating_add(1);
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    let runtime = runtime.runtime();
    use crate::surface_view_targets::{SurfaceViewOutput, SurfaceViewTargetKey};
    let outputs: Vec<_> = requests.render_outputs().map_or_else(
        || vec![(SurfaceViewOutput::Offscreen(0), record.viewport)],
        |outputs| {
            outputs
                .into_iter()
                .map(|(id, viewport)| (SurfaceViewOutput::Window(id), viewport))
                .collect()
        },
    );
    let mut desired = Vec::new();
    let primary = requests.primary_window();
    for (output, viewport) in outputs {
        if viewport.is_empty() {
            continue;
        }
        for camera in cameras.iter() {
            if !camera_target_matches_output(camera.surface.target, output, primary) {
                continue;
            }
            if surface_view_fits(&camera.surface, viewport) {
                let extent = crate::surface_view_targets::surface_view_render_extent(
                    camera.surface.viewport,
                    camera.surface.resolution_scale,
                );
                desired.push(SurfaceViewTargetKey::new(camera.camera(), output, extent));
            } else {
                stats.surface_view_output_rejections.push(
                    crate::stats::SurfaceViewOutputRejection {
                        camera: camera.camera(),
                        output,
                        viewport: camera.surface.viewport,
                        output_viewport: viewport,
                    },
                );
            }
        }
    }
    for camera in cameras.iter() {
        let routed = match camera.surface.target {
            prism_camera::RenderTarget::PrimaryWindow => {
                primary.is_some() || requests.render_outputs().is_none()
            }
            prism_camera::RenderTarget::Window(id) => requests
                .render_outputs()
                .is_some_and(|outputs| outputs.iter().any(|(window, _)| window.0 == id)),
            prism_camera::RenderTarget::Texture(id) => {
                requests.render_outputs().is_none() && id == 0
            }
        };
        if !routed {
            stats
                .surface_view_target_rejections
                .push(crate::stats::SurfaceViewTargetRejection {
                    camera: camera.camera(),
                    target: camera.surface.target,
                });
        }
    }
    if let Err(error) = camera_targets
        .collect_retired(|target, device, generation| {
            lighting_gpu.evict_target(target, device, generation)
        })
        .and_then(|_| {
            camera_targets.reconcile(runtime, &desired, |target, device, generation| {
                lighting_gpu.evict_target(target, device, generation)
            })
        })
    {
        stats.surface_view_error = Some(crate::stats::SurfaceViewExecutionError::Target(error));
        stats.frames_failed += 1;
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    if let Err(error) = lighting_gpu.prepare(&plan.plan, &inputs, record.frame_index, runtime) {
        stats.lighting_upload_error = Some(error);
        stats.frames_failed = stats.frames_failed.saturating_add(1);
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    stats.lighting_upload_error = None;
    let lighting_plan_pass = lighting_gpu.pass(&plan.plan);
    let trace_device: std::sync::Arc<dyn prism_render_driver::RenderDevice + Send + Sync> =
        runtime.device_handle();
    let trace = trace_gpu.prepare(
        (plan.plan.candidate_source == GiCandidateSource::SoftwareBvhGpu)
            .then_some(inputs.trace_scene.as_ref())
            .flatten(),
        &inputs.surfaces,
        record.frame_index,
        runtime.device_generation(),
        trace_device,
        runtime.queue(),
        lighting_plan_pass
            .filter(|pass| pass.run_gi)
            .map(crate::lighting_gpu::LightingGpuPass::gi_candidate_target),
    );
    if plan.plan.candidate_source == GiCandidateSource::SoftwareBvhGpu
        && lighting_plan_pass.is_some_and(|pass| pass.run_gi)
        && trace.is_none()
    {
        stats.frames_failed += 1;
        stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
        return;
    }
    let submitted_before = stats.frames_submitted;
    let skipped_before = stats.frames_skipped;
    let failed_before = stats.frames_failed;
    let window_receipt = windows.drive_with_surface_views(
        runtime,
        &requests,
        &overrides,
        &queues,
        &functions,
        Some((&*lighting_gpu, &plan.plan)),
        trace,
        particle.as_ref().map(|value| &**value),
        crate::presentation::SurfaceViewDriveContext {
            cameras: &cameras,
            targets: &mut camera_targets,
            frame_index: record.frame_index,
            texture_imports: &frame_texture_imports,
            ember: Some(&ember),
            ember_external: &ember_external,
        },
        &mut stats,
    );
    if window_receipt.handled {
        if let Some(submission) = window_receipt.first_submission.and_then(|submission| {
            lighting_submission_domain(runtime.device_generation()).submission(submission)
        }) {
            lighting_gpu.acknowledge_light_upload(submission);
        }
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
        lighting_gpu.evict_target(
            &offscreen.target,
            &runtime.device_handle(),
            runtime.device_generation(),
        );
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
    let lighting_pass = lighting_gpu.pass_for_target(&plan.plan, &offscreen.target, runtime);
    let upscaler = cameras
        .iter()
        .any(|camera| matches!(camera.surface.target, prism_camera::RenderTarget::PrimaryWindow | prism_camera::RenderTarget::Texture(0)) )
        .then(|| camera_targets.ensure_upscaler(runtime))
        .flatten();
    let view_inputs: Vec<_> = cameras
        .iter()
        .filter_map(|camera| {
            let target_matches = match camera.surface.target {
                prism_camera::RenderTarget::PrimaryWindow => true,
                prism_camera::RenderTarget::Window(_) => false,
                prism_camera::RenderTarget::Texture(id) => id == 0,
            };
            if !target_matches {
                return None;
            }
            let extent = crate::surface_view_targets::surface_view_render_extent(
                camera.surface.viewport,
                camera.surface.resolution_scale,
            );
            let (upscale_pipeline, upscale_bindings) = upscaler?;
            let key =
                SurfaceViewTargetKey::new(camera.camera(), SurfaceViewOutput::Offscreen(0), extent);
            let target = camera_targets.get(runtime, key)?;
            Some(frame::SurfaceViewGraphInput {
                camera: camera.camera(),
                order: camera.surface.order,
                clear: camera.surface.clear,
                viewport: camera.surface.viewport,
                target,
                queues: &camera.queues,
                lighting: lighting_gpu.pass_for_target(&plan.plan, target, runtime),
                upscale_pipeline,
                upscale_bindings,
            })
        })
        .collect();
    let prepared = match frame::prepare_surface_view_graph(
        &offscreen.target,
        &queues,
        &functions,
        lighting_pass,
        trace,
        &view_inputs,
        &frame_texture_imports,
    ) {
        Ok(prepared) => prepared,
        Err(_) => {
            stats.surface_view_error =
                Some(crate::stats::SurfaceViewExecutionError::GraphAdmission);
            stats.frames_failed += 1;
            stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
            return;
        }
    };
    let graph = frame::finalize_surface_view_graph(
        prepared,
        particle.as_ref().map(|value| &**value),
        Some(&ember),
        ember_external.graph_imports(),
        record.frame_index,
    )
    .and_then(|(mut graph, source)| {
        let output = offscreen
            .display
            .append(&mut graph, runtime.queue(), source)?;
        let mut stages = graph
            .blackboard()
            .get::<frame::SurfaceViewGraphStages>()
            .expect("surface graph publishes aggregate stages")
            .clone();
        stages.present_source = output;
        stages.assert_contract();
        graph.blackboard_mut().set(stages);
        Ok(graph)
    });
    let graph = match graph {
        Ok(graph) => graph,
        Err(_) => {
            stats.frames_failed += 1;
            stats.execution_trace.outcome = FrameExecutionOutcome::Failed;
            return;
        }
    };
    let single_stages = *graph
        .blackboard()
        .get::<frame::FrameGraphStages>()
        .expect("the host frame graph must publish its stage receipt");
    let stages = graph
        .blackboard()
        .get::<frame::SurfaceViewGraphStages>()
        .expect("surface graph publishes aggregate stages")
        .clone();
    stages.assert_contract();
    assert_eq!(
        stages.gi_recipes,
        u32::from(lighting_pass.is_some_and(|pass| pass.run_gi)),
        "GI graph admission must match the selected lighting pass"
    );
    stats.execution_trace.graph = Some(single_stages);
    stats.execution_trace.surface_graph = Some(stages.clone());
    let passes = graph.pass_count() as u64;
    let draws = queues.len() as u64
        + view_inputs
            .iter()
            .map(|view| view.queues.len() as u64)
            .sum::<u64>();
    match runtime.execute_tracked(graph) {
        Ok(submission) => {
            seal_surface_view_submission_bookkeeping(runtime, &mut camera_targets, &mut stats);
            if let Some(submission) =
                lighting_submission_domain(runtime.device_generation()).submission(submission)
            {
                lighting_gpu.acknowledge_light_upload(submission);
            }
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
            stats.lighting_gpu_passes_recorded += u64::from(stages.lighting_dispatches);
            stats.trace_gpu_dispatches_recorded += u64::from(stages.trace_dispatches);
            stats.ember_compute_passes_recorded += u64::from(stages.ember_compute_passes);
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

fn camera_target_matches_output(
    target: prism_camera::RenderTarget,
    output: crate::surface_view_targets::SurfaceViewOutput,
    primary: Option<crate::WindowId>,
) -> bool {
    use crate::surface_view_targets::SurfaceViewOutput;

    match target {
        prism_camera::RenderTarget::PrimaryWindow => match output {
            SurfaceViewOutput::Window(id) => Some(id) == primary,
            SurfaceViewOutput::Offscreen(_) => primary.is_none(),
        },
        prism_camera::RenderTarget::Window(id) => {
            output == SurfaceViewOutput::Window(crate::WindowId(id))
        }
        prism_camera::RenderTarget::Texture(id) => output == SurfaceViewOutput::Offscreen(id),
    }
}

/// Output-local rejection only; never clamp or reinterpret a camera's matrix/rect.
pub(crate) fn surface_view_fits(view: &crate::SurfaceView, output: crate::RenderViewport) -> bool {
    let [x, y, width, height] = view.viewport;
    width != 0
        && height != 0
        && x.checked_add(width).is_some_and(|end| end <= output.width)
        && y.checked_add(height)
            .is_some_and(|end| end <= output.height)
}

/// Record camera-target completion after a successful offscreen submission.
///
/// A bookkeeping failure cannot undo the submission, so it is a diagnostic
/// rather than a panic or a failed-frame rewrite. The next reconciliation can
/// retry normal ownership handling.
fn seal_surface_view_submission_bookkeeping<I>(
    runtime: &prism_render_runtime::RenderRuntime<I>,
    targets: &mut crate::surface_view_targets::SurfaceViewTargets<I::Device>,
    stats: &mut RenderFrameStats,
) where
    I: Instance + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: FrameSync,
{
    if let Err(error) = targets.seal_submissions(runtime) {
        stats.surface_view_error = Some(crate::stats::SurfaceViewExecutionError::Target(error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prism_ecs::prelude::Entity;
    use prism_render_driver::Extent3d;
    use prism_render_driver_null::NullInstance;

    #[test]
    fn camera_targets_route_only_to_their_stable_output_identity() {
        use crate::surface_view_targets::SurfaceViewOutput;
        use prism_camera::RenderTarget as CameraTarget;

        let primary = Some(crate::WindowId(7));
        assert!(camera_target_matches_output(
            CameraTarget::PrimaryWindow,
            SurfaceViewOutput::Window(crate::WindowId(7)),
            primary,
        ));
        assert!(!camera_target_matches_output(
            CameraTarget::PrimaryWindow,
            SurfaceViewOutput::Window(crate::WindowId(3)),
            primary,
        ));
        assert!(camera_target_matches_output(
            CameraTarget::Window(3),
            SurfaceViewOutput::Window(crate::WindowId(3)),
            primary,
        ));
        assert!(camera_target_matches_output(
            CameraTarget::Texture(42),
            SurfaceViewOutput::Offscreen(42),
            primary,
        ));
        assert!(!camera_target_matches_output(
            CameraTarget::Texture(42),
            SurfaceViewOutput::Offscreen(0),
            primary,
        ));
    }

    #[test]
    fn offscreen_surface_view_seal_identity_mismatch_is_diagnostic_not_panic() {
        let extent = Extent3d::new_2d(16, 16);
        let runtime = prism_render_runtime::RenderRuntime::<NullInstance>::headless(
            NullInstance::new(),
            extent,
        )
        .expect("null runtime");
        let other = prism_render_runtime::RenderRuntime::<NullInstance>::headless(
            NullInstance::new(),
            extent,
        )
        .expect("independent null runtime");
        let view = crate::surface_view_targets::SurfaceViewTargetKey::new(
            Entity::from_bits((1_u64 << 32) | 1).expect("valid entity"),
            crate::surface_view_targets::SurfaceViewOutput::Offscreen(1),
            crate::RenderViewport::from_extent(extent),
        );
        let mut targets = crate::surface_view_targets::SurfaceViewTargets::default();
        targets
            .reconcile(&other, &[view], |_, _, _| true)
            .expect("target allocation");
        let mut stats = RenderFrameStats::default();

        seal_surface_view_submission_bookkeeping(&runtime, &mut targets, &mut stats);

        assert_eq!(
            stats.surface_view_error,
            Some(crate::stats::SurfaceViewExecutionError::Target(
                crate::surface_view_targets::SurfaceViewTargetError::DeviceMismatch,
            )),
        );
        assert_eq!(stats.frames_failed, 0, "submission state is not rewritten");
    }
}
