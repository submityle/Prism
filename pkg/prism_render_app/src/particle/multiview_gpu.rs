//! Physical GPU acceptance of the production camera-local/composition path.
//! No native window/compositor is required: both outputs are real HDR targets.

use super::*;
use crate::draw::DrawFunctions;
use crate::frame::{
    finalize_surface_view_graph, prepare_surface_view_graph, FrameGraphTextureImports,
    SurfaceViewGraphInput, SurfaceViewGraphStages,
};
use crate::queue::RenderQueues;
use crate::surface_views::{extract_surface_views, ExtractedSurfaceViews, SurfaceView};
use crate::target::RenderTarget;
use crate::visibility::{PrismCamera, PrismFrustum};
use prism_app::{App, FixedPreUpdate};
use prism_math::{Frustum, Mat4, Plane, Vec3};
use prism_particle_app::{
    set_particle_fixed_timestep, ParticleMaterialStyle, ParticleRendererConfig,
    ParticleSimulationConfig, ParticleSimulationPlugin, ParticleSimulationSource,
};
use prism_render_driver::{
    BufferReadback, BufferReadbackError, BufferReadbackRange, CommandBuffer, CompletionToken,
    Extent3d, ImageDataLayout, Pass, RenderCommand, TextureWrite,
};
use prism_render_driver_wgpu::{
    require_hardware_gpu, HardwareDeviceError, WgpuDevice, WgpuInstance, WgpuQueue,
};
use prism_render_graph::RecordedGraph;
use prism_transform::GlobalTransform;
use std::sync::atomic::AtomicUsize;

#[path = "multiview_ember_gpu.rs"]
mod ember;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Submissions {
    simulations: usize,
    simulation_phases: BTreeMap<String, usize>,
    bridge_phases: BTreeMap<String, usize>,
    output_draws: Vec<Vec<BufferId>>,
}

fn assert_bridge_once(submitted: &Submissions) {
    // This fixture has one emitter route, so no hierarchical route-scan levels
    // are needed. Check every production bridge phase, not only clear/scatter.
    let expected = BTreeMap::from(
        [
            "particle-simulation-renderer-bridge-clear",
            "particle-simulation-renderer-bridge-mark-events",
            "particle-simulation-renderer-bridge-count-routes",
            "particle-simulation-renderer-bridge-prefix-routes",
            "particle-simulation-renderer-bridge-finalize-routes",
            "particle-simulation-renderer-bridge-scatter",
        ]
        .map(|phase| (String::from(phase), 1)),
    );
    assert_eq!(submitted.bridge_phases, expected);
}

/// Observe only commands actually handed to the real backend. Readback remains
/// delegated (including the async checkpoint API), never emulated by the test.
struct ObservedQueue {
    inner: WgpuQueue,
    submissions: Mutex<Submissions>,
    waits: AtomicUsize,
}

impl ObservedQueue {
    fn observe(&self, buffers: &[CommandBuffer]) {
        let mut receipt = self.submissions.lock().unwrap();
        for buffer in buffers {
            let simulation = buffer.label.as_deref() == Some("particle-gpu-simulation");
            receipt.simulations += usize::from(simulation);
            let mut draws = Vec::new();
            for pass in &buffer.passes {
                match pass {
                    Pass::Compute {
                        label: Some(label), ..
                    } => {
                        if simulation {
                            *receipt.simulation_phases.entry(label.clone()).or_default() += 1;
                        }
                        if label.starts_with("particle-simulation-renderer-bridge-") {
                            *receipt.bridge_phases.entry(label.clone()).or_default() += 1;
                        }
                    }
                    Pass::Render { commands, .. } => {
                        for command in commands {
                            if let RenderCommand::DrawIndirect { buffer, .. } = command {
                                draws.push(*buffer);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !draws.is_empty() {
                receipt.output_draws.push(draws);
            }
        }
    }

    fn reset(&self) {
        *self.submissions.lock().unwrap() = Submissions::default();
        self.waits.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> Submissions {
        self.submissions.lock().unwrap().clone()
    }
}

impl RenderQueue for ObservedQueue {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout) {
        self.inner.write_texture(destination, data, layout);
    }

    fn submit(&self, buffers: &[CommandBuffer]) {
        self.inner.submit(buffers);
        self.observe(buffers);
    }

    fn request_buffer_readback(
        &self,
        ranges: &[BufferReadbackRange],
        max_bytes: u64,
    ) -> Result<Box<dyn BufferReadback>, BufferReadbackError> {
        self.inner.request_buffer_readback(ranges, max_bytes)
    }
}

impl FrameSync for ObservedQueue {
    fn submit_tracked(&self, buffers: &[CommandBuffer]) -> CompletionToken {
        let token = <WgpuQueue as FrameSync>::submit_tracked(&self.inner, buffers);
        self.observe(buffers);
        token
    }

    fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
        self.waits.fetch_add(1, Ordering::Relaxed);
        <WgpuQueue as FrameSync>::wait_for(&self.inner, token)
    }

    fn wait_idle(&self) -> bool {
        self.waits.fetch_add(1, Ordering::Relaxed);
        self.inner.wait_idle()
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        self.inner.on_submitted_work_done(callback);
    }
}

fn hardware() -> Option<(WgpuDevice, ObservedQueue)> {
    let pair = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no hardware adapter"
            );
            eprintln!("SKIPPED physical multioutput particle regression: no hardware adapter");
            return None;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(!require_hardware_gpu(), "physical GPU required: {evidence}");
            eprintln!("SKIPPED physical multioutput particle regression: {evidence}");
            return None;
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("hardware device creation failed: {error}")
        }
    };
    Some((
        pair.0,
        ObservedQueue {
            inner: pair.1,
            submissions: Mutex::new(Submissions::default()),
            waits: AtomicUsize::new(0),
        },
    ))
}

const OUTPUT_EXTENT: [u32; 2] = [160, 100];
const RECTS: [[u32; 4]; 2] = [[3, 2, 80, 64], [90, 4, 64, 48]];
const PROJECTION_X: [f32; 2] = [-0.4, 0.4];

struct Output {
    target: RenderTarget,
    cameras: [RenderTarget; 2],
}

impl Output {
    fn new(device: &WgpuDevice) -> Self {
        Self {
            target: RenderTarget::new(device, Extent3d::new_2d(OUTPUT_EXTENT[0], OUTPUT_EXTENT[1])),
            cameras: RECTS.map(|[_, _, width, height]| {
                RenderTarget::new(device, Extent3d::new_2d(width, height))
            }),
        }
    }

    fn graph(
        &self,
        state: &ParticleGpuState,
        views: &[SurfaceView],
        frame: u64,
    ) -> Result<RenderGraph, crate::display_transform_gpu::DisplayTransformGpuError> {
        let queues = RenderQueues::default();
        let functions = DrawFunctions::new();
        let inputs: Vec<_> = views
            .iter()
            .zip(&self.cameras)
            .map(|(view, target)| SurfaceViewGraphInput {
                camera: view.camera,
                order: view.order,
                clear: prism_camera::ClearMode::ColorAndDepth,
                viewport: view.viewport,
                exposure_stops: view.exposure_stops,
                target,
                surface: *view,
                queues: &queues,
                lighting: None,
                upscale_pipeline: prism_render_driver::RenderPipelineId::from_parts(30, 1),
                upscale_bindings: prism_render_driver::BindGroupLayoutId::from_parts(31, 1),
            })
            .collect();
        let prepared = prepare_surface_view_graph(
            &self.target,
            &queues,
            &functions,
            None,
            None,
            &inputs,
            &FrameGraphTextureImports::default(),
            crate::TaaOutput::PrimaryWindow,
        )
        .unwrap();
        let (graph, _) = finalize_surface_view_graph::<prism_render_driver_null::NullInstance>(
            prepared,
            Some(state),
            None,
            &BTreeMap::new(),
            frame,
            None,
            crate::TaaOutput::PrimaryWindow,
            &mut crate::TaaHistoryCache::default(),
        )?;
        let stages = graph.blackboard().get::<SurfaceViewGraphStages>().unwrap();
        stages.assert_contract();
        assert_eq!(
            stages.manual.particle_passes, 0,
            "native mode must not draw a headless fallback"
        );
        assert_eq!(stages.cameras.len(), 2);
        assert!(stages
            .cameras
            .iter()
            .all(|(_, camera)| camera.particle_passes == 2));
        assert_eq!(stages.particle_passes, 4);
        assert_eq!(stages.copy_passes, 2);
        Ok(graph)
    }

    fn destroy(self, device: &WgpuDevice) {
        for target in self.cameras {
            target.destroy(device);
        }
        self.target.destroy(device);
    }
}

fn fixture(
    device: &WgpuDevice,
    queue: &ObservedQueue,
) -> (
    App,
    World,
    ParticleGpuState,
    ParticleGpuSimulationInputs,
    Vec<SurfaceView>,
) {
    fixture_with_generation(device, queue, 1)
}

fn fixture_with_generation(
    device: &WgpuDevice,
    queue: &ObservedQueue,
    generation: u64,
) -> (
    App,
    World,
    ParticleGpuState,
    ParticleGpuSimulationInputs,
    Vec<SurfaceView>,
) {
    let mut app = App::new();
    app.add_plugins(ParticleSimulationPlugin);
    set_particle_fixed_timestep(&mut app, core::time::Duration::from_millis(20));
    let config = app.world_mut().resource_mut::<ParticleSimulationConfig>();
    config.capacity = 4;
    config.params.spawn_position = [0.1, 0.0, 0.5];
    // More than one pixel per measured tick on the smaller camera. This keeps
    // the cache-invalidation image check independent of subpixel coverage.
    config.params.spawn_velocity = [2.5, 0.0, 0.0];
    config.params.spawn_lifetime = 10.0;
    app.world_mut().spawn((
        ParticleSimulationSource,
        ParticleRendererComponent {
            renderers: [
                ([6.0; 2], [2.0, 0.25, 0.0, 1.0]),
                ([3.0; 2], [0.0, 2.0, 0.5, 1.0]),
            ]
            .into_iter()
            .map(|(scale, tint)| ParticleRendererConfig {
                material: ParticleMaterialStyle {
                    scale,
                    tint,
                    softness: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
        },
    ));
    for (rect, projection_x) in RECTS.into_iter().zip(PROJECTION_X) {
        app.world_mut().spawn((
            PrismCamera::new(
                Mat4::from_translation(Vec3::new(projection_x, 0.0, 0.0)),
                rect,
            ),
            PrismFrustum(Frustum::from_planes([
                Plane::new(Vec3::X, 10.0),
                Plane::new(-Vec3::X, 10.0),
                Plane::new(Vec3::Y, 10.0),
                Plane::new(-Vec3::Y, 10.0),
                Plane::new(Vec3::Z, 10.0),
                Plane::new(Vec3::ZERO, 1.0),
            ])),
            GlobalTransform::default(),
        ));
    }
    let mut render = World::new();
    extract_particle_simulation_inputs(app.world_mut(), &mut render);
    extract_surface_views(app.world_mut(), &mut render);
    extract_particle_views(app.world_mut(), &mut render);
    let mut views: Vec<_> = render
        .resource::<ExtractedSurfaceViews>()
        .iter()
        .cloned()
        .collect();
    views.sort_by_key(|view| (view.camera.index(), view.camera.generation()));
    assert_eq!(views.len(), 2);
    assert_eq!(
        views.iter().map(|view| view.viewport).collect::<Vec<_>>(),
        RECTS
    );
    let mut inputs = render
        .remove_resource::<ParticleGpuSimulationInputs>()
        .unwrap();
    let mut state = ParticleGpuState::new(device, Arc::new(device.clone()), generation, 4).unwrap();
    let mut native = NativeParticleViews::default();
    native.requested = render.remove_resource::<ExtractedParticleViews>().unwrap();
    state.native_views = Some(native);
    // Bootstrap intentionally performs a dt=0 submission. Exclude it from the
    // two measured frames, each of which must consume exactly one fixed tick.
    state
        .prepare_gpu_snapshot(queue, &RenderAppConfig::default(), &mut inputs, 99)
        .unwrap();
    assert_eq!(state.applied_steps, 0);
    assert!(state.fabric_gpu_frame().is_none());
    assert_eq!(state.fabric_gpu_views().count(), 2);
    (app, render, state, inputs, views)
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn scalar(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(word(bytes, offset))
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "actual {actual}, expected {expected}"
    );
}

fn assert_gpu_buffers(state: &ParticleGpuState, queue: &WgpuQueue, tick: u64) -> f32 {
    let particles = queue
        .read_buffer(
            state.gpu_simulation.particles(),
            4 * prism_render_gpudriven::GPU_PARTICLE_STRIDE,
        )
        .expect("particle multiview pool readback must complete");
    let live: Vec<_> = particles
        .chunks_exact(prism_render_gpudriven::GPU_PARTICLE_STRIDE as usize)
        .enumerate()
        .filter(|(_, particle)| word(particle, 36) != 0)
        .collect();
    assert_eq!(
        live.len(),
        1,
        "repeated output rendering must not spawn another particle"
    );
    let (slot, particle) = live[0];
    let x = 0.1 + 0.05 * tick as f32;
    close(scalar(particle, 0), x);
    close(scalar(particle, 4), 0.0);
    close(scalar(particle, 8), 0.5);
    close(scalar(particle, 12), 0.02 * tick as f32);
    let count = queue
        .read_buffer(state.gpu_renderer.input_count_buffer(), 4)
        .expect("particle multiview input-count readback must complete");
    assert_eq!(word(&count, 0), 1, "bridge output count is GPU-authored");
    let stream = queue
        .read_buffer(
            state.gpu_renderer.input_buffer(),
            PARTICLE_RENDER_INPUT_STRIDE,
        )
        .expect("particle multiview stream readback must complete");
    close(scalar(&stream, 0), x);
    close(scalar(&stream, 4), 0.0);
    close(scalar(&stream, 8), 0.5);
    assert_eq!(
        word(&stream, 20),
        slot as u32,
        "bridge must retain the actual pool slot identity"
    );
    let mut private_work = BTreeSet::new();
    for view in state.native_views.as_ref().unwrap().active.values() {
        assert_eq!(view.renderer.emitters().len(), 2);
        for renderer in view.renderer.emitters() {
            assert_eq!(renderer.input_buffer(), state.gpu_renderer.input_buffer());
            assert_eq!(
                renderer.input_count_buffer(),
                state.gpu_renderer.input_count_buffer()
            );
            assert!(
                private_work.insert(renderer.sprite_args()),
                "camera/material work must not alias"
            );
            let control = queue
                .read_buffer(renderer.control(), 16)
                .expect("particle camera control readback must complete");
            assert_eq!(
                word(&control, 0),
                1,
                "every camera/material must cull the live particle"
            );
            let visible = queue
                .read_buffer(renderer.visible_indices(), 4)
                .expect("particle camera visibility readback must complete");
            assert_eq!(
                word(&visible, 0),
                0,
                "visible index addresses the compact bridge stream"
            );
            let args = queue
                .read_buffer(renderer.sprite_args(), 16)
                .expect("particle camera draw-args readback must complete");
            assert_eq!(word(&args, 0), 6);
            assert_eq!(
                word(&args, 4),
                1,
                "each indirect draw has one actual GPU instance"
            );
            assert_eq!(word(&args, 8), 0);
            // first_instance includes camera history identity; not always zero.
        }
    }
    assert_eq!(private_work.len(), 4);
    x
}

fn assert_image(pixels: &[[f32; 4]], width: u32, height: u32, expected_x: f32) -> f32 {
    assert_eq!(pixels.len(), (width * height) as usize);
    assert!(pixels.iter().flatten().all(|value| value.is_finite()));
    let mut red = 0;
    let mut green = 0;
    let mut lit = 0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for (index, pixel) in pixels.iter().enumerate() {
        red += usize::from(pixel[0] > 1.5 && pixel[1] < 0.5 && pixel[2] < 0.1);
        green += usize::from(pixel[0] < 0.1 && pixel[1] > 1.5 && pixel[2] > 0.4);
        if pixel[0] > 0.1 || pixel[1] > 0.1 || pixel[2] > 0.1 {
            lit += 1;
            sum_x += (index as u32 % width) as f32 + 0.5;
            sum_y += (index as u32 / width) as f32 + 0.5;
        }
    }
    assert!(
        red > 4 && green > 1,
        "both independent materials must rasterize: red={red}, green={green}"
    );
    let center_x = 2.0 * sum_x / lit as f32 / width as f32 - 1.0;
    let center_y = 2.0 * sum_y / lit as f32 / height as f32 - 1.0;
    assert!(
        (center_x - expected_x).abs() <= 2.0 / width as f32,
        "camera-local projection/viewport mismatch: got {center_x}, expected {expected_x}"
    );
    assert!(center_y.abs() <= 2.0 / height as f32);
    center_x
}

fn read_output(queue: &WgpuQueue, output: &Output, world_x: f32) -> [Vec<[f32; 4]>; 2] {
    let image = queue
        .read_texture_rgba16f(output.target.texture(), OUTPUT_EXTENT[0], OUTPUT_EXTENT[1])
        .expect("particle multiview composite readback must complete");
    let local: [_; 2] = core::array::from_fn(|camera| {
        let [_, _, width, height] = RECTS[camera];
        let pixels = queue
            .read_texture_rgba16f(output.cameras[camera].texture(), width, height)
            .expect("particle multiview camera readback must complete");
        assert_image(&pixels, width, height, world_x + PROJECTION_X[camera]);
        pixels
    });
    for y in 0..OUTPUT_EXTENT[1] {
        for x in 0..OUTPUT_EXTENT[0] {
            let pixel = image[(y * OUTPUT_EXTENT[0] + x) as usize];
            if let Some(camera) = RECTS.iter().position(|[left, top, width, height]| {
                x >= *left && x < left + width && y >= *top && y < top + height
            }) {
                let [left, top, width, _] = RECTS[camera];
                assert_eq!(
                    pixel,
                    local[camera][((y - top) * width + x - left) as usize],
                    "output camera copy misplaced/omitted at ({x}, {y})"
                );
            } else {
                assert!(
                    pixel[..3].iter().all(|channel| channel.abs() < 1e-5),
                    "particles escaped camera-local rectangle at ({x}, {y}): {pixel:?}"
                );
            }
        }
    }
    local
}

fn submit(queue: &ObservedQueue, recorded: &RecordedGraph) {
    // Match the runtime contract: mark only after the actual backend submit.
    let _completion = queue.submit_tracked(core::slice::from_ref(recorded.commands()));
    recorded.allocations().mark_submitted();
}

#[test]
fn physical_gpu_multioutput_cameras_execute_simulation_and_bridge_once() {
    let Some((device, queue)) = hardware() else {
        return;
    };
    let (mut app, mut render, mut state, mut inputs, views) = fixture(&device, &queue);
    let outputs = [Output::new(&device), Output::new(&device)];
    let expected_simulation: BTreeMap<String, usize> = state
        .gpu_simulation
        .dispatch_plan()
        .into_iter()
        .fold(BTreeMap::new(), |mut phases, dispatch| {
            use prism_render_gpudriven::GpuSimulationStageKind as Stage;
            // Independently assert the public dispatch plan against recorded
            // labels; the backend's entry-point helper is deliberately private.
            let entry = match dispatch.kind {
                Stage::Reset => "reset".to_owned(),
                Stage::Spawn => "spawn".to_owned(),
                Stage::Simulate => "simulate".to_owned(),
                Stage::Compact => "compact".to_owned(),
                Stage::EventScatter => "event_scatter".to_owned(),
                Stage::Prefix => "prefix".to_owned(),
                Stage::ScanSums(level) => format!("prefix_sums_{level}"),
                Stage::AddOffsets(level) => format!("add_offsets_{level}"),
                Stage::FinalizePrefix => "finalize_prefix".to_owned(),
                Stage::MarkActive => "mark_active".to_owned(),
            };
            *phases.entry(format!("particle-{entry}")).or_default() += 1;
            phases
        });
    let mut previous_images: Option<[Vec<[f32; 4]>; 2]> = None;
    for tick in 1..=2 {
        queue.reset();
        render.insert_resource(inputs);
        app.world_mut()
            .resource_mut::<ParticleSimulationConfig>()
            .params
            .spawn_count = u32::from(tick == 1);
        app.world_mut().run_schedule(FixedPreUpdate);
        extract_particle_simulation_inputs(app.world_mut(), &mut render);
        inputs = render
            .remove_resource::<ParticleGpuSimulationInputs>()
            .unwrap();
        let frame = 99 + tick;
        state
            .prepare_gpu_snapshot(&queue, &RenderAppConfig::default(), &mut inputs, frame)
            .unwrap();
        assert_eq!(state.applied_steps, tick);
        assert!(!inputs.recovering);
        let producer_graph = outputs[0].graph(&state, &views, frame).unwrap();
        let lease = state
            .graph_output_lease(&producer_graph)
            .expect("fresh frame publishes a producer");
        let token = producer_graph.submission_token().upgrade().unwrap();
        let producer = producer_graph.record(&device).unwrap();
        assert!(
            !token.is_submitted(),
            "record must not masquerade as queue submission"
        );
        assert!(!lease.confirm_submitted());
        let before_submit = queue.snapshot();
        assert!(
            matches!(
                outputs[1].graph(&state, &views, frame),
                Err(crate::display_transform_gpu::DisplayTransformGpuError::ParticleGraph)
            ),
            "a live unsubmitted producer cannot be reused by another output"
        );
        assert_eq!(
            queue.snapshot(),
            before_submit,
            "rejected consumer must not submit work"
        );
        submit(&queue, &producer);
        assert!(token.is_submitted());
        assert!(lease.confirm_submitted());
        let after_first = queue.snapshot();
        let consumer_graph = outputs[1].graph(&state, &views, frame).unwrap();
        assert!(
            state.graph_output_lease(&consumer_graph).is_none(),
            "consumer must import the published output"
        );
        let consumer = consumer_graph.record(&device).unwrap();
        submit(&queue, &consumer);
        let after_second = queue.snapshot();
        assert_eq!(
            queue.waits.load(Ordering::Relaxed),
            0,
            "output reuse must not wait for GPU retirement (readbacks start below)"
        );
        assert_eq!(after_first.simulations, 1);
        assert_eq!(after_second.simulations, 1);
        assert_eq!(after_second.simulation_phases, expected_simulation);
        assert_eq!(
            after_first.bridge_phases, after_second.bridge_phases,
            "second output must not submit any bridge phase"
        );
        assert_bridge_once(&after_second);
        assert_eq!(after_second.output_draws.len(), 2);
        let expected_args: BTreeSet<_> = state
            .native_views
            .as_ref()
            .unwrap()
            .active
            .values()
            .flat_map(|view| {
                view.renderer
                    .emitters()
                    .iter()
                    .map(|stage| stage.sprite_args())
            })
            .collect();
        assert_eq!(expected_args.len(), 4);
        for draws in &after_second.output_draws {
            assert_eq!(
                draws.len(),
                4,
                "two cameras times two independently authored materials per output"
            );
            assert_eq!(
                draws.iter().copied().collect::<BTreeSet<_>>(),
                expected_args
            );
        }
        // The first synchronization/readback occurs only AFTER both submissions.
        // Producer allocations and imported resource owners remain pinned here.
        let world_x = assert_gpu_buffers(&state, &queue.inner, tick);
        let first = read_output(&queue.inner, &outputs[0], world_x);
        let second = read_output(&queue.inner, &outputs[1], world_x);
        assert_eq!(
            first, second,
            "every output must receive both real camera images"
        );
        if let Some(previous) = previous_images.replace(first.clone()) {
            for camera in 0..2 {
                assert_ne!(previous[camera], first[camera],
                    "new frame must invalidate the bridge publication and rasterize the moved particle");
            }
        }
        assert!(queue.inner.wait_idle());
        consumer.release(&device);
        producer.release(&device);
    }
    state.destroy();
    for output in outputs {
        output.destroy(&device);
    }
}
