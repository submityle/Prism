use super::*;

use prism_app::presentation::{PresentationBridge, PresentationClockDomain, PresentationHostClock};
use prism_ecs::prelude::Entity;
use prism_render_driver::{Extent3d, Instance as _, TextureFormat};
use prism_render_driver_null::NullInstance;
use prism_render_runtime::RenderRuntime;
use std::sync::Arc;

struct FakeWindow;

#[derive(Clone, Copy)]
enum FirstParticleOutput {
    TwoVisible,
    Hidden,
    FailedBeforeSubmit,
}

// Keep this synchronous and Null-only: the parent owns physical GPU acceptance.
fn native_particle_multi_output_case(first: FirstParticleOutput) {
    use crate::particle::{EmberGpuAssetRegistry, ExtractedEmberExternalGpuResources};
    use crate::surface_view_frame::{SurfaceViewQueue, SurfaceViewQueues};
    use prism_render_architecture::particle::{
        asset_registry::EmberAssetRevision,
        attributes::AttributeLayoutPlan,
        authoring::{EmberAssetHandle, GraphAssetRef},
        compiler::{CompiledDispatch, CompiledGraph},
        modules::ResourceDecl,
        IterationDomain,
    };
    use prism_render_driver::{Backend, DeviceCapabilities, Features, Limits};

    let extent = Extent3d::new_2d(32, 16);
    let runtime = RenderRuntime::headless(
        NullInstance::with_capabilities(DeviceCapabilities {
            backend: Backend::Noop,
            features: Features::INDIRECT_DRAW,
            limits: Limits::desktop(),
        }),
        extent,
    )
    .unwrap();
    let device = runtime.device_handle();
    let baseline = device.live_resource_count();
    let (mut particle, mut inputs, mut surfaces) = crate::particle::multiview::native_fixture(
        device.clone(),
        runtime.queue(),
        runtime.device_generation(),
    );
    // Give each extracted camera a distinct stable window target, so this
    // fixture exercises output isolation rather than primary-window fallback.
    for (index, surface) in surfaces.iter_mut().enumerate() {
        surface.target = prism_camera::RenderTarget::Window((index + 1) as u64);
    }
    let mut ember = EmberGpuAssetRegistry::default();
    ember
        .upsert(
            device.as_ref(),
            runtime.queue(),
            EmberAssetHandle(7),
            GraphAssetRef(1),
            EmberAssetRevision::from_value(1),
            CompiledGraph {
                source: "@compute @workgroup_size(64) fn main() {}".into(),
                stage_order: vec![0],
                entry_points: vec!["main".into()],
                dispatches: vec![CompiledDispatch {
                    stage_index: 0,
                    domain: IterationDomain::PerParticle,
                    default_count: 1,
                    iterations: 1,
                    entry_point: "main".into(),
                }],
                bindings: vec![ResourceDecl::buffer("ember_dispatch_counts", 1, 0)],
                parameter_blocks: vec![],
                attribute_layout: AttributeLayoutPlan::build(0, &[]),
                attribute_bindings: vec![],
            },
            &prism_render_gpudriven::CompiledGraphGpuResourceRegistry::default(),
        )
        .unwrap();
    assert_eq!(ember.compute_pass_count(), 1);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    for id in [WindowId(1), WindowId(2)] {
        requests.attach_factory(id, RenderViewport::from_extent(extent), |runtime| {
            runtime.instance().create_surface(
                Arc::new(FakeWindow),
                runtime.device(),
                runtime.queue(),
            )
        });
    }
    match first {
        FirstParticleOutput::TwoVisible => {}
        FirstParticleOutput::Hidden => {
            requests.attach_factory(
                WindowId(0),
                RenderViewport::from_extent(extent),
                |_runtime| panic!("a hidden window must not acquire a surface"),
            );
            assert!(requests.update(WindowId(0), RenderViewport::from_extent(extent), false));
        }
        FirstParticleOutput::FailedBeforeSubmit => {
            requests.attach_factory(
                WindowId(0),
                RenderViewport::from_extent(extent),
                |_runtime| Err(SurfaceError::Unsupported),
            );
        }
    }
    let mut camera_targets = SurfaceViewTargets::default();
    let keys: Vec<_> = [WindowId(1), WindowId(2)]
        .into_iter()
        .flat_map(|id| {
            surfaces.iter().map(move |surface| {
                SurfaceViewTargetKey::new(
                    surface.camera,
                    SurfaceViewOutput::Window(id),
                    RenderViewport::new(surface.viewport[2], surface.viewport[3]),
                )
            })
        })
        .collect();
    camera_targets
        .reconcile(&runtime, &keys, |_, _, _| true)
        .unwrap();
    let mut targets = WindowRenderTargets::with_bridge(PresentationBridge::new(
        PresentationHostClock::new(PresentationClockDomain(1)),
    ));
    let functions = DrawFunctions::new();
    let manual = RenderQueues::new();
    let overrides = WindowRenderQueues::default();
    let texture_imports = crate::frame::FrameGraphTextureImports::default();
    let external = ExtractedEmberExternalGpuResources::default();
    let mut stats = RenderFrameStats::default();

    // First exercise the real driver, including first-*actual*-submit selection.
    // Next inspect each output's receipt separately, with the same frame-work
    // decision as drive_internal. Native cameras retain their local rectangles.
    for frame in [1, 2] {
        if frame == 2 {
            particle
                .prepare_gpu_snapshot(
                    runtime.queue(),
                    &crate::RenderAppConfig::default(),
                    &mut inputs,
                    frame,
                )
                .unwrap();
        }
        let mut cameras = SurfaceViewQueues::pending(frame, runtime.device_generation());
        cameras.publish(
            surfaces
                .iter()
                .enumerate()
                .map(|(index, surface)| SurfaceViewQueue {
                    surface: *surface,
                    frame_slot: index as u32,
                    queues: RenderQueues::new(),
                })
                .collect(),
        );
        let mut context = SurfaceViewDriveContext {
            cameras: &cameras,
            targets: &mut camera_targets,
            frame_index: frame,
            texture_imports: &texture_imports,
            ember: Some(&ember),
            ember_external: &external,
        };
        let before_ember = stats.ember_compute_passes_recorded;
        let before_submit = stats.frames_submitted;
        if frame == 1 {
            let receipt = targets.drive_with_surface_views(
                &runtime,
                &requests,
                &overrides,
                &manual,
                &functions,
                None,
                None,
                Some(&particle),
                context,
                &mut stats,
            );
            assert!(receipt.handled);
            assert!(receipt.first_submission.is_some());
            assert!(matches!(
                targets.status(WindowId(1)),
                Some(WindowFrameStatus::Presented { .. })
            ));
            assert!(matches!(
                targets.status(WindowId(2)),
                Some(WindowFrameStatus::Presented { .. })
            ));
            match first {
                FirstParticleOutput::TwoVisible => {}
                FirstParticleOutput::Hidden => assert!(matches!(
                    targets.status(WindowId(0)),
                    Some(WindowFrameStatus::Skipped)
                )),
                FirstParticleOutput::FailedBeforeSubmit => assert!(matches!(
                    targets.status(WindowId(0)),
                    Some(WindowFrameStatus::Failed(_))
                )),
            }
        } else {
            let mut first_submission = None;
            for id in [WindowId(1), WindowId(2)] {
                let before = stats.ember_compute_passes_recorded;
                let submission = targets
                    .drive_window(
                        id,
                        &requests.windows[&id],
                        &runtime,
                        &manual,
                        &functions,
                        None,
                        None,
                        Some(&particle),
                        Some(&mut context),
                        requests.primary_window() == Some(id),
                        first_submission.is_none(),
                        &mut stats,
                    )
                    .unwrap_or_else(|failure| panic!("window {id:?}: {:?}", failure.error))
                    .unwrap();
                assert_eq!(
                    stats.ember_compute_passes_recorded - before,
                    u64::from(first_submission.is_none()),
                    "Ember is frame-wide, not per camera/output"
                );
                first_submission.get_or_insert(submission);
                let receipt = stats.execution_trace.surface_graph.as_ref().unwrap();
                assert_eq!(receipt.manual.particle_passes, 0);
                assert_eq!(receipt.cameras.len(), 1);
                let selected = usize::from(id == WindowId(2));
                let (camera, local) = &receipt.cameras[0];
                assert_eq!(*camera, surfaces[selected].camera);
                let expected = particle
                    .fabric_gpu_views()
                    .find(|(entity, _)| entity == camera)
                    .unwrap()
                    .1
                    .draws()
                    .len();
                assert!(expected > 0);
                assert_eq!(local.particle_passes as usize, expected);
            }
        }
        assert_eq!(stats.frames_submitted - before_submit, 2);
        assert_eq!(stats.ember_compute_passes_recorded - before_ember, 1);
        assert!(stats.surface_view_error.is_none());
        assert_eq!(stats.surface_view_target_rejections.len(), 0);
    }
    assert!(runtime.wait_idle());
    assert!(targets.release(&runtime));
    camera_targets
        .reconcile(&runtime, &[], |_, _, _| true)
        .unwrap();
    ember.destroy(device.as_ref());
    particle.destroy();
    assert_eq!(device.live_resource_count(), baseline);
}

#[test]
fn native_particles_draw_each_output_camera_while_frame_work_runs_once() {
    native_particle_multi_output_case(FirstParticleOutput::TwoVisible);
}

#[test]
fn hidden_output_does_not_consume_particle_bridge_or_ember_frame_work() {
    native_particle_multi_output_case(FirstParticleOutput::Hidden);
}

#[test]
fn failed_presubmit_output_does_not_consume_particle_bridge_or_ember_frame_work() {
    native_particle_multi_output_case(FirstParticleOutput::FailedBeforeSubmit);
}

impl raw_window_handle::HasWindowHandle for FakeWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

impl raw_window_handle::HasDisplayHandle for FakeWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

#[test]
fn null_window_request_lifecycle_is_explicit() {
    let id = WindowId(7);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    assert!(!requests.contains(id));
    assert_eq!(requests.primary_window(), None);

    requests.attach_factory(id, RenderViewport::new(64, 32), |_runtime| {
        Err(SurfaceError::Unsupported)
    });
    assert!(requests.contains(id));
    assert_eq!(requests.primary_window(), Some(id));
    assert!(!requests.set_primary_window(WindowId(99)));
    assert!(requests.update(id, RenderViewport::new(80, 40), false));
    assert_eq!(requests.ids().collect::<Vec<_>>(), vec![id]);
    assert!(requests.remove(id));
    assert!(!requests.contains(id));
    assert_eq!(requests.primary_window(), None);

    requests.use_offscreen();
    assert!(!requests.enabled);
}

#[test]
fn primary_window_selection_is_stable_and_reselects_after_removal() {
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    for id in [WindowId(8), WindowId(3)] {
        requests.attach_factory(id, RenderViewport::new(8, 8), |_runtime| {
            Err(SurfaceError::Unsupported)
        });
    }
    assert_eq!(requests.primary_window(), Some(WindowId(8)));
    assert!(requests.set_primary_window(WindowId(3)));
    assert_eq!(requests.primary_window(), Some(WindowId(3)));
    assert!(requests.remove(WindowId(3)));
    assert_eq!(requests.primary_window(), Some(WindowId(8)));
    requests.use_offscreen();
    assert_eq!(requests.primary_window(), None);
}

#[test]
fn surface_view_seal_identity_mismatch_is_diagnostic_not_panic() {
    let extent = Extent3d::new_2d(16, 16);
    let runtime =
        RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent).expect("null runtime");
    let other = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
        .expect("independent null runtime");
    let view = SurfaceViewTargetKey::new(
        Entity::from_bits((1_u64 << 32) | 1).expect("valid entity"),
        SurfaceViewOutput::Offscreen(1),
        RenderViewport::from_extent(extent),
    );
    let mut targets = SurfaceViewTargets::default();
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
}

#[test]
fn null_surface_presents_and_releases_window_resources() {
    let extent = Extent3d::new_2d(16, 16);
    let runtime =
        RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent).expect("null runtime");
    let id = WindowId(1);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    requests.attach_factory(id, RenderViewport::from_extent(extent), |runtime| {
        runtime
            .instance()
            .create_surface(Arc::new(FakeWindow), runtime.device(), runtime.queue())
    });

    let bridge = PresentationBridge::new(PresentationHostClock::new(PresentationClockDomain(1)));
    let mut targets = WindowRenderTargets::with_bridge(bridge);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let mut stats = RenderFrameStats::default();
    let drive_receipt = targets.drive(
        &runtime,
        &requests,
        &WindowRenderQueues::default(),
        &queues,
        &functions,
        None,
        None,
        &mut stats,
    );
    assert!(drive_receipt.handled);
    assert!(drive_receipt.first_submission.is_some());
    assert_eq!(targets.len(), 1);
    assert!(matches!(
        targets.status(id),
        Some(WindowFrameStatus::Presented { .. })
    ));
    let receipt = stats
        .execution_trace
        .graph
        .expect("presented frame publishes graph receipt");
    assert_eq!(receipt.present_passes, 1);
    assert!(runtime.wait_idle());
    assert!(targets.release(&runtime));
    assert!(targets.is_empty());
}

#[test]
fn suboptimal_surface_frame_is_presented_and_reconfigured_before_next_acquire() {
    let extent = Extent3d::new_2d(16, 16);
    let runtime =
        RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent).expect("null runtime");
    let id = WindowId(2);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    requests.attach_factory(id, RenderViewport::from_extent(extent), |runtime| {
        runtime
            .instance()
            .create_surface(Arc::new(FakeWindow), runtime.device(), runtime.queue())
            .map(|surface| {
                surface.simulate_next_suboptimal_frame();
                surface
            })
    });
    let mut targets = WindowRenderTargets::default();
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let mut stats = RenderFrameStats::default();

    for _ in 0..2 {
        let receipt = targets.drive(
            &runtime,
            &requests,
            &WindowRenderQueues::default(),
            &queues,
            &functions,
            None,
            None,
            &mut stats,
        );
        assert!(receipt.first_submission.is_some());
        assert!(matches!(
            targets.status(id),
            Some(WindowFrameStatus::Presented { .. })
        ));
    }

    assert!(runtime.wait_idle());
    assert!(targets.release(&runtime));
}

#[test]
fn physical_gpu_hdr_present_applies_display_transform_once_for_unorm_and_srgb_targets() {
    use prism_render_architecture::particle::half_float_f16::{f16_bits_to_f32, f32_to_f16_bits};
    use prism_render_driver::{
        ImageDataLayout, RenderQueue as _, SurfaceImage, TextureDescriptor, TextureState,
        TextureUsages, TextureViewDescriptor, TextureWrite,
    };
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{ImportedTexture, RenderGraph, TextureDesc};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(device_and_queue) => device_and_queue,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
            );
            eprintln!("SKIPPED HDR present readback: no adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SKIPPED HDR present readback: non-hardware adapter: {evidence}");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical GPU was found but device creation failed: {error}");
        }
    };
    let device = Arc::new(device);
    let device_dyn: Arc<dyn RenderDevice + Send + Sync> = device.clone();
    let extent = Extent3d::new_2d(3, 2);
    let target = RenderTarget::new(&*device, extent);
    assert_eq!(target.format(), TextureFormat::Rgba16Float);
    // Direct HDR texture fixtures validate display/capture, not scene lighting
    // or the OS compositor. Different rows also catch present-coordinate errors.
    let pixels: [[f32; 4]; 6] = [
        [2.0, 4.0, 8.0, 0.25],
        [0.001, 0.003, 0.004, 0.5],
        [0.125, 0.5, 1.5, 0.75],
        [32.0, 8.0, 2.0, 1.0],
        [-0.25, 0.18, 0.75, 0.0],
        [0.25, 1.0, 4.0, 0.125],
    ];
    let encoded = pixels.map(|pixel| pixel.map(f32_to_f16_bits));
    let uploaded = encoded.map(|pixel| pixel.map(f16_bits_to_f32));
    queue.write_texture(
        TextureWrite {
            texture: target.texture(),
            mip_level: 0,
            origin: [0, 0, 0],
            size: extent,
        },
        bytemuck::cast_slice(&encoded),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * 8),
            rows_per_image: Some(extent.height),
        },
    );
    let linear_capture = queue.read_texture_rgba16f(target.texture(), extent.width, extent.height);
    assert_eq!(linear_capture.as_slice(), uploaded.as_slice());

    let lifetime = WindowLifetime::new();
    let transform = DisplayTransformGpu::new(
        device_dyn,
        target.view(),
        extent,
        Arc::clone(&lifetime.in_flight),
        Arc::clone(&lifetime.destroy_allowed),
    )
    .expect("HDR display transform");
    // Use the production present pass with readable attachments of both types.
    // In particular the sRGB attachment must decode encoded input before its
    // hardware store encodes it again; the resulting bytes must not brighten.
    let outputs = [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb].map(|format| {
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            format,
            TextureUsages::RENDER_ATTACHMENT.union(TextureUsages::COPY_SRC),
        ));
        let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
        let pipeline = PresentPipeline::new(
            Arc::clone(&device),
            transform.output_view(),
            format,
            Arc::clone(&lifetime.in_flight),
            Arc::clone(&lifetime.destroy_allowed),
        );
        (texture, view, format, pipeline)
    });
    for config in [
        OcioConfig::srgb(),
        OcioConfig {
            exposure_stops: -3.0,
            ..OcioConfig::aces()
        },
        OcioConfig {
            exposure_stops: 1.0,
            ..OcioConfig::aces()
        },
    ] {
        let mut graph = RenderGraph::new(extent);
        let source = graph.import_texture(
            "hdr-present-source",
            TextureDesc::color(target.format()).with_usage(TextureUsages::TEXTURE_BINDING),
            ImportedTexture {
                texture: target.texture(),
                default_view: target.view(),
                entry_state: TextureState::initial(),
                exit_state: TextureState::copy_src(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let transformed = transform
            .append(&mut graph, &queue, &config, source)
            .expect("HDR compute pass");
        for (texture, view, format, pipeline) in &outputs {
            pipeline.append(
                &mut graph,
                transformed,
                SurfaceImage {
                    texture: *texture,
                    view: *view,
                    format: *format,
                    size: extent,
                },
            );
        }
        assert_eq!(
            graph.pass_count(),
            3,
            "one transform followed by two presents"
        );
        let recorded = graph
            .execute(&*device, &queue)
            .expect("HDR compute and present graph");
        let display =
            queue.read_texture_rgba8(transform.output_texture(), extent.width, extent.height);
        assert_eq!(display.len(), pixels.len() * 4, "SDR capture stays RGBA8");
        for (texture, _, format, _) in &outputs {
            let presented = queue.read_texture_rgba8(*texture, extent.width, extent.height);
            assert_eq!(presented.len(), display.len());
            for (index, ((source, encoded), actual)) in uploaded
                .iter()
                .zip(display.chunks_exact(4))
                .zip(presented.chunks_exact(4))
                .enumerate()
            {
                let expected = config
                    .reference_transform([source[0], source[1], source[2]])
                    .expect("finite HDR input");
                for channel in 0..3 {
                    assert!(
                        (f32::from(encoded[channel]) / 255.0 - expected[channel]).abs()
                            <= 2.0 / 255.0,
                        "SDR capture: exposure={} pixel={index} channel={channel}",
                        config.exposure_stops
                    );
                    assert!(
                        actual[channel].abs_diff(encoded[channel]) <= 1,
                        "present must not repeat exposure/tone-map/transfer: format={format:?} exposure={} pixel={index} channel={channel} display={} present={}",
                        config.exposure_stops, encoded[channel], actual[channel]
                    );
                }
                let expected_alpha = (source[3] * 255.0).round() as u8;
                assert!(
                    encoded[3].abs_diff(expected_alpha) <= 1,
                    "untransformed alpha"
                );
                assert_eq!(actual[3], encoded[3], "present preserves alpha");
            }
        }
        assert_eq!(
            queue.read_texture_rgba16f(target.texture(), extent.width, extent.height),
            linear_capture,
            "linear capture must retain HDR/negative values across exposure changes"
        );
        // All preceding compute/present work completed before synchronous readback.
        recorded.release(&*device);
    }
    for (texture, view, _, pipeline) in outputs {
        drop(pipeline);
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
    }
    drop(transform);
    target.destroy(&*device);
}

#[cfg(feature = "winit")]
#[test]
#[ignore = "requires a real OS window and display compositor"]
fn real_window_presentation_acceptance_entry() {
    use crate::PrismRenderPlugin;
    use prism_app::runner::WinitRunner;
    use prism_app::{App, Plugin};
    use prism_render_driver_wgpu::WgpuInstance;

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::new(WgpuInstance::new()));
    app.add_plugins(WinitPresentationPlugin::<WgpuInstance>::default());
    let _exit = WinitRunner::new(app).run();
}
