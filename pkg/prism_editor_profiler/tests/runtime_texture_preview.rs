//! Hardware acceptance of previews inside the real prepared/recorded graph.

use prism_editor_profiler::{
    gpu_preview::{TexturePreview, TexturePreviewBinding, TexturePreviewLimits},
    runtime_preview::{RuntimePreviewError, RuntimePreviewFrame, RuntimePreviewSubmission},
    GraphCaptureInputs, GraphCaptureLimits,
};
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceId, ResourceLifetime},
    render_fabric::*,
};
use prism_render_driver::{BufferId, BufferState, BufferUsages, Extent3d, TextureState};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_graph::{capacity::GraphLimits, ImportedBuffer, ImportedTexture};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

const FEATURE: FeatureId = FeatureId(973);
const RED: ResourceKey = ResourceKey::of(FEATURE, 0);
const GREEN: ResourceKey = ResourceKey::of(FEATURE, 1);
const IMAGE: ResourceKey = ResourceKey::of(FEATURE, 2);

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
fn usage(key: ResourceKey, access: AccessKind, input: ResourceInput) -> ResourceUse {
    ResourceUse {
        key,
        access,
        input,
        scope: ResourceScope::Whole,
        frame_lag: None,
    }
}
fn pass(key: u32, queue: QueueClass, accesses: Vec<ResourceUse>) -> PassTemplate {
    PassTemplate {
        id: PassKey(key),
        queue,
        depends_on: &[],
        accesses: table(accesses),
    }
}
fn external(template: ResourceTemplate, physical: PhysicalResource) -> FrameResourceBinding {
    FrameResourceBinding {
        key: template.key,
        frame: 71,
        version: 0,
        physical,
        layout: template.layout,
        alignment: template.alignment,
    }
}
struct Sources {
    device: WgpuDevice,
    buffers: [BufferId; 2],
}
#[derive(Clone, Copy)]
struct CopyImage {
    extent: [u32; 3],
    mip_levels: u32,
    mip: u32,
    layer: u32,
}
impl Default for CopyImage {
    fn default() -> Self {
        Self {
            extent: [3, 2, 1],
            mip_levels: 1,
            mip: 0,
            layer: 0,
        }
    }
}
impl Sources {
    fn new(device: &WgpuDevice, queue: &WgpuQueue) -> Self {
        let buffers = [[255, 0, 0, 255], [0, 255, 0, 255]].map(|color| {
            let buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("runtime preview source"),
                size: 512,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut padded = [0; 512];
            for y in 0..2 {
                for x in 0..3 {
                    padded[y * 256 + x * 4..y * 256 + x * 4 + 4].copy_from_slice(&color);
                }
            }
            queue.wgpu_queue().write_buffer(&buffer, 0, &padded);
            device.register_buffer(&buffer)
        });
        Self {
            device: device.clone(),
            buffers,
        }
    }
    fn graph(
        &self,
        imported: Option<&wgpu::Texture>,
        callbacks: Arc<AtomicUsize>,
    ) -> (
        RuntimeFeatureGraph,
        Option<(TextureId, prism_render_driver::TextureViewId)>,
    ) {
        self.graph_with_lifetime(imported, callbacks, ResourceLifetime::Transient)
    }
    fn graph_with_lifetime(
        &self,
        imported: Option<&wgpu::Texture>,
        callbacks: Arc<AtomicUsize>,
        lifetime: ResourceLifetime,
    ) -> (
        RuntimeFeatureGraph,
        Option<(TextureId, prism_render_driver::TextureViewId)>,
    ) {
        self.graph_with_subresource(imported, callbacks, lifetime, CopyImage::default())
    }
    fn graph_with_subresource(
        &self,
        imported: Option<&wgpu::Texture>,
        callbacks: Arc<AtomicUsize>,
        lifetime: ResourceLifetime,
        image: CopyImage,
    ) -> (
        RuntimeFeatureGraph,
        Option<(TextureId, prism_render_driver::TextureViewId)>,
    ) {
        let resources = [
            ResourceTemplate {
                key: RED,
                lifetime: ResourceLifetime::Imported,
                alignment: 256,
                layout: ResourceLayout::Buffer {
                    bytes: 512,
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                },
            },
            ResourceTemplate {
                key: GREEN,
                lifetime: ResourceLifetime::Imported,
                alignment: 256,
                layout: ResourceLayout::Buffer {
                    bytes: 512,
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                },
            },
            ResourceTemplate {
                key: IMAGE,
                lifetime: if imported.is_some() {
                    if lifetime == ResourceLifetime::Persistent {
                        ResourceLifetime::Persistent
                    } else {
                        ResourceLifetime::Imported
                    }
                } else {
                    lifetime
                },
                alignment: 256,
                layout: ResourceLayout::Texture {
                    extent: image.extent,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: image.mip_levels,
                    samples: 1,
                    usage: TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST
                        | TextureUsages::TEXTURE_BINDING,
                },
            },
        ];
        // Deliberately scramble registration; actual dependencies choose order.
        // Original buffer-before-texture accesses also differ from RHI order.
        let passes = vec![
            pass(
                20,
                QueueClass::Transfer,
                vec![
                    usage(GREEN, AccessKind::TransferRead, ResourceInput::Initial),
                    usage(
                        IMAGE,
                        AccessKind::TransferWrite,
                        ResourceInput::ProducedBy(PassKey(10)),
                    ),
                ],
            ),
            pass(
                30,
                QueueClass::Compute,
                vec![usage(
                    IMAGE,
                    AccessKind::SampledRead,
                    ResourceInput::ProducedBy(PassKey(20)),
                )],
            ),
            pass(
                10,
                QueueClass::Transfer,
                vec![
                    usage(RED, AccessKind::TransferRead, ResourceInput::Initial),
                    usage(IMAGE, AccessKind::TransferWrite, ResourceInput::Initial),
                ],
            ),
        ];
        let catalog = FeatureCatalog::new(
            table(vec![FeatureDefinition::subsystem(
                FEATURE,
                RenderDomain::Surface,
                ServiceSet::EMPTY,
                table(vec![FeatureVariant {
                    id: VariantId(1),
                    rank: 0,
                    min_quality: RenderQualityTier::Low,
                    required_caps: CapabilitySet::EMPTY,
                    requires_services_extra: ServiceSet::EMPTY,
                    passes: table(passes),
                    shader_permutation: None,
                }]),
            )]),
            table(vec![RED, GREEN, IMAGE]),
        )
        .unwrap();
        let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap();
        let mut bindings: Vec<_> = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, &id)| external(resources[i], PhysicalResource::Buffer { id, offset: 0 }))
            .collect();
        let mut imports: Vec<_> = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, &buffer)| RuntimeImport::Buffer {
                resource: ResourceId(i as u32),
                imported: ImportedBuffer {
                    buffer,
                    entry_state: BufferState::initial(),
                },
            })
            .collect();
        let registered = imported.map(|texture| {
            let id = self.device.register_texture(texture);
            let view = self
                .device
                .register_texture_view(&texture.create_view(&Default::default()));
            bindings.push(external(resources[2], PhysicalResource::Texture(id)));
            imports.push(RuntimeImport::Texture {
                resource: ResourceId(2),
                imported: ImportedTexture {
                    texture: id,
                    default_view: view,
                    entry_state: TextureState::undefined(),
                    exit_state: TextureState::undefined(),
                    mip_levels: image.mip_levels,
                    array_layers: image.extent[2],
                },
            });
            (id, view)
        });
        let graph = ResourceTemplates::new(&catalog, &resources)
            .unwrap()
            .compile(&selection, 71, &bindings, &[], GraphLimits::default())
            .unwrap();
        let mut recorders = Vec::new();
        for key in [10, 20] {
            let callbacks = callbacks.clone();
            recorders.push(RuntimePassBinding::new(PassKey(key), move |ctx| {
                callbacks.fetch_add(1, Ordering::Relaxed);
                let RuntimeResourceHandle::Buffer {
                    handle: source,
                    offset,
                    ..
                } = ctx.accesses()[0].handle
                else {
                    panic!("source buffer")
                };
                let RuntimeResourceHandle::Texture(target) = ctx.accesses()[1].handle else {
                    panic!("target texture")
                };
                ctx.commands().copy_buffer_to_texture(
                    source,
                    offset,
                    Some(256),
                    Some(2),
                    target,
                    image.mip,
                    [0, 0, image.layer],
                    Extent3d::new_2d(3, 2),
                );
            }));
        }
        recorders.push(RuntimePassBinding::new(PassKey(30), move |_| {
            callbacks.fetch_add(1, Ordering::Relaxed);
        }));
        (
            graph
                .prepare_runtime(
                    Extent3d::new_2d(image.extent[0], image.extent[1]),
                    &imports,
                    recorders,
                )
                .unwrap(),
            registered,
        )
    }
}
impl Drop for Sources {
    fn drop(&mut self) {
        for id in self.buffers {
            self.device.release_buffer(id);
        }
    }
}
fn selected(frame: &RuntimePreviewFrame<'_>, key: u32) -> TexturePreviewBinding {
    let pass = frame
        .report()
        .passes()
        .iter()
        .position(|p| p.key == key)
        .unwrap() as u32;
    let resource = frame
        .report()
        .resources()
        .iter()
        .position(|r| r.slot == 2)
        .unwrap() as u32;
    frame.binding(pass, resource).unwrap()
}
fn finish(device: &WgpuDevice, pending: &mut RuntimePreviewSubmission) -> TexturePreview {
    device
        .wgpu_device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(pending.submission().clone()),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    let result = pending.poll().unwrap().expect("one actual map callback");
    assert_eq!(result.handle, pending.handle());
    assert!(Arc::ptr_eq(
        result.preview.as_ref().unwrap().binding().report(),
        pending.binding().report()
    ));
    assert!(
        pending.is_complete(),
        "full graph finished and allocations retired"
    );
    assert!(
        pending.poll().unwrap().is_none(),
        "publication is single use"
    );
    result.preview.unwrap()
}
fn record<'a>(
    graph: RuntimeFeatureGraph,
    device: &WgpuDevice,
    queue: &'a WgpuQueue,
) -> RuntimePreviewFrame<'a> {
    RuntimePreviewFrame::record(
        graph,
        device,
        queue,
        GraphCaptureInputs::default(),
        GraphCaptureLimits::default(),
    )
    .unwrap()
}
fn assert_color(preview: &TexturePreview, color: [u8; 4]) {
    assert_eq!((preview.width(), preview.height()), (3, 2));
    assert_eq!(preview.bytes(), color.repeat(6));
}

#[test]
fn prepared_runtime_preview_copies_exact_pass_and_retires_after_full_submission() {
    let instance = WgpuInstance::new();
    let (device, queue) = instance
        .request_default_hardware_device()
        .expect("requires a physical GPU, never skips or accepts CPU rendering");
    eprintln!(
        "physical GPU runtime preview: {:?}",
        instance.default_adapter_evidence()
    );
    let sources = Sources::new(&device, &queue);
    let native = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("imported real runtime image"),
            size: wgpu::Extent3d {
                width: 3,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
    for imported in [false, true] {
        for (key, expected) in [(10, [255, 0, 0, 255]), (20, [0, 255, 0, 255])] {
            let callbacks = Arc::new(AtomicUsize::new(0));
            let (graph, registered) = sources.graph(imported.then_some(&native), callbacks.clone());
            let frame = record(graph, &device, &queue);
            assert_eq!(
                callbacks.load(Ordering::Relaxed),
                3,
                "all callbacks recorded exactly once"
            );
            let binding = selected(&frame, key);
            let mut pending = frame
                .submit(binding, 0, 0, TexturePreviewLimits::default())
                .unwrap();
            assert_color(&finish(&device, &mut pending), expected);
            assert_eq!(
                callbacks.load(Ordering::Relaxed),
                3,
                "splitting does not rerun recorders"
            );
            if let Some((texture, view)) = registered {
                // Graph retirement must never release host-owned imports.
                assert_eq!(device.wgpu_texture(texture).width(), 3);
                device.release_texture_view(view);
                device.release_texture(texture);
            }
        }
    }
    // Rejected requests return the original, still unsubmitted recording.
    let (graph, _) = sources.graph(None, Arc::new(AtomicUsize::new(0)));
    let frame = record(graph, &device, &queue);
    let binding = selected(&frame, 10);
    let stale = TexturePreviewBinding::new(
        Arc::new((**frame.report()).clone()),
        binding.pass(),
        binding.resource(),
    )
    .unwrap();
    let failure = frame
        .submit(stale, 0, 0, TexturePreviewLimits::default())
        .err()
        .expect("stale report rejected");
    assert!(matches!(failure.error, RuntimePreviewError::StaleReport));
    let frame = failure.frame;
    let failure = frame
        .submit(
            binding.clone(),
            0,
            0,
            TexturePreviewLimits {
                max_pixels: 1,
                ..Default::default()
            },
        )
        .err()
        .expect("over-budget copy rejected");
    assert!(matches!(failure.error, RuntimePreviewError::Texture(_)));
    let mut pending = failure
        .frame
        .submit(binding, 0, 0, TexturePreviewLimits::default())
        .unwrap();
    assert_color(&finish(&device, &mut pending), [255, 0, 0, 255]);
    // Cancel suppresses pixels while graph suffix and resource retirement run.
    let (graph, _) = sources.graph(None, Arc::new(AtomicUsize::new(0)));
    let frame = record(graph, &device, &queue);
    let binding = selected(&frame, 10);
    let mut pending = frame
        .submit(binding, 0, 0, TexturePreviewLimits::default())
        .unwrap();
    pending.cancel().unwrap();
    device
        .wgpu_device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(pending.submission().clone()),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    assert!(pending.poll().unwrap().is_none());
    assert!(pending.is_complete());
    // The queue retains retirement ownership even when the UI drops its handle.
    let (graph, _) = sources.graph(None, Arc::new(AtomicUsize::new(0)));
    let frame = record(graph, &device, &queue);
    let binding = selected(&frame, 10);
    let pending = frame
        .submit(binding, 0, 0, TexturePreviewLimits::default())
        .unwrap();
    let submission = pending.submission().clone();
    drop(pending);
    device
        .wgpu_device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    // Discarding a recording is also legal and submits no graph commands.
    let (graph, _) = sources.graph(None, Arc::new(AtomicUsize::new(0)));
    drop(record(graph, &device, &queue));
    // Persistent resources are imports; the host keeps them across frames.
    let (graph, registered) = sources.graph_with_lifetime(
        Some(&native),
        Arc::new(AtomicUsize::new(0)),
        ResourceLifetime::Persistent,
    );
    let frame = record(graph, &device, &queue);
    let binding = selected(&frame, 10);
    let mut pending = frame
        .submit(binding, 0, 0, TexturePreviewLimits::default())
        .unwrap();
    assert_color(&finish(&device, &mut pending), [255, 0, 0, 255]);
    drop(pending);
    let (persistent, view) = registered.unwrap();
    assert_eq!(device.wgpu_texture(persistent).width(), 3);
    device.release_texture_view(view);
    device.release_texture(persistent);
    let (graph, registered) = sources.graph(Some(&native), Arc::new(AtomicUsize::new(0)));
    let frame = record(graph, &device, &queue);
    let binding = selected(&frame, 20);
    let mut pending = frame
        .submit(binding, 0, 0, TexturePreviewLimits::default())
        .unwrap();
    assert_color(&finish(&device, &mut pending), [0, 255, 0, 255]);
    let (import, view) = registered.unwrap();
    device.release_texture_view(view);
    device.release_texture(import);
}

#[test]
fn runtime_preview_copies_selected_mip_and_array_layer_with_exact_accounting() {
    let instance = WgpuInstance::new();
    let (device, queue) = instance
        .request_default_hardware_device()
        .expect("requires a physical GPU, never skips or accepts CPU rendering");
    let sources = Sources::new(&device, &queue);
    // The selected 3x2 image lives at mip 1, layer 1 of a 6x4x2 texture.
    // Other subresources remain zero, so copying either default would fail.
    let image = CopyImage {
        extent: [6, 4, 2],
        mip_levels: 2,
        mip: 1,
        layer: 1,
    };
    let native = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("runtime mip and array layer acceptance"),
            size: wgpu::Extent3d {
                width: image.extent[0],
                height: image.extent[1],
                depth_or_array_layers: image.extent[2],
            },
            mip_level_count: image.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
    for imported in [false, true] {
        for (key, expected) in [(10, [255, 0, 0, 255]), (20, [0, 255, 0, 255])] {
            let callbacks = Arc::new(AtomicUsize::new(0));
            let (graph, registered) = sources.graph_with_subresource(
                imported.then_some(&native),
                callbacks.clone(),
                ResourceLifetime::Transient,
                image,
            );
            let mut frame = record(graph, &device, &queue);
            let binding = selected(&frame, key);
            for (mip, layer) in [(2, 1), (1, 2)] {
                let failure = frame
                    .submit(binding.clone(), mip, layer, TexturePreviewLimits::default())
                    .err()
                    .expect("out-of-range subresource must return the unsubmitted graph");
                assert!(matches!(failure.error, RuntimePreviewError::Texture(_)));
                frame = failure.frame;
            }
            let mut pending = frame
                .submit(
                    binding,
                    image.mip,
                    image.layer,
                    TexturePreviewLimits::default(),
                )
                .unwrap();
            assert_eq!((pending.mip(), pending.layer()), (1, 1));
            assert_eq!(pending.occupied_slots(), 1);
            assert_eq!(pending.allocated_bytes(), 512);
            let preview = finish(&device, &mut pending);
            assert_color(&preview, expected);
            assert_eq!((preview.mip(), preview.layer()), (1, 1));
            assert_eq!(pending.occupied_slots(), 0);
            assert_eq!(pending.allocated_bytes(), 0);
            assert_eq!(callbacks.load(Ordering::Relaxed), 3);
            if let Some((texture, view)) = registered {
                assert_eq!(device.wgpu_texture(texture).mip_level_count(), 2);
                device.release_texture_view(view);
                device.release_texture(texture);
            }
        }
    }
}
