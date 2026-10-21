//! App-owned production runtime capture and native compositor evidence.

use super::*;
use crate::{DesktopContentImages, DesktopGpu};
use alloc::sync::Arc;
use prism_editor_profiler::{
    runtime_preview::RuntimePreviewFrame, GraphCaptureInputs, GraphCaptureLimits,
};
use prism_editor_ui::{FrameGraphPanelAction, FrameGraphPanelController, UiEvent};
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceLifetime},
    render_fabric::*,
};
use prism_render_driver::{Extent3d, TextureState};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_graph::{capacity::GraphLimits, ImportedTexture};
use prism_ui_layout::{Point, Rect, Size};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

fn table<T>(items: Vec<T>) -> &'static [T] {
    Box::leak(items.into_boxed_slice())
}

/// A real host producer: the queue and imported texture belong to the host,
/// while the app consumes each borrowed frame synchronously through the seam.
struct Source {
    device: WgpuDevice,
    queue: WgpuQueue,
    texture: wgpu::Texture,
    colors: [[u8; 4]; 2],
    next: usize,
    captures: Arc<AtomicUsize>,
}

impl Source {
    fn new(device: WgpuDevice, queue: WgpuQueue) -> Self {
        let texture = device
            .wgpu_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("app runtime framegraph source"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 2,
                    depth_or_array_layers: 2,
                },
                mip_level_count: 2,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
        Self {
            device,
            queue,
            texture,
            colors: [[255, 0, 0, 255], [0, 255, 0, 255]],
            next: 0,
            captures: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn write_color(&self, color: [u8; 4]) {
        self.queue.wgpu_queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &color.repeat(8),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 4,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl DesktopRuntimePreviewSource for Source {
    fn capture(
        &mut self,
        submit: &mut dyn FnMut(RuntimePreviewFrame<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let color = self.colors[self.next % self.colors.len()];
        self.next = self.next.saturating_add(1);
        self.write_color(color);
        let frame = runtime_frame(&self.device, &self.queue, &self.texture)?;
        self.captures.fetch_add(1, Ordering::Relaxed);
        submit(frame)
    }
}

fn runtime_frame<'q>(
    device: &WgpuDevice,
    queue: &'q WgpuQueue,
    source: &wgpu::Texture,
) -> Result<RuntimePreviewFrame<'q>, String> {
    let feature = FeatureId(19_994);
    let key = ResourceKey::of(feature, 0);
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            feature,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            table(vec![FeatureVariant {
                id: VariantId(1),
                rank: 0,
                min_quality: RenderQualityTier::Low,
                required_caps: CapabilitySet::EMPTY,
                requires_services_extra: ServiceSet::EMPTY,
                shader_permutation: None,
                passes: table(vec![PassTemplate {
                    id: PassKey(1),
                    queue: QueueClass::Transfer,
                    depends_on: &[],
                    accesses: table(vec![ResourceUse {
                        key,
                        access: AccessKind::TransferRead,
                        scope: ResourceScope::Whole,
                        frame_lag: None,
                        input: ResourceInput::Initial,
                    }]),
                }]),
            }]),
        )]),
        table(vec![key]),
    )
    .map_err(|e| e.to_string())?;
    let enabled_subsystems = table(vec![feature]);
    let referenceable_responses = table(Vec::<FeatureId>::new());
    let roots =
        FrozenRoots::from_build_profile(&catalog, enabled_subsystems, referenceable_responses)
            .map_err(|e| e.to_string())?;
    let selected = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .map_err(|e| e.to_string())?;
    let template = ResourceTemplate {
        key,
        lifetime: ResourceLifetime::Imported,
        alignment: 256,
        layout: ResourceLayout::Texture {
            extent: [4, 2, 2],
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
        },
    };
    let texture = device.register_texture(source);
    let view = device.register_texture_view(&source.create_view(&Default::default()));
    let graph = ResourceTemplates::new(&catalog, &[template])
        .map_err(|e| e.to_string())?
        .compile(
            &selected,
            73,
            &[FrameResourceBinding {
                key,
                frame: 73,
                version: 0,
                physical: PhysicalResource::Texture(texture),
                layout: template.layout,
                alignment: 256,
            }],
            &[],
            GraphLimits::default(),
        )
        .map_err(|e| e.to_string())?
        .prepare_runtime(
            Extent3d::new_2d(4, 2),
            &[RuntimeImport::Texture {
                resource: prism_render_architecture::frame_graph::ResourceId(0),
                imported: ImportedTexture {
                    texture,
                    default_view: view,
                    entry_state: TextureState::undefined(),
                    exit_state: TextureState::undefined(),
                    mip_levels: 2,
                    array_layers: 2,
                },
            }],
            vec![RuntimePassBinding::new(PassKey(1), |_| {})],
        )
        .map_err(|e| e.to_string())?;
    RuntimePreviewFrame::record(
        graph,
        device,
        queue,
        GraphCaptureInputs::default(),
        GraphCaptureLimits::default(),
    )
    .map_err(|e| e.to_string())
}

fn gpu() -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = futures_lite::future::block_on(instance.request_adapter(&Default::default()))
        .expect("runtime preview requires a real adapter");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&Default::default()))
            .expect("runtime preview device");
    (adapter, device, queue)
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let stride = (texture.width() * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("runtime preview native composite readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks(stride as usize)
        .flat_map(|row| row[..texture.width() as usize * 4].iter().copied())
        .collect()
}

#[test]
fn native_runtime_capture_drains_and_composites_real_frame_pixels() {
    let runtime = WgpuInstance::new()
        .request_default_hardware_device()
        .expect("runtime graph requires a real render device");
    let (runtime_device, runtime_queue) = runtime;
    let mut source = Source::new(runtime_device.clone(), runtime_queue);
    let mut initial = None;
    source
        .capture(&mut |frame| {
            initial = Some(frame.report().clone());
            Ok(())
        })
        .unwrap();
    let mut panel = DesktopFrameGraphPanel::new(FrameGraphPanelController::default(), |_| {
        Ok((None, DesktopPanelUpdate::default()))
    });
    panel.controller.publish(initial).unwrap();
    panel
        .controller
        .apply(FrameGraphPanelAction::Preview {
            pass: 0,
            resource: 0,
        })
        .unwrap();
    panel.build().unwrap();
    let captures = source.captures.clone();
    panel
        .mount_runtime_preview(
            Box::new(source),
            prism_editor_profiler::gpu_preview::TexturePreviewLimits {
                max_in_flight: 1,
                max_bytes_per_texture: 512,
                max_total_bytes: 512,
                max_pixels: 64,
            },
        )
        .unwrap();
    panel
        .event("frame-graph:preview:capture", &UiEvent::Activate)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while panel.controller.preview().build().image().is_none() {
        runtime_device
            .wgpu_device()
            .poll(wgpu::PollType::Poll)
            .unwrap();
        panel.poll(Instant::now()).unwrap();
        assert!(Instant::now() < deadline, "runtime preview did not publish");
        std::thread::yield_now();
    }
    assert_eq!(captures.load(Ordering::Relaxed), 2);
    let first = panel.controller.preview().build().image().unwrap().clone();
    assert_eq!(first.rgba8(), [0, 255, 0, 255].repeat(8));

    let (adapter, device, queue) = gpu();
    let gpu = DesktopGpu {
        adapter,
        device,
        queue,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    let mut images = DesktopContentImages::new(&gpu).unwrap();
    let preview = panel.controller.build().preview().unwrap().clone();
    let native = DesktopFrameGraphImage {
        view: preview,
        rect: Rect::new(Point::ZERO, Size::new(4., 2.)),
        clip: Rect::new(Point::ZERO, Size::new(4., 2.)),
    };
    images
        .prepare_with_preview(&[], Some(&native), 1., [4, 2])
        .unwrap();
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runtime preview native target"),
        size: wgpu::Extent3d {
            width: 4,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let attachments = [Some(wgpu::RenderPassColorAttachment {
        view: &target_view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    })];
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &attachments,
            ..Default::default()
        });
    }
    images.encode(&mut encoder, &target).unwrap();
    gpu.queue.submit([encoder.finish()]);
    let pixels = readback(&gpu.device, &gpu.queue, &target);
    assert_eq!(&pixels[0..4], &[0, 255, 0, 255]);
    assert_eq!(images.texture_uploads(), 1);
    images
        .prepare_with_preview(&[], Some(&native), 2., [8, 4])
        .unwrap();
    assert_eq!(
        images.texture_uploads(),
        1,
        "zoom geometry must reuse pixels"
    );

    panel
        .event("frame-graph:preview:capture", &UiEvent::Activate)
        .unwrap();
    panel.poll(Instant::now()).unwrap();
    assert_eq!(panel.runtime_previews_pending(), 1);
    panel.cancel_runtime_preview().unwrap();
    while panel.runtime_previews_pending() != 0 {
        runtime_device
            .wgpu_device()
            .poll(wgpu::PollType::Poll)
            .unwrap();
        panel.poll(Instant::now()).unwrap();
        assert!(
            Instant::now() < deadline,
            "cancelled runtime preview did not drain"
        );
        std::thread::yield_now();
    }
    assert!(panel.controller.preview().build().image().is_none());
}
