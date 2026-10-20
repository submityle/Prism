//! Hardware proof of exact pass-point color/depth/HDR texture readback.

use prism_editor_profiler::{
    capture_feature_graph, gpu_preview::*, GraphCaptureInputs, GraphCaptureLimits,
    GraphCaptureReport,
};
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceLifetime},
    render_fabric::*,
};
use prism_render_graph::capacity::GraphLimits;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(10)),
        }
    }
}
fn table<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}
fn report(format: TextureFormat, extent: [u32; 3], mip_levels: u32) -> Arc<GraphCaptureReport> {
    let feature = FeatureId(701);
    let key = ResourceKey::of(feature, 0);
    let depth = format == TextureFormat::Depth32Float;
    let passes = table(vec![PassTemplate {
        id: PassKey(3),
        queue: QueueClass::Graphics,
        depends_on: &[],
        accesses: table(vec![ResourceUse {
            key,
            access: if depth {
                AccessKind::DepthAttachment
            } else {
                AccessKind::ColorAttachment
            },
            scope: ResourceScope::Whole,
            frame_lag: None,
            input: ResourceInput::Automatic,
        }]),
    }]);
    let variants = table(vec![FeatureVariant {
        id: VariantId(1),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes,
        shader_permutation: None,
    }]);
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            feature,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            variants,
        )]),
        table(vec![key]),
    )
    .unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, table(vec![feature]), &[]).unwrap();
    let selected = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    let templates = ResourceTemplates::new(
        &catalog,
        &[ResourceTemplate {
            key,
            lifetime: ResourceLifetime::Transient,
            alignment: 256,
            layout: ResourceLayout::Texture {
                extent,
                dimension: TextureDimension::D2,
                format,
                mip_levels,
                samples: 1,
                usage: TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT,
            },
        }],
    )
    .unwrap();
    let graph = templates
        .compile(&selected, 77, &[], &[], GraphLimits::default())
        .unwrap();
    Arc::new(
        capture_feature_graph(
            &graph,
            GraphCaptureInputs::default(),
            GraphCaptureLimits::default(),
        )
        .unwrap(),
    )
}
fn binding(report: Arc<GraphCaptureReport>) -> TexturePreviewBinding {
    TexturePreviewBinding::new(report, 0, 0).unwrap()
}
fn texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    extent: [u32; 3],
    mip_level_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("real preview texture"),
        size: wgpu::Extent3d {
            width: extent[0],
            height: extent[1],
            depth_or_array_layers: extent[2],
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}
fn clear(encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, color: wgpu::Color) {
    let view = texture.create_view(&Default::default());
    let attachments = [Some(wgpu::RenderPassColorAttachment {
        view: &view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(color),
            store: wgpu::StoreOp::Store,
        },
    })];
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &attachments,
        ..Default::default()
    });
}
fn finish(
    producer: &mut TexturePreviewProducer,
    device: &wgpu::Device,
    handle: prism_editor_profiler::capture::CaptureHandle,
) -> TexturePreview {
    let submission = producer.submit(handle).unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    let mut results = producer.poll().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].handle, handle);
    let preview = results.remove(0).preview.unwrap();
    assert_eq!(producer.allocated_bytes(), 0);
    assert_eq!(producer.occupied_slots(), 0);
    preview
}

#[test]
fn actual_pass_snapshot_padding_mips_hdr_depth_and_cancelled_readbacks() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&Default::default()))
        .expect("real GPU required; no silent skip");
    let info = adapter.get_info();
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
    eprintln!("REAL GPU texture preview adapter={info:?}");
    let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
    let limits = TexturePreviewLimits {
        max_in_flight: 1,
        max_bytes_per_texture: 4096,
        max_total_bytes: 4096,
        max_pixels: 1024,
    };
    let mut producer = TexturePreviewProducer::new(&device, &queue, limits).unwrap();
    let capture = report(TextureFormat::Rgba8Unorm, [3, 2, 1], 1);
    let selected = binding(capture.clone());
    assert!(selected.matches(&capture, 0, 0));
    assert!(!selected.matches(&Arc::new((*capture).clone()), 0, 0));
    assert!(matches!(
        TexturePreviewBinding::new(capture.clone(), 0, 1),
        Err(TexturePreviewError::MissingAccess)
    ));
    let color = texture(&device, wgpu::TextureFormat::Rgba8Unorm, [3, 2, 1], 1);
    let handle = producer.begin(selected).unwrap();
    assert_eq!(
        producer.begin(binding(capture.clone())),
        Err(TexturePreviewError::AtCapacity)
    );
    assert!(matches!(
        producer.submit(handle),
        Err(TexturePreviewError::InvalidState)
    ));
    clear(producer.encoder(handle).unwrap(), &color, wgpu::Color::RED);
    producer.copy_texture(handle, &color, 0, 0).unwrap();
    assert_eq!(producer.allocated_bytes(), 512);
    assert_eq!(
        producer.copy_texture(handle, &color, 0, 0),
        Err(TexturePreviewError::InvalidState)
    );
    // Reusing the actual texture after the selected pass cannot mutate its snapshot.
    clear(
        producer.encoder(handle).unwrap(),
        &color,
        wgpu::Color::GREEN,
    );
    let result = finish(&mut producer, &device, handle);
    assert_eq!(result.bytes(), [255, 0, 0, 255].repeat(6));
    assert_eq!(result.pixel(2, 1), Some([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(result.pixel(3, 1), None);
    assert_eq!(result.rgba8(Default::default()).unwrap(), result.bytes());
    assert_eq!(
        producer.cancel(handle),
        Err(TexturePreviewError::StaleHandle)
    );
    assert!(result.binding().matches(&capture, 0, 0));
    assert_eq!(result.format(), wgpu::TextureFormat::Rgba8Unorm);
    for display in [
        TexturePreviewDisplay {
            range: [-f64::MAX, f64::MAX],
            ..Default::default()
        },
        TexturePreviewDisplay {
            exposure_ev: f32::NAN,
            ..Default::default()
        },
        TexturePreviewDisplay {
            channel: Some(4),
            ..Default::default()
        },
    ] {
        assert_eq!(
            result.rgba8(display),
            Err(TexturePreviewError::InvalidConfig)
        );
    }

    // Enforce pixel and padded byte limits before recording a copy/allocation.
    for constrained in [
        TexturePreviewLimits {
            max_pixels: 5,
            ..limits
        },
        TexturePreviewLimits {
            max_bytes_per_texture: 256,
            ..limits
        },
    ] {
        let mut bounded = TexturePreviewProducer::new(&device, &queue, constrained).unwrap();
        let h = bounded.begin(binding(capture.clone())).unwrap();
        assert_eq!(
            bounded.copy_texture(h, &color, 0, 0),
            Err(TexturePreviewError::AtCapacity)
        );
        assert_eq!(bounded.allocated_bytes(), 0);
        bounded.cancel(h).unwrap();
        assert_eq!(bounded.occupied_slots(), 0);
    }
    let mut bounded = TexturePreviewProducer::new(
        &device,
        &queue,
        TexturePreviewLimits {
            max_in_flight: 2,
            max_bytes_per_texture: 512,
            max_total_bytes: 512,
            ..limits
        },
    )
    .unwrap();
    let a = bounded.begin(binding(capture.clone())).unwrap();
    let b = bounded.begin(binding(capture.clone())).unwrap();
    assert_eq!(
        producer.copy_texture(a, &color, 0, 0),
        Err(TexturePreviewError::StaleHandle)
    );
    bounded.copy_texture(a, &color, 0, 0).unwrap();
    assert_eq!(
        bounded.copy_texture(b, &color, 0, 0),
        Err(TexturePreviewError::AtCapacity)
    );
    assert_eq!(bounded.allocated_bytes(), 512);
    bounded.cancel(a).unwrap();
    bounded.copy_texture(b, &color, 0, 0).unwrap();
    bounded.cancel(b).unwrap();
    assert_eq!(bounded.allocated_bytes(), 0);
    assert_eq!(bounded.occupied_slots(), 0);

    // IDs above 2^24 must remain exact when inspecting actual integer texels.
    let ids = texture(&device, wgpu::TextureFormat::R32Uint, [3, 1, 1], 1);
    let id_values = [16_777_217_u32, u32::MAX, 0];
    let id_bytes: Vec<u8> = id_values.into_iter().flat_map(u32::to_le_bytes).collect();
    queue.write_texture(
        ids.as_image_copy(),
        &id_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(12),
            rows_per_image: Some(1),
        },
        ids.size(),
    );
    let h = producer
        .begin(binding(report(TextureFormat::R32Uint, [3, 1, 1], 1)))
        .unwrap();
    producer.copy_texture(h, &ids, 0, 0).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!(p.bytes(), id_bytes);
    for (x, value) in id_values.into_iter().enumerate() {
        assert_eq!(
            p.pixel(x as u32, 0),
            Some([f64::from(value), 0.0, 0.0, 1.0])
        );
    }
    assert_eq!(
        p.rgba8(TexturePreviewDisplay {
            channel: Some(0),
            range: [16_777_216.0, 16_777_218.0],
            ..Default::default()
        })
        .unwrap(),
        [128, 128, 128, 255, 255, 255, 255, 255, 0, 0, 0, 255]
    );

    // Invalid GPU floats are visible diagnostics, never silently displayed as zero.
    let floats = texture(&device, wgpu::TextureFormat::R32Float, [4, 1, 1], 1);
    let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25];
    let raw: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    queue.write_texture(
        floats.as_image_copy(),
        &raw,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        floats.size(),
    );
    let h = producer
        .begin(binding(report(TextureFormat::R32Float, [4, 1, 1], 1)))
        .unwrap();
    producer.copy_texture(h, &floats, 0, 0).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!(p.bytes(), raw);
    assert_eq!(p.pixel(3, 0), Some([-0.25, 0.0, 0.0, 1.0]));
    let shown = p
        .rgba8(TexturePreviewDisplay {
            channel: Some(0),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(&shown[..12], [255, 0, 255, 255].repeat(3));
    assert_eq!(&shown[12..], [0, 0, 0, 255]);

    // Real half-float render target, with values outside the display range.
    let hdr = texture(&device, wgpu::TextureFormat::Rgba16Float, [2, 2, 1], 1);
    let h = producer
        .begin(binding(report(TextureFormat::Rgba16Float, [2, 2, 1], 1)))
        .unwrap();
    clear(
        producer.encoder(h).unwrap(),
        &hdr,
        wgpu::Color {
            r: 4.0,
            g: 0.5,
            b: 0.0,
            a: 1.0,
        },
    );
    producer.copy_texture(h, &hdr, 0, 0).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!(p.pixel(0, 0), Some([4.0, 0.5, 0.0, 1.0]));
    let displayed = p
        .rgba8(TexturePreviewDisplay {
            reinhard: true,
            ..Default::default()
        })
        .unwrap();
    assert!(displayed[0] > displayed[1] && displayed[0] < 255);
    assert_eq!(&displayed[2..4], &[0, 255]);
    let boundary_bits = [0x0001_u16, 0x03ff, 0x0400, 0x7bff];
    let raw: Vec<u8> = boundary_bits
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    queue.write_texture(
        hdr.as_image_copy(),
        &raw,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let h = producer
        .begin(binding(report(TextureFormat::Rgba16Float, [2, 2, 1], 1)))
        .unwrap();
    producer.copy_texture(h, &hdr, 0, 0).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!(
        p.pixel(0, 0),
        Some([
            2.0_f64.powi(-24),
            1023.0 * 2.0_f64.powi(-24),
            2.0_f64.powi(-14),
            65504.0
        ])
    );

    // Depth contents are captured from a real depth attachment, not color metadata.
    let depth = texture(&device, wgpu::TextureFormat::Depth32Float, [3, 2, 1], 1);
    let h = producer
        .begin(binding(report(TextureFormat::Depth32Float, [3, 2, 1], 1)))
        .unwrap();
    let view = depth.create_view(&Default::default());
    {
        let _pass = producer
            .encoder(h)
            .unwrap()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.25),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
    }
    producer.copy_texture(h, &depth, 0, 0).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!(p.pixel(2, 1), Some([0.25, 0.0, 0.0, 1.0]));
    assert_eq!(
        p.rgba8(TexturePreviewDisplay {
            channel: Some(0),
            ..Default::default()
        })
        .unwrap(),
        [64, 64, 64, 255].repeat(6)
    );

    // Array layers and mip dimensions use real uploaded texels; padding is stripped.
    let array = texture(&device, wgpu::TextureFormat::Bgra8UnormSrgb, [8, 4, 2], 3);
    let capture = report(TextureFormat::Bgra8UnormSrgb, [8, 4, 2], 3);
    let bytes = [128, 64, 32, 255].repeat(2);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &array,
            mip_level: 2,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let h = producer.begin(binding(capture.clone())).unwrap();
    assert_eq!(
        producer.copy_texture(h, &color, 0, 0),
        Err(TexturePreviewError::LayoutMismatch)
    );
    assert_eq!(
        producer.copy_texture(h, &array, 3, 0),
        Err(TexturePreviewError::InvalidSubresource)
    );
    assert_eq!(producer.allocated_bytes(), 0);
    producer.copy_texture(h, &array, 2, 1).unwrap();
    let p = finish(&mut producer, &device, h);
    assert_eq!((p.width(), p.height(), p.mip(), p.layer()), (2, 1, 2, 1));
    assert_eq!(p.bytes(), bytes);
    assert_eq!(
        p.rgba8(Default::default()).unwrap(),
        [32, 64, 128, 255].repeat(2)
    );

    // A submitted cancellation cannot free capacity before its map callback.
    let h = producer.begin(binding(capture.clone())).unwrap();
    producer.copy_texture(h, &array, 2, 1).unwrap();
    let submission = producer.submit(h).unwrap();
    producer.cancel(h).unwrap();
    assert_eq!(producer.allocated_bytes(), 256);
    assert_eq!(
        producer.begin(binding(capture.clone())),
        Err(TexturePreviewError::AtCapacity)
    );
    assert_eq!(producer.cancel(h), Err(TexturePreviewError::StaleHandle));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    assert!(producer.poll().unwrap().is_empty());
    assert_eq!(producer.allocated_bytes(), 0);
    let h = producer.begin(binding(capture)).unwrap();
    producer.cancel(h).unwrap();
    assert!(matches!(
        producer.encoder(h),
        Err(TexturePreviewError::StaleHandle)
    ));
}
