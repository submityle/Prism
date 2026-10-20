//! Opt-in real hardware timestamps feeding the interactive Loom timeline.
#![cfg(feature = "profiler-gpu-tests")]

use prism_editor_profiler::{
    gpu::{GpuCaptureConfig, GpuProducer, REQUIRED_TIMESTAMP_FEATURES},
    FrameId, FrameStore, ProfilerLimits,
};
use prism_editor_ui::{ProfilerController, ProfilerSources, UiEvent};
use prism_ui::{
    layout::{AvailableSpace, Size},
    Ui,
};
use prism_ui_render_backend::{rasterize, RetainedScene};
use std::{
    sync::{mpsc, Arc},
    time::Duration,
};

#[test]
fn hardware_query_readback_drives_actual_selectable_timeline() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = futures_lite::future::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .expect("a real timestamp-capable GPU is required; no skip");
    let info = adapter.get_info();
    assert_ne!(
        info.device_type,
        wgpu::DeviceType::Cpu,
        "software adapters are not hardware proof"
    );
    assert!(
        adapter.features().contains(REQUIRED_TIMESTAMP_FEATURES),
        "timestamp capabilities unavailable: {info:?}"
    );
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: REQUIRED_TIMESTAMP_FEATURES,
            ..Default::default()
        }))
        .unwrap();
    let mut gpu = GpuProducer::new(
        &device,
        &queue,
        ProfilerLimits::default(),
        GpuCaptureConfig {
            max_in_flight_frames: 1,
            max_spans_per_frame: 1,
        },
    )
    .unwrap();
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("profiler UI real workload"),source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> output:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {var x=id.x; for(var i=0u;i<2048u;i++){x=x*1664525u+1013904223u;} output[id.x]=x;}".into())});
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let handle = gpu.begin_frame(FrameId::from_raw(42), 42).unwrap();
    {
        let mut pass = gpu
            .begin_compute_pass(
                handle,
                "app measured GPU pass",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    gpu.encoder(handle)
        .unwrap()
        .copy_buffer_to_buffer(&output, 0, &readback, 0, 256 * 4);
    let submission = gpu.submit_frame(handle).unwrap();
    let (tx, rx) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.try_send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
    {
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let mut expected = i as u32;
            for _ in 0..2048 {
                expected = expected.wrapping_mul(1664525).wrapping_add(1013904223);
            }
            assert_eq!(u32::from_ne_bytes(chunk.try_into().unwrap()), expected);
        }
    }
    readback.unmap();
    let mut completed = gpu.poll().unwrap();
    assert_eq!(completed.len(), 1);
    let frame = completed.remove(0).frame.unwrap();
    assert!(frame.duration_ns() > 0 && frame.spans()[0].duration_ns() > 0);
    eprintln!(
        "REAL GPU UI adapter={info:?}; frame_ns={}; span_ns={}; compute output verified",
        frame.duration_ns(),
        frame.spans()[0].duration_ns()
    );
    let mut store = FrameStore::new(ProfilerLimits::default());
    store.insert(frame).unwrap();
    let mut c = ProfilerController::default();
    c.publish(ProfilerSources {
        cpu: None,
        gpu: Some(Arc::new(store)),
    });
    let empty = c.build();
    c.handle_event(&empty, "profiler:gpu", &UiEvent::Activate)
        .unwrap();
    let view = c.build();
    assert_eq!(view.spans().len(), 1);
    let span = view.spans()[0].clone();
    assert!(span.width > 0.0);
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(900.0),
        AvailableSpace::MaxContent,
    ));
    let pixels = rasterize(&ui.backend().to_draw_list(), 900, 1200);
    assert!(
        (0..1200).any(|y| (0..900).any(|x| {
            let p = pixels.pixel(x, y);
            (p[0] - 0.5).abs() < 0.01 && (p[1] - 0.8).abs() < 0.01 && (p[2] - 0.3).abs() < 0.01
        })),
        "actual hardware span must paint GPU green pixels"
    );
    c.handle_event(&view, &span.key, &UiEvent::Activate)
        .unwrap();
    assert_eq!(c.selected_span(), Some(span.span));
    let view = c.build();
    c.handle_event(&view, "profiler:fit-span", &UiEvent::Activate)
        .unwrap();
    assert!(c.build().spans()[0].width > 0.99);
}
