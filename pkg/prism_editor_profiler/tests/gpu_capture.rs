//! Opt-in hardware tests: absence of a real timestamp-capable GPU is a failure.

use prism_editor_profiler::{
    capture::CaptureError,
    gpu::{GpuCaptureConfig, GpuProducer, REQUIRED_TIMESTAMP_FEATURES},
    FrameId, FrameStore, ProfilerLimits,
};
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
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(10)),
        }
    }
}
fn wait(device: &wgpu::Device, submission: wgpu::SubmissionIndex) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .expect("real GPU submission must complete");
}

#[test]
fn real_gpu_timestamp_readback_cancellation_and_generation_fence() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("gpu-tests requires a real GPU adapter; no silent skip");
    let info = adapter.get_info();
    assert_ne!(
        info.device_type,
        wgpu::DeviceType::Cpu,
        "software adapter is not real GPU evidence"
    );
    assert!(
        adapter.features().contains(REQUIRED_TIMESTAMP_FEATURES),
        "hardware timestamp features unavailable: {info:?}"
    );
    eprintln!("REAL GPU adapter={info:?}; required={REQUIRED_TIMESTAMP_FEATURES:?}");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: REQUIRED_TIMESTAMP_FEATURES,
        ..Default::default()
    }))
    .unwrap();
    let limits = ProfilerLimits {
        max_frames: 2,
        max_spans_per_frame: 2,
        max_span_depth: 2,
        ..ProfilerLimits::default()
    };
    let config = GpuCaptureConfig {
        max_in_flight_frames: 1,
        max_spans_per_frame: 2,
    };
    let mut gpu = GpuProducer::new(&device, &queue, limits, config).unwrap();
    let discarded = gpu.begin_frame(FrameId::from_raw(99), 99).unwrap();
    gpu.cancel(discarded).unwrap();
    assert_eq!(gpu.occupied_slots(), 0);
    assert!(matches!(
        gpu.encoder(discarded),
        Err(CaptureError::StaleHandle)
    ));
    let handle = gpu.begin_frame(FrameId::from_raw(1), 1).unwrap();
    assert_ne!(discarded, handle);
    assert_eq!(
        gpu.begin_frame(FrameId::from_raw(2), 2),
        Err(CaptureError::AtCapacity)
    );
    assert!(matches!(
        gpu.submit_frame(handle),
        Err(CaptureError::InvalidState)
    ));

    // Real compute work and independently checked output prove the measured
    // encoder was submitted and executed; no fabricated timestamp inputs.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("profiler real workload"), source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { var x = id.x; for (var i = 0u; i < 2048u; i++) { x = x * 1664525u + 1013904223u; } output[id.x] = x; }".into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("profiler real workload"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("workload output"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("workload proof"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
    {
        let mut pass = gpu
            .begin_compute_pass(
                handle,
                "compute workload",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    // Exercise the same encoder API used by an app rendering a real viewport.
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("profiler render proof"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("profiler draw proof"),
        source: wgpu::ShaderSource::Wgsl("@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { var p = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0)); return vec4<f32>(p[i],0.0,1.0); } @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0,1.0,0.0,1.0); }".into()),
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("profiler actual draw"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = gpu
            .begin_render_pass(
                handle,
                "render clear",
                &wgpu::RenderPassDescriptor {
                    label: Some("profiler actual render pass"),
                    color_attachments: &attachments,
                    ..Default::default()
                },
            )
            .unwrap();
        pass.set_pipeline(&render_pipeline);
        pass.draw(0..3, 0..1);
    }
    gpu.encoder(handle)
        .unwrap()
        .copy_buffer_to_buffer(&output, 0, &readback, 0, 256 * 4);
    assert!(gpu
        .begin_compute_pass(
            handle,
            "over limit",
            &wgpu::ComputePassDescriptor::default()
        )
        .is_err());
    let submission = gpu.submit_frame(handle).unwrap();
    eprintln!(
        "REAL GPU submission={submission:?}; timestamp_period_ns={}",
        queue.get_timestamp_period()
    );
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.try_send(result).unwrap();
        });
    wait(&device, submission);
    receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    {
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        for (id, chunk) in bytes.chunks_exact(4).enumerate() {
            let mut expected = id as u32;
            for _ in 0..2048 {
                expected = expected.wrapping_mul(1664525).wrapping_add(1013904223);
            }
            assert_eq!(u32::from_ne_bytes(chunk.try_into().unwrap()), expected);
        }
    }
    readback.unmap();
    let completed = gpu.poll().unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].handle, handle);
    let frame = completed.into_iter().next().unwrap().frame.unwrap();
    assert!(
        frame.duration_ns() > 0,
        "real GPU must return nonzero elapsed ticks"
    );
    assert!(frame.spans().iter().all(|s| s.duration_ns() > 0));
    assert_eq!(frame.span_tree().roots().len(), 2);
    eprintln!(
        "REAL GPU frame duration_ns={}; spans={:?}; compute output verified",
        frame.duration_ns(),
        frame.spans()
    );
    let mut store = FrameStore::new(limits);
    store.insert(frame).unwrap();
    assert!(gpu.poll().unwrap().is_empty());
    assert_eq!(gpu.cancel(handle), Err(CaptureError::StaleHandle));

    // Cancel after submission: fixed pool remains full until old callback drains.
    let old = gpu.begin_frame(FrameId::from_raw(2), 2).unwrap();
    {
        let mut pass = gpu
            .begin_compute_pass(
                old,
                "cancelled workload",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    let submission = gpu.submit_frame(old).unwrap();
    gpu.cancel(old).unwrap();
    assert_eq!(gpu.occupied_slots(), 1);
    assert_eq!(gpu.cancel(old), Err(CaptureError::StaleHandle));
    assert_eq!(
        gpu.begin_frame(FrameId::from_raw(3), 3),
        Err(CaptureError::AtCapacity)
    );
    wait(&device, submission);
    assert!(
        gpu.poll().unwrap().is_empty(),
        "cancelled generation must never reach store"
    );
    assert_eq!(gpu.occupied_slots(), 0);
    let next = gpu.begin_frame(FrameId::from_raw(3), 3).unwrap();
    assert_ne!(next, old);
    assert!(matches!(gpu.encoder(old), Err(CaptureError::StaleHandle)));
    {
        let mut pass = gpu
            .begin_compute_pass(
                next,
                "next generation workload",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    let submission = gpu.submit_frame(next).unwrap();
    wait(&device, submission);
    let result = gpu.poll().unwrap().pop().unwrap();
    assert_eq!(result.handle, next);
    store.insert(result.frame.unwrap()).unwrap();
    assert_eq!(store.frames().len(), 2);
    // On Metal, an empty pass may omit its samples. Reused slots must never
    // return the previous generation's timestamps for that empty pass.
    let empty = gpu.begin_frame(FrameId::from_raw(4), 4).unwrap();
    drop(
        gpu.begin_compute_pass(empty, "empty", &wgpu::ComputePassDescriptor::default())
            .unwrap(),
    );
    let submission = gpu.submit_frame(empty).unwrap();
    wait(&device, submission);
    let empty_result = gpu.poll().unwrap().pop().unwrap();
    if info.backend == wgpu::Backend::Metal && info.name == "Apple M2" {
        assert!(matches!(
            empty_result.frame,
            Err(CaptureError::InvalidTimestamp)
        ));
    }
    let mut other = GpuProducer::new(&device, &queue, limits, config).unwrap();
    let draining = gpu.begin_frame(FrameId::from_raw(5), 5).unwrap();
    assert!(matches!(
        other.encoder(draining),
        Err(CaptureError::StaleHandle)
    ));
    {
        let mut pass = gpu
            .begin_compute_pass(
                draining,
                "cancel all",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    let submission = gpu.submit_frame(draining).unwrap();
    gpu.cancel_all();
    assert_eq!(gpu.occupied_slots(), 1);
    assert!(matches!(
        gpu.encoder(draining),
        Err(CaptureError::StaleHandle)
    ));
    wait(&device, submission);
    assert!(gpu.poll().unwrap().is_empty());
    assert_eq!(gpu.occupied_slots(), 0);

    // Dropping a submitted producer must not let its late callback touch a
    // new producer or block device polling.
    let dropped = other.begin_frame(FrameId::from_raw(6), 6).unwrap();
    {
        let mut pass = other
            .begin_compute_pass(
                dropped,
                "drop in flight",
                &wgpu::ComputePassDescriptor::default(),
            )
            .unwrap();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    let submission = other.submit_frame(dropped).unwrap();
    drop(other);
    wait(&device, submission);

    // No features enabled is an explicit capability refusal, on a real device.
    let (plain_device, plain_queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    assert!(matches!(
        GpuProducer::new(&plain_device, &plain_queue, limits, config),
        Err(CaptureError::UnsupportedTimestamps)
    ));
    assert!(matches!(
        GpuProducer::new(
            &device,
            &queue,
            limits,
            GpuCaptureConfig {
                max_in_flight_frames: 0,
                ..config
            }
        ),
        Err(CaptureError::InvalidConfig)
    ));
    assert!(matches!(
        GpuProducer::new(
            &device,
            &queue,
            limits,
            GpuCaptureConfig {
                max_spans_per_frame: usize::MAX,
                ..config
            }
        ),
        Err(CaptureError::InvalidConfig)
    ));
    eprintln!("REAL GPU cancellation fence, slot reuse and disabled-feature refusal verified");
}
